import argparse
import grpc
from concurrent import futures
import vad_service_pb2
import vad_service_pb2_grpc
import cv2
import torch
import numpy as np
import yaml
from pathlib import Path
import uuid
from typing import Dict, Any

from mmcv.parallel.data_container import DataContainer

import sys
sys.path.append('')
import numpy as np
import argparse
import mmcv
import os
import copy
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
# from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.VAD.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import json


class VehicleParams:
   def __init__(self, vehicle_info_path: str = None):
       self.vehicle_info_path = vehicle_info_path
       self.params = self._load_vehicle_params()

   def _load_vehicle_params(self) -> Dict[str, Any]:
       """車両パラメータをYAMLファイルから読み込む"""
       yaml_path = Path(self.vehicle_info_path)
       if not yaml_path.exists():
           raise ValueError(f"Vehicle info file not found: {yaml_path}")
       
       with open(yaml_path, 'r') as f:
           params = yaml.safe_load(f)
       return params['vehicle_params']

   @property
   def wheel_base(self) -> float:
       return self.params['wheel_base']

   @property
   def vehicle_length(self) -> float:
       return self.params['vehicle_length']

   @property
   def vehicle_width(self) -> float:
       return self.params['vehicle_width']

   @property
   def max_steer_angle(self) -> float:
       return self.params['max_steer_angle']

class VADWrapper:
    def __init__(self, vad_config_path = "/workspace/VAD/projects/configs/VAD/VAD_base_e2e.py", checkpoint_path = "/workspace/VAD/ckpts/VAD_base.pth"):
        cfg = Config.fromfile(vad_config_path)
        # import modules from string list.
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])

        # import modules from plguin/xx, registry will be updated
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)
                # else:
                #     # import dir is the dirpath for the config file
                #     _module_dir = os.path.dirname(args.config)
                #     _module_dir = _module_dir.split('/')
                #     _module_path = _module_dir[0]
                #     for m in _module_dir[1:]:
                #         _module_path = _module_path + '.' + m
                #     print(_module_path)
                #     plg_lib = importlib.import_module(_module_path)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        cfg.model.pretrained = None
        # in case the test dataset is concatenated
        samples_per_gpu = 1
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            samples_per_gpu = max(
                [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
            if samples_per_gpu > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)


        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']

        self.model = MMDataParallel(model, device_ids=[0])
        self.cfg = cfg



class VADDummy:
    def __init__(self):
        self.setup_model()
    
    def setup_model(self):
        # Dummy model setup
        pass

    def __call__(self, return_loss=False, rescale=True, **kwargs):
        """VADモデルと同じインターフェースを提供"""
        return self.forward_test(**kwargs)

    def forward_test(self, **kwargs):

        if not isinstance(kwargs["img"][0], DataContainer):
            raise ValueError("Input image should be wrapped in DataContainer")
        
        if 'ego_lcf_feat' in kwargs:
            ego_lcf = kwargs['ego_lcf_feat'][0]
            if not isinstance(ego_lcf, DataContainer):
                raise ValueError("ego_lcf_feat should be wrapped in DataContainer")
            
            feat_tensor = ego_lcf.data
            if feat_tensor.shape != (1, 1, 1, 9):
                raise ValueError(f"Expected ego_lcf_feat shape (1, 1, 1, 9), got {feat_tensor.shape}")
            
        if 'ego_fut_cmd' in kwargs:
            ego_cmd_list = kwargs['ego_fut_cmd']
            if not isinstance(ego_cmd_list, list) or len(ego_cmd_list) != 1:
                raise ValueError("ego_fut_cmd should be a list with single element")
                
            ego_cmd = ego_cmd_list[0]
            if not isinstance(ego_cmd, DataContainer):
                raise ValueError("ego_fut_cmd element should be wrapped in DataContainer")
            
            cmd_tensor = ego_cmd.data
            if cmd_tensor.shape != (1, 1, 1, 3):
                raise ValueError(f"Expected ego_fut_cmd shape (1, 1, 1, 3), got {cmd_tensor.shape}")
            
            # one-hotベクトルの検証
            if not torch.allclose(cmd_tensor.sum(), torch.tensor(1.0)):
                raise ValueError("ego_fut_cmd should be a one-hot vector")
            
        # ego_his_trajsの検証
        if 'ego_his_trajs' in kwargs:
            ego_his_trajs_list = kwargs['ego_his_trajs']
            if not isinstance(ego_his_trajs_list, list) or len(ego_his_trajs_list) != 1:
                raise ValueError("ego_his_trajs should be a list with single element")
            
            ego_his_trajs = ego_his_trajs_list[0]
            if not isinstance(ego_his_trajs, DataContainer):
                raise ValueError("ego_his_trajs element should be wrapped in DataContainer")
            
            trajs_tensor = ego_his_trajs.data
            if trajs_tensor.shape != (1, 1, 2, 2):
                raise ValueError(f"Expected ego_his_trajs shape (1, 1, 2, 2), got {trajs_tensor.shape}")
            
            # テンソルの値の妥当性を確認（オプション）
            if torch.isnan(trajs_tensor).any():
                raise ValueError("ego_his_trajs contains NaN values")

        # ダミーの予測結果を生成
        ego_fut_preds = torch.zeros(3, 6, 2)  # [3フレーム, 6パターン, (x, y)]
        
        # 単純な直線的な軌道を生成
        for i in range(6):  # 6つの予測パターン
            for t in range(3):  # 3フレーム
                ego_fut_preds[t, i, 0] = t * (i + 1)  # x座標
                ego_fut_preds[t, i, 1] = t * (i + 1)  # y座標

        # VADモデルの出力形式に合わせる
        return [{
            'pts_bbox': {
                'ego_fut_preds': ego_fut_preds
            }
        }]

class VADServicer(vad_service_pb2_grpc.VADServiceServicer):
    def __init__(self, vehicle_info_path: Path, device: str = "cuda:0"):
        self.vad_model = VADDummy()
        self.vad_wrapper = VADWrapper()
        self.vehicle_params = VehicleParams(vehicle_info_path)
        self.device = device

    def create_predicted_object_from_single_trajectory(self, trajectory: torch.Tensor, current_pos: list, timestamp_sec: int, timestamp_nanosec: int):
        """単一の軌道からPredictedObjectを作成する
        Args:
            trajectory: shape [6, 2] のtensor。6つの予測点のx, y座標
            current_pos: 現在位置 [x, y]
            timestamp_sec: タイムスタンプ（秒）
            timestamp_nanosec: タイムスタンプ（ナノ秒）
        """
        obj = vad_service_pb2.PredictedObject(
            uuid=str(uuid.uuid4()),
            existence_probability=0.9
        )
        
        # 現在位置を初期位置として設定
        kinematics = obj.kinematics
        kinematics.initial_pose_with_covariance.pose.position.x = float(current_pos[0])
        kinematics.initial_pose_with_covariance.pose.position.y = float(current_pos[1])
        kinematics.initial_pose_with_covariance.pose.position.z = 0.0
        kinematics.initial_pose_with_covariance.pose.orientation.w = 1.0
        kinematics.initial_pose_with_covariance.covariance.extend([0.0] * 36)
        
        # 単一の予測パスを追加
        path = kinematics.predicted_paths.add()
        
        # 6ステップの予測位置を追加
        for waypoint in trajectory:
            pose = path.path.add()
            # 現在位置からの相対位置として設定
            pose.position.x = float(waypoint[0]) + float(current_pos[0])
            pose.position.y = float(waypoint[1]) + float(current_pos[1])
            pose.position.z = 0.0
            pose.orientation.w = 1.0
        
        # タイムスタンプと信頼度を設定（信頼度は1.0固定）
        path.time_step.sec = timestamp_sec
        path.time_step.nanosec = timestamp_nanosec
        path.confidence = 1.0
        
        return obj

    def create_predicted_object_with_trajectories(self, trajectories: torch.Tensor, ego_fut_cmd_idx: int, current_pos: list, timestamp_sec: int, timestamp_nanosec: int):
        """複数の軌道を1つのPredictedObjectにまとめる"""
        obj = vad_service_pb2.PredictedObject(
            uuid=str(uuid.uuid4()),
            existence_probability=0.9
        )
        
        # 現在位置を初期位置として設定
        kinematics = obj.kinematics
        kinematics.initial_pose_with_covariance.pose.position.x = float(current_pos[0])
        kinematics.initial_pose_with_covariance.pose.position.y = float(current_pos[1])
        kinematics.initial_pose_with_covariance.pose.position.z = 0.0
        kinematics.initial_pose_with_covariance.pose.orientation.w = 1.0
        kinematics.initial_pose_with_covariance.covariance.extend([0.0] * 36)
        
        # 0.6秒分のタイムステップを生成 (0.1秒間隔で6ステップ)
        num_steps = 6
        dt = 0.1  # 0.1秒間隔
        
        # 3つの異なる軌道を生成
        for traj_idx in range(3):
            path = kinematics.predicted_paths.add()
            
            # 基本の軌道を取得
            base_trajectory = trajectories[traj_idx]
            
            # 0.6秒分の予測を生成
            for t in range(num_steps):
                pose = path.path.add()
                
                # 時間ステップに基づいて位置を計算
                t_norm = t / (num_steps - 1)  # 0から1の範囲に正規化
                if t < len(base_trajectory):
                    # 元の予測点を使用
                    dx = float(base_trajectory[t][0])
                    dy = float(base_trajectory[t][1])
                else:
                    # 最後の予測点までの軌道を線形補間
                    last_idx = len(base_trajectory) - 1
                    dx = float(base_trajectory[last_idx][0])
                    dy = float(base_trajectory[last_idx][1])
                
                # 現在位置からの相対位置として設定
                pose.position.x = dx + float(current_pos[0])
                pose.position.y = dy + float(current_pos[1])
                pose.position.z = 0.0
                pose.orientation.w = 1.0
            
            # タイムスタンプと信頼度を設定
            path.time_step.sec = timestamp_sec
            path.time_step.nanosec = timestamp_nanosec
            # ego_fut_cmdに対応するtrajectoryは信頼度1.0、それ以外は0.3
            path.confidence = 1.0 if traj_idx == ego_fut_cmd_idx else 0.3
        
        return obj


    def ProcessData(self, request, context):
        try:
            # カメラ0の時刻情報を取得
            timestamp_sec = request.images[0].time_step_sec
            timestamp_nanosec = request.images[0].time_step_nanosec

            camera_images = {}
            for camera_image in request.images:
                # 画像データをデコード
                nparr = np.frombuffer(camera_image.image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                camera_images[camera_image.camera_id] = image

            resized_images = [cv2.resize(img, (1280, 736), interpolation=cv2.INTER_LINEAR) for img in camera_images.values()]
            
            print("Processing images")
            img_tensor = torch.stack([
                torch.from_numpy(img).permute(2, 0, 1).float() 
                for img in resized_images
            ]).to(self.device)
            img_tensor = img_tensor.view(1, 6, 3, 736, 1280)
            img_container = DataContainer([img_tensor], stack=True, padding_value=0)

            # Odometryから速度と角速度を取得
            latest_odom = request.ego_history[-1]
            ego_vx = latest_odom.twist.twist.linear.x
            ego_vy = latest_odom.twist.twist.linear.y
            ego_w = latest_odom.twist.twist.angular.z

            # IMUから加速度を取得
            if hasattr(request, 'imu_data') and request.imu_data:
                ax = request.imu_data.linear_acceleration.x
                ay = request.imu_data.linear_acceleration.y
            else:
                # IMUデータがない場合はダミー値
                ax, ay = 0.0, 0.0
                self.get_logger().warn("No IMU data available, using dummy values")
            
            # 速度の大きさ
            v0 = np.sqrt(ego_vx**2 + ego_vy**2)
            
            # ステアリング値の取得と曲率計算
            if hasattr(request, 'steering') and request.steering:
                steering = request.steering.steering_tire_angle
                print(f"Using steering angle: {steering} rad")
            else:
                steering = 0.0
                print("No steering data available, using dummy value")
            steering *= -1  # 左ハンドル交通の場合は反転
            Kappa = 2 * np.tan(steering) / self.vehicle_params.wheel_base

            # ego_lcf_featの作成
            ego_lcf_feat_raw = torch.tensor([
                ego_vx, ego_vy,      # 速度
                ax, ay,              # 加速度（IMU）
                ego_w,               # 角速度
                self.vehicle_params.vehicle_length,
                self.vehicle_params.vehicle_width,
                v0,                  # 速度の大きさ
                Kappa                # 曲率
            ], dtype=torch.float32).to(self.device)

            # [1, 1, 1, 9]の形状に変形
            ego_lcf_feat_tensor = ego_lcf_feat_raw.view(1, 1, 1, -1)
            ego_lcf_feat_container = DataContainer(ego_lcf_feat_tensor, stack=True, padding_value=0)

            # can_busの配列を作成（18要素）
            can_bus = np.zeros(18, dtype=np.float32)

            # ego2global_translation
            can_bus[0] = request.can_bus.ego2global_translation.x
            can_bus[1] = request.can_bus.ego2global_translation.y
            can_bus[2] = request.can_bus.ego2global_translation.z

            # ego2global_rotation
            can_bus[3:7] = [
                request.can_bus.ego2global_rotation.x,
                request.can_bus.ego2global_rotation.y,
                request.can_bus.ego2global_rotation.z,
                0.0
            ]

            # 加速度
            can_bus[7] = ax
            can_bus[8] = ay

            # yaw角速度
            can_bus[12] = ego_w

            # 速度
            can_bus[13] = ego_vx
            can_bus[14] = ego_vy

            # patch_angle
            patch_angle = request.can_bus.patch_angle
            can_bus[-2] = patch_angle / 180 * np.pi
            can_bus[-1] = patch_angle

            lidar2img_stacked = np.stack([
                np.array([
                    [ 9.93813460e+02, 6.73319543e+02, 2.75733627e+01, -2.84137097e+02], 
                    [-1.38022223e+01, 4.29775748e+02, -9.80293378e+02, -5.16318806e+02], 
                    [-1.25768144e-02, 9.98442012e-01, 5.43633383e-02, -4.28792161e-01], 
                    [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
                ]),
                np.array([
                    [ 1.09226932e+03, -4.94774976e+02, -3.14710481e+01, -3.67329105e+02], 
                    [ 3.04357841e+02, 2.56876688e+02, -9.91431030e+02, -5.52985866e+02], 
                    [ 8.43070387e-01, 5.36777384e-01, 3.32018426e-02, -6.09714569e-01], 
                    [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
                ]),
                np.array([
                    [ 2.45182066e+01, 1.20253470e+03, 6.24384489e+01, -2.49669138e+02], 
                    [-3.10434650e+02, 2.56270779e+02, -9.90304548e+02, -5.44706073e+02], 
                    [-8.24078173e-01, 5.65041891e-01, 4.02843207e-02, -5.35067645e-01], 
                    [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
                ]),
                np.array([
                    [-6.43027089e+02, -6.80713597e+02, -2.16415309e+01, -6.98239277e+02], 
                    [-8.32595991e+00, -3.55999922e+02, -6.52047284e+02, -5.67401993e+02], 
                    [-8.09745037e-03, -9.99188356e-01, -3.94595969e-02, -1.01836906e+00], 
                    [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
                ]),
                np.array([
                    [-9.49209822e+02, 7.38663920e+02, 4.26213282e+01, -4.97745049e+02], 
                    [-3.70047172e+02, -8.19211684e+01, -1.00201050e+03, -4.49684719e+02], 
                    [-9.47606632e-01, -3.19424713e-01, 3.08614988e-03, -4.33192210e-01], 
                    [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
                ]),
                np.array([
                    [ 2.28671785e+02, -1.17530448e+03, -4.80341886e+01, -2.18502572e+02], 
                    [ 3.56575704e+02, -9.76423127e+01, -1.00009447e+03, -4.69595942e+02], 
                    [ 9.24213055e-01, -3.81863834e-01, -3.20023987e-03, -4.63930291e-01], 
                    [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
                ])
            ])


            # img_metasの作成
            img_metas = [DataContainer([[{
                'scene_token': '0',  # ダミーのscene_token
                'can_bus': can_bus,
                'img_shape': [(736, 1280, 3) for _ in range(6)],
                'lidar2img': lidar2img_stacked,
            }]], cpu_only=True)]

            # driving_commandを[1, 1, 1, 3]の形状に変形
            ego_fut_cmd_tensor = torch.tensor(
                request.driving_command, 
                dtype=torch.float32
            ).view(1, 1, 1, 1, 3).to(self.device)
            ego_fut_cmd_container = DataContainer(ego_fut_cmd_tensor, stack=True, padding_value=0)

            latest_odom = request.ego_history[-1]
            
            # 現在の位置
            current_pos = [
                latest_odom.pose.pose.position.x, 
                latest_odom.pose.pose.position.y
            ]
            
            # 過去の位置（past_posesが存在する場合）
            past_pos = current_pos  # デフォルトは現在の位置
            if latest_odom.past_poses:
                past_pos = [
                    latest_odom.past_poses[0].x, 
                    latest_odom.past_poses[0].y
                ]
            
            # テンソルに変換
            ego_his_trajs_tensor = torch.tensor([[[past_pos, current_pos]]]).float().to(self.device)
            ego_his_trajs = DataContainer(ego_his_trajs_tensor, stack=True, padding_value=0)

            # モデル入力の準備
            input_data = {
                'img': [img_container],
                'img_metas': img_metas,
                'ego_fut_cmd': [ego_fut_cmd_container],
                'ego_lcf_feat': [ego_lcf_feat_container],
                'gt_bboxes_3d': None,
                'gt_labels_3d': None,
                # 'prev_bev': None,
                'points': None,
                'fut_valid_flag': False,  # 追加
                'ego_his_trajs': [ego_his_trajs],
                'ego_fut_trajs': None,
                'ego_fut_masks': None,
                'map_gt_labels_3d': None,
                'map_gt_bboxes_3d': None,
                'gt_attr_labels': None
            }

            with torch.no_grad():
                output = self.vad_wrapper.model(return_loss=False, rescale=True, **input_data)
            # ego_fut_predsを取得 (shape: [3, 6, 2])
            ego_fut_preds = output[0]['pts_bbox']['ego_fut_preds']

            ego_fut_cmd = ego_fut_cmd_tensor.data.cpu()[0, 0, 0]
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0, 0]
            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]
            ego_fut_pred = ego_fut_pred.cumsum(dim=-2)

            predicted_object = self.create_predicted_object_from_single_trajectory(
                trajectory=ego_fut_pred,  # shape: [6, 2]
                current_pos=current_pos,
                timestamp_sec=timestamp_sec,
                timestamp_nanosec=timestamp_nanosec,
            )

            # Create response
            response = vad_service_pb2.VADResponse()
            if len(request.ego_history) > 0:
                response.header.CopyFrom(request.ego_history[-1].header)
            response.objects.append(predicted_object)
            print("send response")
            return response

        except Exception as e:
            print(f"Error in ProcessData: {str(e)}")
            import traceback
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

def serve(vehicle_info_path: Path):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vad_service_pb2_grpc.add_VADServiceServicer_to_server(VADServicer(vehicle_info_path), server)
    server.add_insecure_port('[::]:50051')
    print("Starting server on port 50051...")
    server.start()
    server.wait_for_termination()

def parse_args():
   parser = argparse.ArgumentParser(description='VAD Server')
   parser.add_argument("--vehicle-info-path", type=str, default=None,
                      help="Path to vehicle info YAML file")
   return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    serve(args.vehicle_info_path)
