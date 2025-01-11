import argparse
import grpc
from concurrent import futures
import vad_service_pb2
import vad_service_pb2_grpc
import cv2
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
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
       params = self._load_vehicle_params()
       self.vehicle_params =  params['vehicle_params']
       self.camera_order = params["camera_order"]

   def _load_vehicle_params(self) -> Dict[str, Any]:
       """車両パラメータをYAMLファイルから読み込む"""
       yaml_path = Path(self.vehicle_info_path)
       if not yaml_path.exists():
           raise ValueError(f"Vehicle info file not found: {yaml_path}")
       
       with open(yaml_path, 'r') as f:
           params = yaml.safe_load(f)
       return params

   @property
   def wheel_base(self) -> float:
       return self.vehicle_params['wheel_base']

   @property
   def vehicle_length(self) -> float:
       return self.vehicle_params['vehicle_length']

   @property
   def vehicle_width(self) -> float:
       return self.vehicle_params['vehicle_width']

   @property
   def max_steer_angle(self) -> float:
       return self.vehicle_params['max_steer_angle']

def aw2ns_xy(aw_x, aw_y):
    ns_x = -aw_y
    ns_y = aw_x
    return ns_x, ns_y

def ns2aw_xy(ns_x, ns_y):
    aw_x = ns_y
    aw_y = -ns_x

    return aw_x, aw_y

def get_nsego2global(awego2global_translation, awego2global_rotation):
    """
    autowareのego座標系からglobal座標系への変換を、
    nuscenesのego座標系からglobal座標系への変換に変換する

    Args:
        awego2global_translation: autowareのego座標系からglobal座標系への並進 (Vector3)
        awego2global_rotation: autowareのego座標系からglobal座標系への回転 (Vector3)

    Returns:
        nsego2global_translation: nuscenesのego座標系からglobal座標系への並進
        nsego2global_rotation: nuscenesのego座標系からglobal座標系への回転（オイラー角 [roll, pitch, yaw]）
    """
    # Vector3からnumpy arrayに変換
    translation = np.array([
        float(awego2global_translation.x),
        float(awego2global_translation.y),
        float(awego2global_translation.z)
    ])
    
    rotation = np.array([
        float(awego2global_rotation.x),
        float(awego2global_rotation.y),
        float(awego2global_rotation.z)
    ])

    # 1. nuscenes ego → autoware ego の回転（z軸周り90度）をquaternionで表現
    ns2aw_rot = R.from_euler('z', 90, degrees=True)

    # 2. autoware ego → autoware global の回転をquaternionに変換
    aw2global_rot = R.from_euler('xyz', rotation)

    # 3. 回転の合成: ns_ego → aw_global = (aw_ego → aw_global) ∘ (ns_ego → aw_ego)
    final_rot = aw2global_rot * ns2aw_rot

    # 4. 並進はそのまま使用（すでにglobal座標系なので）
    final_trans = translation

    # 5. 四元数からオイラー角に変換（xyz順）
    final_euler = final_rot.as_euler('xyz')

    return final_trans, final_euler

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

        aw_x, aw_y = ns2aw_xy(float(current_pos[0]), float(current_pos[1]))
        
        # 現在位置を初期位置として設定
        kinematics = obj.kinematics
        kinematics.initial_pose_with_covariance.pose.position.x = aw_x
        kinematics.initial_pose_with_covariance.pose.position.y = aw_y
        kinematics.initial_pose_with_covariance.pose.position.z = 0.0
        kinematics.initial_pose_with_covariance.pose.orientation.w = 1.0
        kinematics.initial_pose_with_covariance.covariance.extend([0.0] * 36)
        
        # 単一の予測パスを追加
        path = kinematics.predicted_paths.add()
        
        # 6ステップの予測位置を追加
        for waypoint in trajectory:
            pose = path.path.add()
            # 現在位置からの相対位置として設定
            aw_dx, aw_dy = ns2aw_xy(float(waypoint[0]), float(waypoint[1]))
            pose.position.x = aw_dx + aw_x
            pose.position.y = aw_dy + aw_y
            pose.position.z = 0.0
            pose.orientation.w = 1.0
        
        # タイムスタンプと信頼度を設定（信頼度は1.0固定）
        path.time_step.sec = timestamp_sec
        path.time_step.nanosec = timestamp_nanosec
        path.confidence = 1.0
        
        return obj


    def ProcessData(self, request, context):
        try:
            # カメラ0の時刻情報を取得
            timestamp_sec = request.images[0].time_step_sec
            timestamp_nanosec = request.images[0].time_step_nanosec

            camera_order = self.vehicle_params.camera_order
            camera_images = {}
            for camera_image in request.images:
                # 画像データをデコード
                nparr = np.frombuffer(camera_image.image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                camera_images[camera_image.camera_id] = image

            # camera_orderのkeyの順に画像を並び替え
            ordered_images = []
            for camera_name in camera_order.keys():
                camera_id = camera_order[camera_name]
                if camera_id not in camera_images:
                    raise ValueError(f"Camera {camera_name} (ID: {camera_id}) not found in input images")
                ordered_images.append(camera_images[camera_id])

            resized_images = [cv2.resize(img, (1280, 736), interpolation=cv2.INTER_LINEAR) for img in ordered_images]
            
            print("Processing images")
            img_tensor = torch.stack([
                torch.from_numpy(img).permute(2, 0, 1).float() 
                for img in resized_images
            ]).to(self.device)
            img_tensor = img_tensor.view(1, 6, 3, 736, 1280)
            img_container = DataContainer([img_tensor], stack=True, padding_value=0)

            # Odometryから速度と角速度を取得
            latest_odom = request.ego_history[-1]
            aw_vx = latest_odom.twist.twist.linear.x
            aw_vy = latest_odom.twist.twist.linear.y
            ego_vx, ego_vy = aw2ns_xy(aw_vx, aw_vy)
            ego_w = latest_odom.twist.twist.angular.z

            # IMUから加速度を取得
            if hasattr(request, 'imu_data') and request.imu_data:
                aw_ax = request.imu_data.linear_acceleration.x
                aw_ay = request.imu_data.linear_acceleration.y
                ax, ay = aw2ns_xy(aw_ax, aw_ay)
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


            # 座標変換
            nsego2global_translation, nsego2global_rotation = get_nsego2global(
                request.can_bus.ego2global_translation,
                request.can_bus.ego2global_rotation
            )

            # ego2global_translation
            can_bus[0:3] = nsego2global_translation

            # ego2global_rotation
            can_bus[3:7] = [
                nsego2global_rotation[0],  # roll
                nsego2global_rotation[1],  # pitch
                nsego2global_rotation[2],  # yaw
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

            # patch_angle（座標系の変換を考慮して90度加算）
            patch_angle = request.can_bus.patch_angle + 90.0
            can_bus[-2] = patch_angle / 180 * np.pi
            can_bus[-1] = patch_angle
 
            lidar2imgs = {"CAM_FRONT": np.array([[ 7.33252583e+02,  5.94402608e+02,  9.99453551e-01, 2.15000000e-01],
                [ 3.45975239e+01, -1.02234729e+03,  3.29939448e-02, 3.10000000e-02],
                [ 1.03707383e+03,  9.29369516e+00, -1.99999867e-03, -2.40000000e-02],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]), 
            "CAM_BACK": np.array([[-7.38546650e+02, -5.92403847e+02, -9.99135150e-01, -1.11000000e-01],
                [-4.11266232e+01,  1.01731657e+03, -4.15806624e-02, -2.00000000e-02],
                [ 1.03814499e+03,  1.04110399e+01,  2.77555756e-16, -8.40000000e-02],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]), 
            "CAM_FRONT_LEFT": np.array([[ 5.01563547e+02,  1.16820198e+03,  4.46889888e-01, 8.70000000e-02],
                [ 9.44839810e+02, -3.21524561e+01,  8.26262986e-01, 5.81000000e-01],
                [ 7.16687929e+02, -1.83660407e+02, -3.42897807e-01, -2.74000000e-01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]), 
            "CAM_BACK_LEFT": np.array([[-5.71170781e+02,  5.98564067e+02, -5.02091099e-01, -1.22000000e-01],
                [ 9.13947637e+02,  9.99659194e+02,  7.86437808e-01, 5.70000000e-01],
                [ 6.98030501e+02, -2.11852434e+02, -3.59750055e-01, -2.54000000e-01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]), 
            "CAM_FRONT_RIGHT": np.array([[ 5.76732501e+02, -5.99693981e+02,  4.93127182e-01, 8.30000000e-02],
                [-9.24154212e+02, -1.01048461e+03, -7.85160069e-01, -5.80000000e-01],
                [ 6.83639825e+02, -2.20263217e+02, -3.74632151e-01, -2.34000000e-01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
            "CAM_BACK_RIGHT": np.array([[-4.97987562e+02, -1.16655255e+03, -4.48009743e-01, -7.80000000e-02],
                [-9.56548144e+02,  3.61516779e+01, -8.19679493e-01, -5.65000000e-01],
                [ 6.98527638e+02, -1.78263668e+02, -3.56949294e-01, -2.54000000e-01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])}

            # camera_orderのkeyの順に画像を並び替え
            ordered_lidar2imgs = [] 
            for camera_name in camera_order.keys():
                if camera_name not in lidar2imgs.keys():
                    raise ValueError(f"{camera_name} not found in lidar2imgs")
                ordered_lidar2imgs.append(lidar2imgs[camera_name])
            lidar2img_stacked = np.stack(ordered_lidar2imgs)


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
            aw_current_x = latest_odom.pose.pose.position.x
            aw_current_y = latest_odom.pose.pose.position.y
            current_pos = list(aw2ns_xy(aw_current_x, aw_current_y))
            
            # 過去の位置（past_posesが存在する場合）
            past_pos = current_pos  # デフォルトは現在の位置
            if latest_odom.past_poses:
                aw_past_x = latest_odom.past_poses[0].x
                aw_past_y = latest_odom.past_poses[0].y
                ns_past_x, ns_past_y = aw2ns_xy(aw_past_x, aw_past_y)
                past_pos = [ns_past_x - current_pos[0], ns_past_y - current_pos[1]]
            
            # テンソルに変換
            ego_his_trajs_tensor = torch.tensor([[[past_pos, [0,0]]]]).float().to(self.device)
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
            # ego_fut_pred = ego_fut_pred.cumsum(dim=-2)

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
