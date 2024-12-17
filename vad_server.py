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
        self.vehicle_params = VehicleParams(vehicle_info_path)
        self.device = device

    def create_predicted_object_from_trajectory(self, trajectory: torch.Tensor, confidence: float, timestamp):
        """単一の軌道をPredictedObjectに変換"""
        obj = vad_service_pb2.PredictedObject(
            uuid=str(uuid.uuid4()),
            existence_probability=0.9
        )
        
        # 初期位置を設定
        kinematics = obj.kinematics
        kinematics.initial_pose_with_covariance.pose.position.x = 0.0
        kinematics.initial_pose_with_covariance.pose.position.y = 0.0
        kinematics.initial_pose_with_covariance.pose.position.z = 0.0
        kinematics.initial_pose_with_covariance.pose.orientation.w = 1.0
        kinematics.initial_pose_with_covariance.covariance.extend([0.0] * 36)
        
        # 予測軌道を設定
        path = kinematics.predicted_paths.add()
        for waypoint in trajectory:
            pose = path.path.add()
            pose.position.x = float(waypoint[0])
            pose.position.y = float(waypoint[1])
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            
        # 画像のタイムスタンプを使用
        path.time_step.sec = 0
        path.time_step.nanosec = 100000000  # 0.1 second
        path.confidence = float(confidence)
        
        return obj

    def ProcessData(self, request, context):
        try:
            timestamp = 0.0
            # timestamp = request.image_timestamp

            camera_images = {}
            for camera_image in request.images:
                # 画像データをデコード
                nparr = np.frombuffer(camera_image.image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                camera_images[camera_image.camera_id] = image

            print("Processing images")
            img_tensor = torch.stack([
                torch.from_numpy(img).permute(2, 0, 1).float() 
                for img in camera_images.values()
            ]).to(self.device)
            img_container = DataContainer(img_tensor, stack=True, padding_value=0)

            # img_metasの作成
            img_metas = [[{
                'scene_token': '0',  # ダミーのscene_token
                'can_bus': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            }]]

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
            steering = request.steering_angle if hasattr(request, 'steering_angle') else 0.0
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

            # モデル入力の準備
            input_data = {
                'img': [img_container],
                'img_metas': img_metas,
                'ego_fut_cmd': [[torch.tensor([0, 0, 1], dtype=torch.float32).to(self.device)]],
                'ego_lcf_feat': [ego_lcf_feat_container],
            }

            # Process with VAD model
            output = self.vad_model(return_loss=False, rescale=True, **input_data)
            # ego_fut_predsを取得 (shape: [3, 6, 2])
            ego_fut_preds = output[0]['pts_bbox']['ego_fut_preds']

            predcicted_objects = []
            confidences = np.linspace(1.0, 0.5, ego_fut_preds.shape[1])  # 6つの予測に対する信頼度を設定
            for traj_idx in range(ego_fut_preds.shape[1]):
                trajectory = ego_fut_preds[:, traj_idx, :]  # (3, 2)の軌道データ
                predicted_object = self.create_predicted_object_from_trajectory(
                        trajectory=trajectory,
                        confidence=confidences[traj_idx],
                        timestamp=timestamp,
                    )
                predcicted_objects.append(predicted_object)
            
            # Create response
            response = vad_service_pb2.VADResponse()
            if len(request.ego_history) > 0:
                response.header.CopyFrom(request.ego_history[-1].header)
            for predicted_object in predcicted_objects:
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
