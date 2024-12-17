import grpc
import vad_service_pb2
import vad_service_pb2_grpc
import cv2
import numpy as np
import array
from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Imu
from nav_msgs.msg import Odometry 
from autoware_perception_msgs.msg import (
    PredictedObjects,
    PredictedObject,
    PredictedPath,
    ObjectClassification,
    PredictedObjectKinematics
)
from tier4_planning_msgs.msg import PathWithLaneId, PathPointWithLaneId
from autoware_vehicle_msgs.msg import SteeringReport
from geometry_msgs.msg import (
    PoseWithCovariance,
    TwistWithCovariance,
    Point,
    Quaternion,
    Vector3,
    Pose
)
import uuid
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Header
import time

class VADClient(Node):
    def __init__(self, dummy_mode=False):
        super().__init__('vad_client')
        
        self.dummy_mode = dummy_mode
        
        # QoS設定
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        # カメラのSubscribers
        self.camera_topics = [
            f'/sensing/camera/camera{i}/image_rect_color/compressed' 
            for i in range(6)
        ]

        # 各カメラ画像のSubscriberを作成
        self.latest_images = {topic: None for topic in self.camera_topics}
        for topic in self.camera_topics:
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, topic=topic: self.image_callback(msg, topic),
                image_qos_profile
            )
        self.create_subscription(
            Odometry,
            '/localization/kinematic_state',
            self.ego_callback,
            qos_profile
        )
        self.create_subscription(
            Imu,
            '/sensing/imu/tamagawa/imu_raw',
            self.imu_callback,
            qos_profile
        )
        self.create_subscription(
            PathWithLaneId,
            '/planning/scenario_planning/lane_driving/behavior_planning/path_with_lane_id',
            self.path_callback,
            qos_profile
        )
        self.create_subscription(
            SteeringReport,
            '/vehicle/status/steering_status',
            self.steering_callback,
            qos_profile
        )
        # Store latest data
        self.latest_image = None
        self.latest_imu = None
        self.latest_path = None
        self.latest_steering = None
        self.ego_history = []
        
        # 経路からの運転コマンド（デフォルトは直進）
        self.driving_command = [0, 0, 1]

        # Publisher for predicted objects
        self.trajectory_pub = self.create_publisher(
            PredictedObjects,
            '~/output/objects',
            qos_profile
        )
        
        # Dummy publishers (使用時のみ初期化)
        if self.dummy_mode:
            self.dummy_image_pubs = {}
            for topic in self.camera_topics:
                self.dummy_image_pubs[topic] = self.create_publisher(
                    CompressedImage,
                    topic,
                    image_qos_profile
                )
            self.dummy_odom_pub = self.create_publisher(
                Odometry,
                '/localization/kinematic_state',
                qos_profile
            )
            self.dummy_imu_pub = self.create_publisher(
                Imu,
                '/sensing/imu/tamagawa/imu_raw',
                qos_profile
            )
            self.dummy_path_pub = self.create_publisher(
                PathWithLaneId,
                '/planning/scenario_planning/lane_driving/behavior_planning/path_with_lane_id',
                qos_profile
            )
            self.dummy_steering_pub = self.create_publisher(
                SteeringReport,
                '/vehicle/status/steering_status',
                qos_profile
            )
            # ダミーデータ生成用のタイマー（10Hz）
            self.create_timer(0.1, self.publish_dummy_data)
        
        # gRPC setup
        self.channel = grpc.insecure_channel('localhost:50051')
        self.stub = vad_service_pb2_grpc.VADServiceStub(self.channel)
        
        # Store latest data
        self.latest_image = None
        self.ego_history = []
        self.driving_command = [0, 0, 1]  # Go straight by default
        
        self.get_logger().info('Initialized VADClient')
        if self.dummy_mode:
            self.get_logger().info('Running in dummy mode')

    def publish_dummy_data(self):
        """ダミーデータをパブリッシュする"""
        current_time = self.get_clock().now()
        
        # ダミー画像の生成とパブリッシュ
        for topic, pub in self.dummy_image_pubs.items():
            dummy_image = CompressedImage()
            dummy_image.header.stamp = current_time.to_msg()
            dummy_image.header.frame_id = "camera"
            dummy_image.format = "jpeg"
            
            # 100x100の黒い画像を作成
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            _, img_encoded = cv2.imencode('.jpg', img)
            dummy_image.data = img_encoded.tobytes()
            
            pub.publish(dummy_image)
        
        # ダミーOdometryの生成とパブリッシュ
        dummy_odom = Odometry()
        dummy_odom.header.stamp = current_time.to_msg()
        dummy_odom.header.frame_id = "map"
        dummy_odom.child_frame_id = "base_link"
        
        # 適当な位置と姿勢を設定
        t = time.time()
        dummy_odom.pose.pose.position = Point(x=np.sin(t), y=np.cos(t), z=0.0)
        dummy_odom.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        dummy_odom.pose.covariance = [0.0] * 36
        
        # 適当な速度を設定
        dummy_odom.twist.twist.linear = Vector3(x=1.0, y=0.0, z=0.0)
        dummy_odom.twist.twist.angular = Vector3(x=0.0, y=0.0, z=0.1)
        dummy_odom.twist.covariance = [0.0] * 36
        
        self.dummy_odom_pub.publish(dummy_odom)

        # IMUのダミーデータ
        dummy_imu = Imu()
        dummy_imu.header.stamp = current_time.to_msg()
        dummy_imu.header.frame_id = "base_link"
        dummy_imu.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        dummy_imu.linear_acceleration = Vector3(x=0.0, y=0.0, z=9.81)
        dummy_imu.angular_velocity = Vector3(x=0.0, y=0.0, z=0.0)
        self.dummy_imu_pub.publish(dummy_imu)
        
        # PathWithLaneIdのダミーデータ
        dummy_path = PathWithLaneId()
        dummy_path.header.stamp = current_time.to_msg()
        dummy_path.header.frame_id = "map"
        # ダミーの直進パス
        point = PathPointWithLaneId()
        point.point.pose.position = Point(x=1.0, y=0.0, z=0.0)
        point.point.pose.orientation = Quaternion(w=1.0)
        dummy_path.points.append(point)
        self.dummy_path_pub.publish(dummy_path)
        
        # SteeringReportのダミーデータ
        dummy_steering = SteeringReport()
        dummy_steering.stamp = current_time.to_msg()
        dummy_steering.steering_tire_angle = 0.0
        self.dummy_steering_pub.publish(dummy_steering)
        
    def image_callback(self, msg: CompressedImage, topic: str):
        """カメラ画像のコールバック"""
        self.get_logger().debug(f'Received image from {topic}')
        self.latest_images[topic] = msg.data
        self.try_process()
        
    def ego_callback(self, msg: Odometry):
        self.get_logger().debug('Received odometry')
        # Convert ROS2 Odometry to proto Odometry
        proto_odom = self._convert_odom_to_proto(msg)
        self.ego_history.append(proto_odom)
        if len(self.ego_history) > 10:  # Keep last 10 poses
            self.ego_history.pop(0)
        self.try_process()

    def imu_callback(self, msg: Imu):
        self.get_logger().debug('Received IMU data')
        self.latest_imu = msg
        
    def path_callback(self, msg: PathWithLaneId):
        self.get_logger().debug('Received path data')
        self.latest_path = msg
        # パスから運転コマンドを計算（実装は別途必要）
        self.driving_command = self._compute_driving_command(msg)
        
    def steering_callback(self, msg: SteeringReport):
        """ステアリング情報のコールバック"""
        self.get_logger().debug('Received steering data')
        self.latest_steering = msg.steering_tire_angle  # 直接角度を保存
    
    def _compute_driving_command(self, path_msg: PathWithLaneId, num_points: int = 20) -> List[float]:
        # TODO(Shin-kyoto): 進行方向がyとみなして良いのかを確認すべし
        # TODO(Shin-kyoto): 最初のN点をとって判定してよいのかを確認すべし
        """パスの形状から運転コマンド（右折/左折/直進）を計算
        
        Args:
            path_msg: PathWithLaneIdメッセージ
            num_points: 確認する先読み点数（デフォルト20点）
        
        Returns:
            List[float]: [右折, 左折, 直進]のone-hotベクトル
        """
        if len(path_msg.points) < num_points:
            num_points = len(path_msg.points)
            
        if num_points < 2:
            return [0, 0, 1]  # 点が少なすぎる場合は直進とみなす
            
        # 最初の点の位置と向きを基準にする
        start_x = path_msg.points[0].point.pose.position.x
        start_y = path_msg.points[0].point.pose.position.y
        
        # 最後の点（num_points番目）との相対位置を計算
        end_x = path_msg.points[num_points-1].point.pose.position.x
        end_y = path_msg.points[num_points-1].point.pose.position.y
        
        # 相対位置を計算
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        
        # 進行方向の判定
        # x, yの変化量から大まかな進行方向を推定
        # 閾値は調整が必要
        TURN_THRESHOLD = 2.0  # メートル
        
        if abs(delta_x) > TURN_THRESHOLD:
            if delta_x > 0:
                return [1, 0, 0]  # 右折
            else:
                return [0, 1, 0]  # 左折
        else:
            return [0, 0, 1]  # 直進

    def _convert_odom_to_proto(self, msg: Odometry) -> vad_service_pb2.Odometry:
        proto_odom = vad_service_pb2.Odometry()
        
        # Header
        proto_odom.header.stamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        proto_odom.header.frame_id = msg.header.frame_id
        
        # Child frame id
        proto_odom.child_frame_id = msg.child_frame_id
        
        # PoseWithCovariance
        proto_odom.pose.pose.position.x = msg.pose.pose.position.x
        proto_odom.pose.pose.position.y = msg.pose.pose.position.y
        proto_odom.pose.pose.position.z = msg.pose.pose.position.z
        proto_odom.pose.pose.orientation.x = msg.pose.pose.orientation.x
        proto_odom.pose.pose.orientation.y = msg.pose.pose.orientation.y
        proto_odom.pose.pose.orientation.z = msg.pose.pose.orientation.z
        proto_odom.pose.pose.orientation.w = msg.pose.pose.orientation.w
        proto_odom.pose.covariance.extend(msg.pose.covariance)
        
        # TwistWithCovariance
        proto_odom.twist.twist.linear.x = msg.twist.twist.linear.x
        proto_odom.twist.twist.linear.y = msg.twist.twist.linear.y
        proto_odom.twist.twist.linear.z = msg.twist.twist.linear.z
        proto_odom.twist.twist.angular.x = msg.twist.twist.angular.x
        proto_odom.twist.twist.angular.y = msg.twist.twist.angular.y
        proto_odom.twist.twist.angular.z = msg.twist.twist.angular.z
        proto_odom.twist.covariance.extend(msg.twist.covariance)
        
        return proto_odom
        
    def try_process(self):
        """全てのカメラ画像が揃っているか確認して処理を実行"""
        if any(img is None for img in self.latest_images.values()):
            print("Cannot get all camera data")
            return  # 全てのカメラ画像が揃っていない

        if not self.ego_history:
            print("Cannot get odom")
            return  # オドメトリデータがない
            
        # bytes型に変換
        # TODO(Shin-kyoto): send 6 images
        image_name = "/sensing/camera/camera0/image_rect_color/compressed"
        if isinstance(self.latest_images[image_name], array.array):
            image_data = bytes(self.latest_images[image_name])
        else:
            image_data = self.latest_images[image_name]

        request = vad_service_pb2.VADRequest(
            image_data=image_data,
            image_encoding='jpeg',
            ego_history=self.ego_history,
            map_data=b'dummy_map',  # Replace with actual map data
            driving_command=self.driving_command
        )
        try:
            self.get_logger().info('Sending request to VAD server')
            response = self.stub.ProcessData(request)
            # self.get_logger().info(f'Received response from VAD server: {response}')
            self.publish_trajectory(response)
        except grpc.RpcError as e:
            self.get_logger().error(f'RPC failed')
            # self.get_logger().error(f'RPC failed: {e.code()}: {e.details()}')
                
    def publish_trajectory(self, response: vad_service_pb2.VADResponse):
        msg = PredictedObjects()
        msg.header.stamp.sec = response.header.stamp // 1000000000
        msg.header.stamp.nanosec = response.header.stamp % 1000000000
        msg.header.frame_id = response.header.frame_id
        
        for proto_obj in response.objects:
            obj = PredictedObject()

            uuid_bytes = uuid.UUID(proto_obj.uuid).bytes
            obj.object_id.uuid = np.frombuffer(uuid_bytes, dtype=np.uint8)
            obj.existence_probability = proto_obj.existence_probability
            
            # Convert kinematics
            kinematics = proto_obj.kinematics
            # Initial pose
            obj.kinematics.initial_pose_with_covariance = self._convert_proto_to_pose_with_covariance(
                kinematics.initial_pose_with_covariance
            )
            
            # Predicted paths
            for proto_path in kinematics.predicted_paths:
                predicted_path = PredictedPath()
                # Convert poses
                for proto_pose in proto_path.path:
                    pose = Pose()
                    pose.position.x = proto_pose.position.x
                    pose.position.y = proto_pose.position.y
                    pose.position.z = proto_pose.position.z
                    pose.orientation.x = proto_pose.orientation.x
                    pose.orientation.y = proto_pose.orientation.y
                    pose.orientation.z = proto_pose.orientation.z
                    pose.orientation.w = proto_pose.orientation.w
                    predicted_path.path.append(pose)
                
                # Time step
                predicted_path.time_step.sec = proto_path.time_step.sec
                predicted_path.time_step.nanosec = proto_path.time_step.nanosec
                predicted_path.confidence = proto_path.confidence
                
                obj.kinematics.predicted_paths.append(predicted_path)
            
            msg.objects.append(obj)
        
        self.trajectory_pub.publish(msg)
        self.get_logger().info('Published trajectory')

    def _convert_proto_to_pose_with_covariance(self, proto_pose: vad_service_pb2.PoseWithCovariance) -> PoseWithCovariance:
        ros_pose = PoseWithCovariance()
        
        # Convert position and orientation
        ros_pose.pose.position.x = proto_pose.pose.position.x
        ros_pose.pose.position.y = proto_pose.pose.position.y
        ros_pose.pose.position.z = proto_pose.pose.position.z
        ros_pose.pose.orientation.x = proto_pose.pose.orientation.x
        ros_pose.pose.orientation.y = proto_pose.pose.orientation.y
        ros_pose.pose.orientation.z = proto_pose.pose.orientation.z
        ros_pose.pose.orientation.w = proto_pose.pose.orientation.w
        
        # Convert covariance
        ros_pose.covariance = list(proto_pose.covariance)
        
        return ros_pose
def main(args=None):
   rclpy.init(args=args)
   
   # ダミーモードで起動するかどうかをコマンドライン引数などで制御可能
   dummy_mode = True  # or False
   
   client = VADClient(dummy_mode=dummy_mode)
   try:
       rclpy.spin(client)
   except KeyboardInterrupt:
       pass
   finally:
       client.destroy_node()
       rclpy.shutdown()

if __name__ == '__main__':
   main()
