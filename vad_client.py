import argparse
import grpc
import vad_service_pb2
import vad_service_pb2_grpc
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import array
from typing import List        

import rclpy
from rclpy.node import Node
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import CompressedImage, Imu, CameraInfo
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
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import (
    PoseWithCovariance,
    PoseStamped,
    TransformStamped,
    TwistWithCovariance,
    Point,
    Quaternion,
    Vector3,
    Pose
)
import uuid
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Header
import time

def create_transform_matrix(transform_msg: TransformStamped) -> np.ndarray:
    """TransformStampedメッセージから4x4同次変換行列を作成する
    
    Args:
        transform_msg: ROS2のTransformStampedメッセージ
        
    Returns:
        4x4の同次変換行列（NumPy配列）
    """
    # クォータニオンから回転行列を作成
    quat = [
        transform_msg.transform.rotation.x,
        transform_msg.transform.rotation.y,
        transform_msg.transform.rotation.z,
        transform_msg.transform.rotation.w
    ]
    R = Rotation.from_quat(quat).as_matrix()
    
    # 並進ベクトルを取得
    t = np.array([
        transform_msg.transform.translation.x,
        transform_msg.transform.translation.y,
        transform_msg.transform.translation.z
    ])
    
    # 4x4同次変換行列を作成
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T

def get_lidar2img(projection_matrix: np.ndarray, sensor_kit_base_link2camera_msg: TransformStamped):
    sensor_kit_base_link2camera: np.ndarray = create_transform_matrix(sensor_kit_base_link2camera_msg)
    viewpad = np.eye(4)
    viewpad[:projection_matrix.shape[0], :projection_matrix.shape[1]] = projection_matrix
    ns_lidar2sensor_kit_base_link = np.array([[ 0,  1,  0,  0,],
                                              [-1,  0,  0,  0,],
                                              [ 0,  0,  1,  0,],
                                              [ 0,  0,  0,  1,],])
    ns_lidar2cam = sensor_kit_base_link2camera @ ns_lidar2sensor_kit_base_link
    return viewpad @ ns_lidar2cam.T

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
        self.camera_info_topics = [
            f'/sensing/camera/camera{i}/camera_info'
            for i in range(6)
        ]
        # カメラ情報の保存用ディクショナリ
        self.latest_camera_info: Dict[str, CameraInfo] = {
            topic: None for topic in self.camera_info_topics
        }
        # センサーキットからカメラへのTF保存用ディクショナリ
        self.camera_transforms: Dict[str, TransformStamped] = {}
        self.reset_topic = '/sensing/camera/camera0/image_rect_color/compressed'

        self.tf_subscription = self.create_subscription(
            TFMessage, 
            '/tf', 
            self.tf_callback, 
            10  # QoSプロファイル
        )

        # 最新の座標変換を保存するディクショナリ
        self.latest_transforms = {}

        # 各カメラ画像のSubscriberを作成
        self.latest_images = {topic: None for topic in self.camera_topics}
        for topic in self.camera_topics:
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, topic=topic: self.image_callback(msg, topic),
                image_qos_profile
            )
        # camera_infoのサブスクライバー
        for topic in self.camera_info_topics:
            self.create_subscription(
                CameraInfo,
                topic,
                lambda msg, topic=topic: self.camera_info_callback(msg, topic),
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
        self.past_poses = []
        
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

        # tf2バッファとリスナーの初期化
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
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

        if topic == self.reset_topic:
            self.latest_images = {topic: None for topic in self.camera_topics}

        self.latest_images[topic] = msg
        self.try_process()

    def camera_info_callback(self, msg: CameraInfo, topic: str):
        """カメラキャリブレーション情報のコールバック"""
        self.get_logger().debug(f'Received camera info from {topic}')
        self.latest_camera_info[topic] = msg
        
        # camera_optical_linkのフレーム名を取得
        camera_name: str = topic.split('/')[-2]
        camera_frame = f'{camera_name}/camera_optical_link'
        
        try:
            # sensor_kit_base_linkからカメラへのTFを取得
            transform = self.tf_buffer.lookup_transform(
                'sensor_kit_base_link',
                camera_frame,
                self.get_clock().now(),
                rclpy.duration.Duration(seconds=1.0)
            )
            self.camera_transforms[camera_name] = transform
            self.get_logger().debug(f'Updated transform for {camera_frame}')
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warning(f'Failed to lookup transform for {camera_frame}: {e}')

    def ego_callback(self, msg: Odometry, num_past_ego_pose: int = 2):
        self.get_logger().debug('Received odometry')
        current_point = vad_service_pb2.Point(
            x=msg.pose.pose.position.x, 
            y=msg.pose.pose.position.y, 
            z=msg.pose.pose.position.z
        )

        # Convert ROS2 Odometry to proto Odometry
        proto_odom = self._convert_odom_to_proto(msg)
        if not self.past_poses:
            # past_posesが空の場合、current_pointを追加
            proto_odom.past_poses.append(current_point)
        else:
            # past_posesの最新の位置を追加
            proto_odom.past_poses.extend(self.past_poses[-1:])

        self.ego_history.append(proto_odom)
        if len(self.ego_history) > num_past_ego_pose:  # Keep last num_past_ego_pose frames
            self.ego_history.pop(0)

        self.past_poses.append(current_point)

    def tf_callback(self, msg):
        """
        /tfメッセージを処理するコールバック関数
        """
        for transform in msg.transforms:
            # 座標変換を最新の情報で更新
            key = (transform.header.frame_id, transform.child_frame_id)
            self.latest_transforms[key] = transform

            self.can_bus = self.transform_to_ego2global(transform)

    def transform_to_ego2global(self, transform):
        """
        global2egoの変換からego2globalの変換を計算
        """

        # 元の回転と並進
        global_rot = Rotation.from_quat([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ])

        global_trans = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ])

        # 逆変換の計算
        ego_rot = global_rot.inv()
        ego_trans = -global_rot.inv().apply(global_trans)

        # クォータニオンの取得
        ego_rotation = ego_rot.as_quat()

        def quaternion_yaw(quat):
            """
            四元数からyaw角を取得
            quat: [x, y, z, w]の形式の四元数
            """
            return np.arctan2(
                2 * (quat[3] * quat[2] + quat[0] * quat[1]),
                1 - 2 * (quat[1]**2 + quat[2]**2)
            )

        # パッチ角の計算
        patch_angle = quaternion_yaw(ego_rotation) / np.pi * 180

        # 負の角度を360度に変換
        if patch_angle < 0:
            patch_angle += 360

        return {
            'ego2global_translation': ego_trans,
            'ego2global_rotation': ego_rotation,
            'patch_angle': patch_angle
        }

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
        # autowareにおいては，進行方向がx. 左向きにy軸を取る．よって，delta_yが負なら右折．delta_yが正なら左折．
        # TODO(Shin-kyoto): 最初のN点をとって判定してよいのかを確認すべし
        # TODO(Shin-kyoto): path_msgの中に入っている情報はbase_link座標系と考えて良いのか？
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
        start_y = path_msg.points[0].point.pose.position.y
        
        # 最後の点（num_points番目）との相対位置を計算
        end_y = path_msg.points[num_points-1].point.pose.position.y
        
        # 相対位置を計算
        delta_y = end_y - start_y
        
        # 進行方向の判定
        # x, yの変化量から大まかな進行方向を推定
        # 閾値は調整が必要
        TURN_THRESHOLD = 2.0  # メートル
        
        if abs(delta_y) > TURN_THRESHOLD:
            if delta_y > 0:
                return [0, 1, 0]  # 左折
            else:
                return [1, 0, 0]  # 右折
        else:
            return [0, 0, 1]  # 直進

    def _convert_odom_to_proto(self, msg: Odometry) -> vad_service_pb2.Odometry:
        proto_odom = vad_service_pb2.Odometry()
        
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

        # 最初にすべての画像のサイズをチェック
        first_shape = None
        for camera_id in range(6):
            topic = f'/sensing/camera/camera{camera_id}/image_rect_color/compressed'
            image_data = self.latest_images[topic].data
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if first_shape is None:
                first_shape = image.shape
            elif image.shape != first_shape:
                self.get_logger().error(
                    f'Image size mismatch: camera0 is {first_shape} but camera{camera_id} is {image.shape}. If you are running rosbag while using --dummy-pub, you may get this error.'
                )
                return 

        # bytes型に変換
        # TODO(Shin-kyoto): send 6 images
        # カメラ画像を準備
        camera_images = []
        for camera_id in range(6):
            topic = f'/sensing/camera/camera{camera_id}/image_rect_color/compressed'
            info_topic = f'/sensing/camera/camera{camera_id}/camera_info'
            image_topic = self.latest_images[topic]
            camera_info = self.latest_camera_info[info_topic]
            
            # bytes型に変換
            if isinstance(image_topic.data, array.array):
                image_data = bytes(image_topic.data)
                
            camera_image = vad_service_pb2.CameraImage(
                image_data=image_data,
                encoding='jpeg',
                camera_id=camera_id,
                time_step_sec=image_topic.header.stamp.sec,
                time_step_nanosec=image_topic.header.stamp.nanosec
                # カメラキャリブレーション情報を追加
            )
            camera_images.append(camera_image)

        # lidar2img
        lidar2imgs = {}
        for camera_id in range(6):
            info_topic = f'/sensing/camera/camera{camera_id}/camera_info'
            camera_info = self.latest_camera_info[info_topic]
            projection_matrix = np.delete(camera_info.p.reshape(3, 4), 3, 1)
            sensor_kit_base_link2camera_msg: TransformStamped = self.camera_transforms[f"camera{camera_id}"]
            lidar2imgs[camera_id] = get_lidar2img(projection_matrix=projection_matrix, sensor_kit_base_link2camera_msg=sensor_kit_base_link2camera_msg).T
        steering_report = vad_service_pb2.SteeringReport(
            steering_tire_angle=self.latest_steering if self.latest_steering is not None else 0.0
        )
        can_bus_msg = vad_service_pb2.CanBus(
            ego2global_translation=vad_service_pb2.Vector3(
                x=self.can_bus['ego2global_translation'][0],
                y=self.can_bus['ego2global_translation'][1],
                z=self.can_bus['ego2global_translation'][2]
            ),
            ego2global_rotation=vad_service_pb2.Vector3(
                x=self.can_bus['ego2global_rotation'][0],
                y=self.can_bus['ego2global_rotation'][1],
                z=self.can_bus['ego2global_rotation'][2]
            ),
            patch_angle=self.can_bus['patch_angle']
        )
        request = vad_service_pb2.VADRequest(
            images=camera_images,
            ego_history=self.ego_history,
            map_data=b'dummy_map',  # Replace with actual map data
            driving_command=self.driving_command,
            steering=steering_report,
            can_bus=can_bus_msg,
            # lidar2img=lidar2imgs,
        )
        try:
            self.get_logger().info('Sending request to VAD server')
            response = self.stub.ProcessData(request)
            # self.get_logger().info(f'Received response from VAD server: {response}')
            self.publish_trajectory(response)
        except grpc.RpcError as e:
            self.get_logger().error(f'RPC failed: code={e.code()}, details={e.details()}')
            self.get_logger().error(f'Debug error string: {e.debug_error_string()}')
        except Exception as e:
            self.get_logger().error(f'Unexpected error: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def do_transform_pose(self, pose_base_link, time_sec, time_nanosec, target_frame: str):
        """
        tf2の変換を使用してPoseを変換
        
        Args:
            pose_base_link: 変換元のPoseWithCovariance (base_linkフレーム)
            time_sec: タイムスタンプの秒部分
            time_nanosec: タイムスタンプのナノ秒部分
            target_frame: 変換先のフレーム名（例：'map'）
            
        Returns:
            変換後のPoseWithCovariance
        """
        # PoseStampedメッセージの作成
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = 'base_link'  # 変換元のフレーム
        pose_stamped.header.stamp.sec = time_sec
        pose_stamped.header.stamp.nanosec = time_nanosec

        try:
            # 変換のタイムアウトを1秒に設定
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                'base_link',
                pose_stamped.header.stamp,
                Duration(seconds=1.0)
            )
            
            # tf2_geometry_msgsを使用してポーズを変換
            return tf2_geometry_msgs.do_transform_pose(pose_base_link,transform)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'Failed to transform pose: {e}')
            raise
        
    def publish_trajectory(self, response: vad_service_pb2.VADResponse):
        msg = PredictedObjects()
        msg.header.frame_id = "map"  # フレームIDをmapに変更

        # TODO(Shin-kyoto): tf2による座標変換を実装
        # デフォルトのタイムスタンプ
        # default_time_step_sec = 0
        # default_time_step_nanosec = 0

        # try:
        #     # オブジェクトがない場合は早期リターン
        #     if not response.objects:
        #         self.get_logger().warn('No objects in VAD response')
        #         return

        #     # 最初に見つかったパスのタイムスタンプを使用
        #     for proto_obj in response.objects:
        #         if proto_obj.kinematics.predicted_paths:
        #             default_time_step_sec = proto_obj.kinematics.predicted_paths[0].time_step.sec
        #             default_time_step_nanosec = proto_obj.kinematics.predicted_paths[0].time_step.nanosec
        #             break

        #     # 変換時間は最初に見つかったパスの時間を使用
        #     transform_time = rclpy.time.Time(
        #         seconds=default_time_step_sec, 
        #         nanoseconds=default_time_step_nanosec
        #     )
            
        #     # base_linkからmap座標系への座標変換を取得
        #     transform = self.tf_buffer.lookup_transform(
        #         'map', 
        #         'base_link', 
        #         transform_time,
        #         timeout=rclpy.duration.Duration(seconds=0.1)
        #     )
        # except Exception as e:
        #     self.get_logger().error(f'座標変換エラー: {e}')
        for proto_obj in response.objects:
            obj = PredictedObject()

            uuid_bytes = uuid.UUID(proto_obj.uuid).bytes
            obj.object_id.uuid = np.frombuffer(uuid_bytes, dtype=np.uint8)
            obj.existence_probability = proto_obj.existence_probability
            
            # Convert kinematics
            kinematics = proto_obj.kinematics
            # 初期位置の座標変換
            initial_pose = self._convert_proto_to_pose_with_covariance(
                kinematics.initial_pose_with_covariance
            )
            # 初期位置を座標変換
            initial_time_sec = kinematics.predicted_paths[0].time_step.sec
            initial_time_nanosec = kinematics.predicted_paths[0].time_step.nanosec
            initial_pose.pose = self.do_transform_pose(initial_pose.pose, initial_time_sec, initial_time_nanosec, target_frame="map")
            obj.kinematics.initial_pose_with_covariance = initial_pose
            
            # Predicted paths
            for proto_path in kinematics.predicted_paths:
                predicted_path = PredictedPath()
                time_sec = proto_path.time_step.sec
                time_nanosec = proto_path.time_step.nanosec
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
                    # 座標変換
                    pose_map = self.do_transform_pose(pose, time_sec, time_nanosec, target_frame="map")
                    predicted_path.path.append(pose_map)
                
                # Time step
                predicted_path.time_step.sec = 1
                predicted_path.time_step.nanosec = 0
                predicted_path.confidence = proto_path.confidence
                
                obj.kinematics.predicted_paths.append(predicted_path)
            
            msg.objects.append(obj)
        msg.header.stamp.sec = proto_path.time_step.sec
        msg.header.stamp.nanosec = proto_path.time_step.nanosec
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

def parse_args():
   parser = argparse.ArgumentParser(description='VAD Client')
   parser.add_argument('--dummy-pub', action='store_true',
                      help='Enable dummy publisher mode')
   return parser.parse_args()

def main(args=None):
   rclpy.init()
   
   args = parse_args()

   client = VADClient(dummy_mode=args.dummy_pub)
   try:
       rclpy.spin(client)
   except KeyboardInterrupt:
       pass
   finally:
       client.destroy_node()
       rclpy.shutdown()

if __name__ == '__main__':
   main()
