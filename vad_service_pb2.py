# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: vad_service.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11vad_service.proto\x12\x03vad\"\xb5\x01\n\nVADRequest\x12 \n\x06images\x18\x01 \x03(\x0b\x32\x10.vad.CameraImage\x12\x16\n\x0eimage_encoding\x18\x02 \x01(\t\x12\"\n\x0b\x65go_history\x18\x03 \x03(\x0b\x32\r.vad.Odometry\x12\x10\n\x08map_data\x18\x04 \x01(\x0c\x12\x17\n\x0f\x64riving_command\x18\x05 \x03(\x02\x12\x1e\n\x08imu_data\x18\x06 \x01(\x0b\x32\x0c.vad.ImuData\"F\n\x0b\x43\x61meraImage\x12\x12\n\nimage_data\x18\x01 \x01(\x0c\x12\x10\n\x08\x65ncoding\x18\x02 \x01(\t\x12\x11\n\tcamera_id\x18\x03 \x01(\x05\"\xef\x01\n\x07ImuData\x12$\n\x0borientation\x18\x01 \x01(\x0b\x32\x0f.vad.Quaternion\x12\x1e\n\x16orientation_covariance\x18\x02 \x03(\x01\x12&\n\x10\x61ngular_velocity\x18\x03 \x01(\x0b\x32\x0c.vad.Vector3\x12#\n\x1b\x61ngular_velocity_covariance\x18\x04 \x03(\x01\x12)\n\x13linear_acceleration\x18\x05 \x01(\x0b\x32\x0c.vad.Vector3\x12&\n\x1elinear_acceleration_covariance\x18\x06 \x03(\x01\"\x8f\x01\n\x08Odometry\x12\x1b\n\x06header\x18\x01 \x01(\x0b\x32\x0b.vad.Header\x12\x16\n\x0e\x63hild_frame_id\x18\x02 \x01(\t\x12%\n\x04pose\x18\x03 \x01(\x0b\x32\x17.vad.PoseWithCovariance\x12\'\n\x05twist\x18\x04 \x01(\x0b\x32\x18.vad.TwistWithCovariance\"Q\n\x0bVADResponse\x12\x1b\n\x06header\x18\x01 \x01(\x0b\x32\x0b.vad.Header\x12%\n\x07objects\x18\x02 \x03(\x0b\x32\x14.vad.PredictedObject\"\xc0\x01\n\x0fPredictedObject\x12\x0c\n\x04uuid\x18\x01 \x01(\t\x12\x1d\n\x15\x65xistence_probability\x18\x02 \x01(\x02\x12\x31\n\x0e\x63lassification\x18\x03 \x03(\x0b\x32\x19.vad.ObjectClassification\x12\x32\n\nkinematics\x18\x04 \x01(\x0b\x32\x1e.vad.PredictedObjectKinematics\x12\x19\n\x05shape\x18\x05 \x01(\x0b\x32\n.vad.Shape\":\n\x14ObjectClassification\x12\r\n\x05label\x18\x01 \x01(\x05\x12\x13\n\x0bprobability\x18\x02 \x01(\x02\"\x90\x02\n\x19PredictedObjectKinematics\x12=\n\x1cinitial_pose_with_covariance\x18\x01 \x01(\x0b\x32\x17.vad.PoseWithCovariance\x12?\n\x1dinitial_twist_with_covariance\x18\x02 \x01(\x0b\x32\x18.vad.TwistWithCovariance\x12\x46\n$initial_acceleration_with_covariance\x18\x03 \x01(\x0b\x32\x18.vad.AccelWithCovariance\x12+\n\x0fpredicted_paths\x18\x04 \x03(\x0b\x32\x12.vad.PredictedPath\")\n\x06Header\x12\r\n\x05stamp\x18\x01 \x01(\x03\x12\x10\n\x08\x66rame_id\x18\x02 \x01(\t\"A\n\x12PoseWithCovariance\x12\x17\n\x04pose\x18\x01 \x01(\x0b\x32\t.vad.Pose\x12\x12\n\ncovariance\x18\x02 \x03(\x02\"D\n\x13TwistWithCovariance\x12\x19\n\x05twist\x18\x01 \x01(\x0b\x32\n.vad.Twist\x12\x12\n\ncovariance\x18\x02 \x03(\x02\"D\n\x13\x41\x63\x63\x65lWithCovariance\x12\x19\n\x05\x61\x63\x63\x65l\x18\x01 \x01(\x0b\x32\n.vad.Accel\x12\x12\n\ncovariance\x18\x02 \x03(\x02\"J\n\x04Pose\x12\x1c\n\x08position\x18\x01 \x01(\x0b\x32\n.vad.Point\x12$\n\x0borientation\x18\x02 \x01(\x0b\x32\x0f.vad.Quaternion\"(\n\x05Point\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\"8\n\nQuaternion\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\x12\t\n\x01w\x18\x04 \x01(\x02\"D\n\x05Twist\x12\x1c\n\x06linear\x18\x01 \x01(\x0b\x32\x0c.vad.Vector3\x12\x1d\n\x07\x61ngular\x18\x02 \x01(\x0b\x32\x0c.vad.Vector3\"D\n\x05\x41\x63\x63\x65l\x12\x1c\n\x06linear\x18\x01 \x01(\x0b\x32\x0c.vad.Vector3\x12\x1d\n\x07\x61ngular\x18\x02 \x01(\x0b\x32\x0c.vad.Vector3\"*\n\x07Vector3\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\"^\n\rPredictedPath\x12\x17\n\x04path\x18\x01 \x03(\x0b\x32\t.vad.Pose\x12 \n\ttime_step\x18\x02 \x01(\x0b\x32\r.vad.Duration\x12\x12\n\nconfidence\x18\x03 \x01(\x02\"(\n\x08\x44uration\x12\x0b\n\x03sec\x18\x01 \x01(\x05\x12\x0f\n\x07nanosec\x18\x02 \x01(\r\"6\n\x05Shape\x12\r\n\x05width\x18\x01 \x01(\x02\x12\x0e\n\x06length\x18\x02 \x01(\x02\x12\x0e\n\x06height\x18\x03 \x01(\x02\x32@\n\nVADService\x12\x32\n\x0bProcessData\x12\x0f.vad.VADRequest\x1a\x10.vad.VADResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'vad_service_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_VADREQUEST']._serialized_start=27
  _globals['_VADREQUEST']._serialized_end=208
  _globals['_CAMERAIMAGE']._serialized_start=210
  _globals['_CAMERAIMAGE']._serialized_end=280
  _globals['_IMUDATA']._serialized_start=283
  _globals['_IMUDATA']._serialized_end=522
  _globals['_ODOMETRY']._serialized_start=525
  _globals['_ODOMETRY']._serialized_end=668
  _globals['_VADRESPONSE']._serialized_start=670
  _globals['_VADRESPONSE']._serialized_end=751
  _globals['_PREDICTEDOBJECT']._serialized_start=754
  _globals['_PREDICTEDOBJECT']._serialized_end=946
  _globals['_OBJECTCLASSIFICATION']._serialized_start=948
  _globals['_OBJECTCLASSIFICATION']._serialized_end=1006
  _globals['_PREDICTEDOBJECTKINEMATICS']._serialized_start=1009
  _globals['_PREDICTEDOBJECTKINEMATICS']._serialized_end=1281
  _globals['_HEADER']._serialized_start=1283
  _globals['_HEADER']._serialized_end=1324
  _globals['_POSEWITHCOVARIANCE']._serialized_start=1326
  _globals['_POSEWITHCOVARIANCE']._serialized_end=1391
  _globals['_TWISTWITHCOVARIANCE']._serialized_start=1393
  _globals['_TWISTWITHCOVARIANCE']._serialized_end=1461
  _globals['_ACCELWITHCOVARIANCE']._serialized_start=1463
  _globals['_ACCELWITHCOVARIANCE']._serialized_end=1531
  _globals['_POSE']._serialized_start=1533
  _globals['_POSE']._serialized_end=1607
  _globals['_POINT']._serialized_start=1609
  _globals['_POINT']._serialized_end=1649
  _globals['_QUATERNION']._serialized_start=1651
  _globals['_QUATERNION']._serialized_end=1707
  _globals['_TWIST']._serialized_start=1709
  _globals['_TWIST']._serialized_end=1777
  _globals['_ACCEL']._serialized_start=1779
  _globals['_ACCEL']._serialized_end=1847
  _globals['_VECTOR3']._serialized_start=1849
  _globals['_VECTOR3']._serialized_end=1891
  _globals['_PREDICTEDPATH']._serialized_start=1893
  _globals['_PREDICTEDPATH']._serialized_end=1987
  _globals['_DURATION']._serialized_start=1989
  _globals['_DURATION']._serialized_end=2029
  _globals['_SHAPE']._serialized_start=2031
  _globals['_SHAPE']._serialized_end=2085
  _globals['_VADSERVICE']._serialized_start=2087
  _globals['_VADSERVICE']._serialized_end=2151
# @@protoc_insertion_point(module_scope)
