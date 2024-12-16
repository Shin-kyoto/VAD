import grpc
from concurrent import futures
import vad_service_pb2
import vad_service_pb2_grpc
import cv2
import numpy as np
import uuid

class VADDummy:
    def __init__(self):
        self.setup_model()
    
    def setup_model(self):
        # Dummy model setup
        pass
        
    def predict(self, image, ego_history, map_data, driving_command):
        # Create dummy prediction
        obj = vad_service_pb2.PredictedObject(
            uuid=str(uuid.uuid4()),
            existence_probability=0.9
        )
        
        # Set kinematics
        kinematics = obj.kinematics
        
        # Initial pose
        kinematics.initial_pose_with_covariance.pose.position.x = 0.0
        kinematics.initial_pose_with_covariance.pose.position.y = 0.0
        kinematics.initial_pose_with_covariance.pose.position.z = 0.0
        kinematics.initial_pose_with_covariance.pose.orientation.w = 1.0
        kinematics.initial_pose_with_covariance.covariance.extend([0.0] * 36)
        
        # Predicted path
        path = kinematics.predicted_paths.add()
        for i in range(10):
            pose = path.path.add()
            pose.position.x = float(i)
            pose.position.y = float(i)
            pose.position.z = 0.0
            pose.orientation.w = 1.0
        
        path.time_step.sec = 0
        path.time_step.nanosec = 100000000  # 0.1 second
        path.confidence = 0.8
        
        return obj

class VADServicer(vad_service_pb2_grpc.VADServiceServicer):
    def __init__(self):
        self.vad_model = VADDummy()
    
    def ProcessData(self, request, context):
        try:
            # Convert image data
            nparr = np.frombuffer(request.image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process with VAD model
            predicted_object = self.vad_model.predict(
                image,
                request.ego_history,
                request.map_data,
                request.driving_command
            )
            
            # Create response
            response = vad_service_pb2.VADResponse()
            if len(request.ego_history) > 0:
                response.header.CopyFrom(request.ego_history[-1].header)
            response.objects.append(predicted_object)
            print("send response")
            return response
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vad_service_pb2_grpc.add_VADServiceServicer_to_server(VADServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("Starting server on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
