import cv2
from wms_camera import Camera
from wms_model import detection_object   

class GenerateVideo:
    def __init__(self):
        self.cam = Camera()

    def generate_video(self):
        try:
            while True:
                frame = self.cam.get_frame()
                if frame is None:
                    continue
                frame = cv2.flip(frame, 1)
                result = detection_object(frame)    
                _, jpeg = cv2.imencode('.jpg', result)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        finally:
            del self.cam
