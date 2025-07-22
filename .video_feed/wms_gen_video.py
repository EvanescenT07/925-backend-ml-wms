from wms_video import VideoFrame
from wms_model import detection_object
import cv2
import logging
from typing import Optional

class GenerateVideo:
    video_writer: Optional[cv2.VideoWriter] = None  # Type hint for video writer

    def __init__(self):
        self.video_writer = None
        self.frame_width = 640
        self.frame_height = 480  
        
    def generate_video(self, video_path):
        video = VideoFrame(video_path)
        try:
            while True:
                frame = video.get_frame()
                if frame is None:
                    continue
                frame = cv2.flip(frame, 1)
                result, _ = detection_object(frame)
                _, jpeg = cv2.imencode('.jpg', result)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        except Exception as e:
            logging.error(f"Error generating video: {e}")
        finally:
            if self.video_writer is not None:
                self.video_writer.release()
            del video