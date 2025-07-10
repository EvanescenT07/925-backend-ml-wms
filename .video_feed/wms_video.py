from wms_model import detection_object
import cv2
import logging
import time

class VideoFrame:
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise ValueError(f"Video file {video_path} not found or cannot be opened.")
        self.frame_count = 0
        
    def __del__(self):
        if self.video.isOpened():
            self.video.release()
            
    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video.read()
        if not ret:
            return None
        self.frame_count += 1
        return frame
    
    def generate_video(self):
        try:
            while True:
                frame = self.get_frame()
                if frame is None:
                    logging.warning("No more frames to read, restarting video.")
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
            if self.video.isOpened():
                self.video.release()
            logging.info("Video generation completed.")
            return
        