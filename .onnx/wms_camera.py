import cv2
import time
import os
import logging
import threading
from collections import deque

# Camera class instance to handle RTSP camera connections and frame retrieval.
class Camera:
    def __init__(self):
        self.running = True
        self.lock = threading.Lock()
        self.frame_buffer = deque(maxlen=10)  # Buffer to store frames
        self.consecutive_errors = 0
        self.error_count = 0
        self.reload_camera()
      
        self.thread = threading.Thread(target=self.update_frame, daemon=True)
        self.thread.start()

    def reload_camera(self):
        if hasattr(self, 'video') and self.video is not None and self.video.isOpened():
            self.video.release()
        
        camera_sources = [
            os.getenv("CAMERA_URL_1"),
            os.getenv("CAMERA_URL_2")
        ]
        
        for source in camera_sources:
            if source and self.connect_camera(source):
                logging.info(f"✅ Camera connected: {source}")
                self.consecutive_errors = 0
                return True
            
        logging.error("All camera connections failed!")
        self.video = None # type: ignore
        return False

    def connect_camera(self, camera_url):
        try:
            self.video = cv2.VideoCapture(camera_url)
            
            if isinstance(camera_url, str) and camera_url.startswith("rtsp://"):
                self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.video.set(cv2.CAP_PROP_FPS, 30)
                self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
            ret, frame = self.video.read()
            if ret and frame is not None:
                logging.info(f"Camera connected: {camera_url}")
                return True
            else:
                logging.warning(f"Failed to read frame from camera: {camera_url}")
                if self.video:
                    self.video.release()
                self.video = None
                return False
            
        except Exception as e:
            logging.error(f"Error connecting to camera {camera_url}: {e}")
            if hasattr(self, 'video') and self.video is not None and self.video.isOpened():
                self.video.release()
            self.video = None
            return False
                
    def update_frame(self):
        while self.running:
            if hasattr(self, 'video') and self.video is not None and self.video.isOpened():
                ret, frame = self.video.read()
                if ret and frame is not None:
                    with self.lock:
                        self.frame_buffer.append(frame)
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= 10:
                        logging.warning("Camera connection lost, reloading...")
                        if not self.reload_camera():
                            logging.error("Camera reload failed, waiting...")
                            time.sleep(3)
                        self.consecutive_failures = 0
            else:
                time.sleep(0.1)
                
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.frame_buffer:
                return self.frame_buffer[-1].copy()  # Return latest frame
            return None
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1)
        if hasattr(self, 'video') and self.video is not None and self.video.isOpened():
            self.video.release()
            
    def __del__(self):
        self.stop()
        logging.info("Camera resources released.")