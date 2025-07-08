import cv2
import os
import time
import logging

# This file is part of the WMS Camera module.
# It handles camera initialization, frame capture, and error management.
class Camera:
    # Initialize the camera
    def __init__(self):
        self.reload_camera()
    
    # Reload the camera if it is not opened or has failed
    def reload_camera(self):
        if hasattr(self, 'video') and self.video.isOpened():
            self.video.release()
        cam_1 = os.getenv("CAMERA_URL_1", 0)
        cam_2 = os.getenv("CAMERA_URL_2", 0)
        self.video = cv2.VideoCapture(cam_1)
        if not self.video.isOpened() and cam_2:
            self.video = cv2.VideoCapture(cam_2)
        if not self.video.isOpened() and cam_1 and cam_2:
            logging.error("Both cameras failed to open.")
        if not self.video.isOpened():
            raise ValueError("Camera not found")
        self.error_count = 0
    
    # Release the camera when the object is deleted
    def __del__(self):
        if self.video.isOpened():
            self.video.release()
    
    # Get a frame from the camera
    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            self.error_count += 1
            if self.error_count >= 10:
                logging.warning("Camera freeze detected, reloading camera...")
                self.reload_camera()
                time.sleep(5)  # Wait for the camera to stabilize (adjustable)
            return None
        self.error_count = 0
        return frame