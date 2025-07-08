import cv2
import time
import os
import logging

# This file is part of the WMS Camera module.
# It handles camera initialization, frame capture, and error management.
class Camera:
    def __init__(self):
        self.reload_camera()
      
    # Reloads the camera based on environment variables.
    # It checks for CAMERA_URL_1 and CAMERA_URL_2, and initializes the camera  
    def reload_camera(self):
        if hasattr(self, 'video') and self.video.isOpened():
            self.video.release()
        cam_1 = os.getenv("CAMERA_URL_1")
        cam_2 = os.getenv("CAMERA_URL_2")
        self.video = cv2.VideoCapture(cam_1)
        if not self.video.isOpened() and cam_2:
            self.video = cv2.VideoCapture(cam_2)
        if not self.video.isOpened() and cam_1 and cam_2:
            logging.error("Both cameras failed to open.")
        if not self.video.isOpened():
            raise ValueError("Camera not found")
        self.error_count = 0
        
    # Destructor to release the camera when the object is deleted.
    # This ensures that the camera resource is properly released.
    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    # Captures a frame from the camera.
    # If the camera fails to capture a frame, it increments an error count.
    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            self.error_count += 1
            if self.error_count >= 10:
                logging.warning("Camera freeze detected, reloading camera...")
                self.reload_camera()
                time.sleep(5)  # time reload
            return None
        self.error_count = 0
        return frame