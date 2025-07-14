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
        
       # Try camera 1
        if cam_1:
            logging.info(f"Attempting to connect to camera 1: {cam_1}")
            try:
                self.video = cv2.VideoCapture(cam_1)
                if cam_1.startswith('http'):
                    self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.video.set(cv2.CAP_PROP_FPS, 15)  # Reduced FPS
                    
                # Test if camera actually works by reading a frame
                ret, frame = self.video.read()
                if ret and frame is not None:
                    logging.info("Successfully connected to camera 1")
                    self.error_count = 0
                    return
                else:
                    logging.warning("Failed to read frame from camera 1")
                    self.video.release()
            except Exception as e:
                logging.error(f"Error connecting to camera 1: {e}")
                if self.video:
                    self.video.release()
        
        # Try camera 2 if camera 1 failed
        if cam_2:
            logging.info(f"Attempting to connect to camera 2: {cam_2}")
            try:
                self.video = cv2.VideoCapture(cam_2)
                if cam_2.startswith('http'):
                    self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.video.set(cv2.CAP_PROP_FPS, 15)  # Reduced FPS
                    
                # Test if camera actually works by reading a frame
                ret, frame = self.video.read()
                if ret and frame is not None:
                    logging.info("Successfully connected to camera 2")
                    self.error_count = 0
                    return
                else:
                    logging.warning("Failed to read frame from camera 2")
                    self.video.release()
            except Exception as e:
                logging.error(f"Error connecting to camera 2: {e}")
                if self.video:
                    self.video.release()
        
        # If both cameras failed, create a dummy camera instead of raising exception
        if cam_1 and cam_2:
            logging.error("Both cameras failed to open. Creating dummy camera.")
        elif not cam_1 and not cam_2:
            logging.error("No camera URLs configured in environment variables. Creating dummy camera.")
        
        
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
                time.sleep(2)  # time reload
            return None
        self.error_count = 0
        return frame