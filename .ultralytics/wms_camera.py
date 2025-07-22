import cv2
import os
import time
import threading
import logging

class Camera:
    # A class to handle RTSP camera connections and frame retrieval.
    def __init__(self):
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        cam_url = os.getenv("CAMERA_URL_1")
        if not cam_url:
            logging.error("Environment variable CAMERA_URL_1 is not set.")
            raise RuntimeError("CAMERA_URL_1 environment variable is not set.")
        self.cap = cv2.VideoCapture(cam_url)
        self.error_count = 0
        if not self.cap.isOpened():
            logging.error(f"Failed to open camera at {cam_url}")
            raise RuntimeError(f"Camera at {cam_url} could not be opened.")
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                self.error_count = 0
            else:
                self.error_count += 1
                if self.error_count >= 10:
                    logging.warning("Camera freeze detected, attempting to reload camera...")
                    self.reload_camera()
                    time.sleep(2)
            time.sleep(0.01)

    def reload_camera(self):
        cam_1 = os.getenv("CAMERA_URL_1")
        cam_2 = os.getenv("CAMERA_URL_2")
        logging.info(f"Attempting to open RTSP camera: {cam_1}")
        self.cap.release()
        if cam_1 is not None:
            self.cap = cv2.VideoCapture(str(cam_1))
        else:
            self.cap = cv2.VideoCapture()
        if not self.cap.isOpened() and cam_2 is not None:
            logging.warning(f"Failed to open {cam_1}, trying backup: {cam_2}")
            self.cap = cv2.VideoCapture(str(cam_2))
        if not self.cap.isOpened():
            logging.error("Both RTSP cameras failed to open.")
            raise RuntimeError("Camera not found")
        self.error_count = 0
        logging.info("RTSP camera opened successfully.")

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join()
        self.cap.release()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass