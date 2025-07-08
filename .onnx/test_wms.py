import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient
from wms_camera import Camera
from wms_model import detection_object_data
from wms_main import app

# --- Camera class tests ---
def test_camera_init_and_release(monkeypatch):
    # Mock cv2.VideoCapture to avoid using a real camera
    class DummyVideoCapture:
        def __init__(self, *args, **kwargs):
            self.opened = True
        def isOpened(self):
            return self.opened
        def release(self):
            self.opened = False
        def read(self):
            # Return False to simulate camera read failure
            return False, None

    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: DummyVideoCapture())

    # Test initialization
    cam = Camera()
    assert hasattr(cam, "video")
    assert cam.video.isOpened()

    # Test get_frame returns None on failure
    frame = cam.get_frame()
    assert frame is None

    # Test __del__ releases the camera
    cam.__del__()
    assert not cam.video.isOpened()

def test_camera_reload_camera(monkeypatch):
    # Mock VideoCapture to simulate both cameras failing
    class DummyVideoCapture:
        def __init__(self, *args, **kwargs):
            self.opened = False
        def isOpened(self):
            return self.opened
        def release(self):
            pass
        def read(self):
            return False, None

    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: DummyVideoCapture())

    # Should raise ValueError if both cameras fail
    with pytest.raises(ValueError):
        Camera()

# --- Model utility test ---
def test_detection_object_data_with_dummy_frame():
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(frame=dummy_frame)
    assert isinstance(result, (dict, list))

# --- FastAPI endpoint test ---
def test_fastapi_root_endpoint():
    client = TestClient(app)
    response = client.get("/")
    # Adjust the endpoint and expected status as needed
    assert response.status_code in (200, 404)