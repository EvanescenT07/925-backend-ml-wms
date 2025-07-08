import pytest
import cv2
import numpy as np
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from wms_camera import Camera
from wms_model import detection_object, detection_object_data
from wms_gen_video import GenerateVideo
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

    # Test __del__ releases the camera
    cam.__del__()
    assert not cam.video.isOpened()

def test_camera_reload_camera_failure(monkeypatch):
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

def test_camera_get_frame_success(monkeypatch):
    # Mock successful frame capture
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    class DummyVideoCapture:
        def __init__(self, *args, **kwargs):
            self.opened = True
        def isOpened(self):
            return self.opened
        def release(self):
            self.opened = False
        def read(self):
            return True, dummy_frame

    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: DummyVideoCapture())

    cam = Camera()
    frame = cam.get_frame()
    assert frame is not None
    assert frame.shape == (480, 640, 3)

# --- Model tests ---
@patch('wms_model.model')
def test_detection_object_with_detections(mock_model):
    # Mock YOLO model result
    mock_box = Mock()
    mock_box.conf = [Mock()]
    mock_box.conf[0].item.return_value = 0.9  # High confidence
    mock_box.xyxy = [np.array([10, 20, 100, 200])]
    mock_box.cls = [Mock()]
    mock_box.cls[0].item.return_value = 0  # Class ID

    mock_result = Mock()
    mock_result.boxes = [mock_box]
    
    mock_model.return_value = [mock_result]
    mock_model.names = {0: "test_object"}

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated_frame, count = detection_object(dummy_frame)
    
    assert count == 1
    assert annotated_frame.shape == dummy_frame.shape

@patch('wms_model.model')
def test_detection_object_no_detections(mock_model):
    # Mock YOLO model with no detections
    mock_result = Mock()
    mock_result.boxes = []
    
    mock_model.return_value = [mock_result]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated_frame, count = detection_object(dummy_frame)
    
    assert count == 0
    assert annotated_frame.shape == dummy_frame.shape

@patch('wms_model.model')
def test_detection_object_data_with_detections(mock_model):
    # Mock YOLO model result for data function
    mock_box = Mock()
    mock_box.conf = [Mock()]
    mock_box.conf[0].item.return_value = 0.9
    mock_box.xyxy = [np.array([10, 20, 100, 200])]
    mock_box.cls = [Mock()]
    mock_box.cls[0].item.return_value = 0

    mock_result = Mock()
    mock_result.boxes = [mock_box]
    
    mock_model.return_value = [mock_result]
    mock_model.names = {0: "test_object"}

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    
    assert isinstance(result, dict)
    assert result["total"] == 1
    assert len(result["detections"]) == 1
    assert result["detections"][0]["class"] == "test_object"
    assert result["detections"][0]["confidence"] == 0.9

@patch('wms_model.model')
def test_detection_object_data_no_detections(mock_model):
    # Mock YOLO model with no detections
    mock_result = Mock()
    mock_result.boxes = []
    
    mock_model.return_value = [mock_result]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    
    assert isinstance(result, dict)
    assert result["total"] == 0
    assert result["detections"] == []

# --- GenerateVideo class tests ---
@patch('wms_gen_video.Camera')
@patch('wms_gen_video.detection_object')
def test_generate_video_success(mock_detection, mock_camera_class):
    # Mock camera and detection
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    mock_camera = Mock()
    mock_camera.get_frame.side_effect = [dummy_frame, None]  # One frame then None to stop
    mock_camera_class.return_value = mock_camera
    
    mock_detection.return_value = (annotated_frame, 1)

    gen_video = GenerateVideo()
    generator = gen_video.generate_video()
    
    # Get one frame
    frame_data = next(generator)
    assert b'--frame' in frame_data
    assert b'Content-Type: image/jpeg' in frame_data

@patch('wms_gen_video.Camera')
def test_generate_video_no_frames(mock_camera_class):
    # Mock camera that returns no frames
    mock_camera = Mock()
    mock_camera.get_frame.return_value = None
    mock_camera_class.return_value = mock_camera

    gen_video = GenerateVideo()
    generator = gen_video.generate_video()
    assert generator is not None

# --- FastAPI endpoint tests ---
def test_fastapi_root_endpoint():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "1.0.1"

@patch('wms_main.GenerateVideo')
def test_video_stream_endpoint(mock_generate_video_class):
    # Mock GenerateVideo
    mock_gen = Mock()
    mock_gen.generate_video.return_value = iter([b'--frame\r\ntest\r\n'])
    mock_generate_video_class.return_value = mock_gen

    client = TestClient(app)
    response = client.get("/video")
    assert response.status_code == 200
    assert response.headers["content-type"] == "multipart/x-mixed-replace; boundary=frame"

# --- Integration test ---
@patch('wms_model.model')
@patch('cv2.VideoCapture')
def test_full_pipeline_integration(mock_video_capture, mock_model):
    # Mock camera
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, dummy_frame)
    mock_video_capture.return_value = mock_cap

    # Mock model
    mock_box = Mock()
    mock_box.conf = [Mock()]
    mock_box.conf[0].item.return_value = 0.9
    mock_box.xyxy = [np.array([10, 20, 100, 200])]
    mock_box.cls = [Mock()]
    mock_box.cls[0].item.return_value = 0

    mock_result = Mock()
    mock_result.boxes = [mock_box]
    mock_model.return_value = [mock_result]
    mock_model.names = {0: "test_object"}

    # Test complete pipeline
    cam = Camera()
    frame = cam.get_frame()
    assert frame is not None
    
    result = detection_object_data(frame)
    assert result["total"] == 1
    assert result["detections"][0]["class"] == "test_object"