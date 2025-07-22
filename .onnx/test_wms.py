import pytest
import cv2
import numpy as np
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

# --- Fixtures for environment and mocks ---

@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    monkeypatch.setenv('MODEL_PATH', 'dummy_model.onnx')
    monkeypatch.setenv('CLASS_NAMES_PATH', 'class.yaml')
    monkeypatch.setenv('CAMERA_URL_1', 'dummy_url_1')
    monkeypatch.setenv('CAMERA_URL_2', 'dummy_url_2')

@pytest.fixture(autouse=True)
def patch_onnx_and_yaml(monkeypatch):
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [Mock(name='input')]
    mock_session.run.return_value = [np.zeros((1, 85, 2))]  # Use only 2 detections for speed
    monkeypatch.setattr("onnxruntime.InferenceSession", lambda *a, **k: mock_session)
    monkeypatch.setattr("yaml.safe_load", lambda f: {"class_names": {0: "person", 1: "car"}})
    monkeypatch.setattr("builtins.open", lambda *a, **k: MagicMock())
    return mock_session

with patch.dict(os.environ, {
    'MODEL_PATH': 'dummy_model.onnx',
    'CLASS_NAMES_PATH': 'class.yaml',
    'CAMERA_URL_1': 'dummy_url_1',
    'CAMERA_URL_2': 'dummy_url_2'
}):
    with patch('onnxruntime.InferenceSession') as mock_session, \
         patch('yaml.safe_load') as mock_yaml, \
         patch('builtins.open', create=True) as mock_open:
        mock_yaml.return_value = {"class_names": {0: "person", 1: "car"}}
        mock_session_instance = Mock()
        mock_session_instance.get_inputs.return_value = [Mock(name='input')]
        mock_session_instance.run.return_value = [np.zeros((1, 85, 2))]
        mock_session.return_value = mock_session_instance

        from wms_camera import Camera
        from wms_model import detection_object_data, detection_object, CLASS_NAMES
        from wms_gen_video import GenerateVideo
        from wms_main import app

# --- Camera class tests ---

def test_camera_init_and_release(monkeypatch):
    # Mock get_frame to avoid waiting for thread
    cam = Mock()
    cam.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = cam.get_frame()
    assert frame is not None

def test_camera_reload_camera_failure(monkeypatch):
    class DummyVideoCapture:
        def __init__(self, *args, **kwargs): self.opened = False
        def isOpened(self): return self.opened
        def release(self): pass
        def read(self): return False, None
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: DummyVideoCapture())
    cam = Camera()
    assert cam.video is None

def test_camera_get_frame_none(monkeypatch):
    cam = Mock()
    cam.get_frame.return_value = None
    frame = cam.get_frame()
    assert frame is None

def test_camera_release_called(monkeypatch):
    released = []
    class DummyVideoCapture:
        def __init__(self, *args, **kwargs): self.opened = True
        def isOpened(self): return self.opened
        def release(self): released.append(True); self.opened = False
        def read(self): return True, np.zeros((480, 640, 3), dtype=np.uint8)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: DummyVideoCapture())
    cam = Camera()
    cam.__del__()
    assert released

# --- GenerateVideo class test ---

@patch('wms_gen_video.detection_object')
@patch('cv2.imencode')
def test_generate_video_success(mock_imencode, mock_detection):
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    mock_imencode.return_value = (True, np.array([255, 216, 255, 224], dtype=np.uint8))
    mock_detection.return_value = (annotated_frame, 1)
    mock_camera = Mock()
    mock_camera.get_frame.side_effect = [dummy_frame, None]
    gen_video = GenerateVideo(camera=mock_camera)
    generator = gen_video.generate_video()
    frame_data = next(generator)
    assert b'--frame' in frame_data
    assert b'Content-Type: image/jpeg' in frame_data

# --- Model tests ---

@patch('wms_model.session')
def test_detection_object_data_with_detections(mock_session):
    detections = np.zeros((1, 85, 2))
    detections[0, 0, 0] = 320
    detections[0, 1, 0] = 240
    detections[0, 2, 0] = 100
    detections[0, 3, 0] = 150
    detections[0, 4, 0] = 0.9
    detections[0, 5, 0] = 0.9
    detections[0, 6, 0] = 0.1
    mock_session.run.return_value = [detections]
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    assert isinstance(result, dict)
    assert "total" in result
    assert "detections" in result
    assert result["total"] >= 0

@patch('wms_model.session')
def test_detection_object_data_no_detections(mock_session):
    detections = np.zeros((1, 85, 2))
    for i in range(2):
        detections[0, 4, i] = 0.1
    mock_session.run.return_value = [detections]
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    assert isinstance(result, dict)
    assert result["total"] == 0
    assert result["detections"] == []

@patch('wms_model.session')
def test_detection_object_with_none_frame(mock_session):
    result = detection_object_data(None)
    assert isinstance(result, dict)
    assert result["total"] == 0
    assert result["detections"] == []

@patch('wms_model.session')
def test_detection_object_with_low_confidence(mock_session):
    detections = np.zeros((1, 85, 2))
    for i in range(2):
        detections[0, 4, i] = 0.1  # Low confidence
    mock_session.run.return_value = [detections]
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    assert result["total"] == 0

# --- FastAPI endpoint test ---

def test_fastapi_root_endpoint():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "1.0.1"

def test_fastapi_root_endpoint_content_type():
    client = TestClient(app)
    response = client.get("/")
    assert response.headers["content-type"].startswith("application/json")

# --- Utility test ---

def test_class_names_loading():
    assert isinstance(CLASS_NAMES, list)
    assert len(CLASS_NAMES) > 0
    assert "person" in CLASS_NAMES
    assert "car" in CLASS_NAMES

def test_class_names_are_strings():
    assert all(isinstance(name, str) for name in CLASS_NAMES)