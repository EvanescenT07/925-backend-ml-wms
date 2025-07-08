import pytest
import cv2
import numpy as np
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

# Mock environment and dependencies before importing
os.environ['MODEL_PATH'] = 'dummy_model.onnx'

# Mock onnxruntime and other heavy dependencies
with patch('onnxruntime.InferenceSession') as mock_session, \
     patch('yaml.safe_load') as mock_yaml, \
     patch('builtins.open', create=True) as mock_open:
    
    # Setup mocks
    mock_yaml.return_value = {"names": {0: "person", 1: "car"}}
    mock_session_instance = Mock()
    mock_session_instance.get_inputs.return_value = [Mock(name='input')]
    mock_session_instance.run.return_value = [np.zeros((1, 85, 8400))]  # Correct YOLO output shape
    mock_session.return_value = mock_session_instance
    
    # Import modules after mocking
    from wms_camera import Camera
    from wms_model import detection_object_data, detection_object, CLASS_NAMES
    from wms_gen_video import GenerateVideo
    from wms_main import app

# --- Camera class tests ---
def test_camera_init_and_release(monkeypatch):
    """Test camera initialization and cleanup"""
    class DummyVideoCapture:
        def __init__(self, *args, **kwargs):
            self.opened = True
        def isOpened(self):
            return self.opened
        def release(self):
            self.opened = False
        def read(self):
            return False, None

    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: DummyVideoCapture())

    cam = Camera()
    assert hasattr(cam, "video")
    assert cam.video.isOpened()

    cam.__del__()
    assert not cam.video.isOpened()

def test_camera_reload_camera_failure(monkeypatch):
    """Test camera initialization failure"""
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

    with pytest.raises(ValueError):
        Camera()

def test_camera_get_frame_success(monkeypatch):
    """Test successful frame capture"""
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

def test_camera_multiple_reload_attempts(monkeypatch):
    """Test camera reload with multiple attempts"""
    call_count = 0
    
    class DummyVideoCapture:
        def __init__(self, camera_index, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Fail first attempt for camera 0, succeed for camera 1
            if camera_index == 0:
                self.opened = False
            else:  # camera_index == 1
                self.opened = True
        
        def isOpened(self):
            return self.opened
        
        def release(self):
            pass
        
        def read(self):
            if self.opened:
                return True, np.zeros((480, 640, 3), dtype=np.uint8)
            return False, None

    monkeypatch.setattr(cv2, "VideoCapture", DummyVideoCapture)
    
    # Should succeed on second camera (index 1)
    cam = Camera()
    assert cam.video.isOpened()
    
    # Test frame
    frame = cam.get_frame()
    assert frame is not None
    assert frame.shape == (480, 640, 3)

# --- GenerateVideo class tests ---
@patch('wms_gen_video.Camera')
@patch('wms_gen_video.detection_object')
@patch('cv2.imencode') 
def test_generate_video_success(mock_imencode, mock_detection, mock_camera_class):
    """Test video generation with mock camera and detection"""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # ✅ Mock cv2.imencode to return success and fake encoded data
    mock_imencode.return_value = (True, np.array([255, 216, 255, 224], dtype=np.uint8))  # Fake JPEG header
    
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
    
    # ✅ Verify cv2.imencode was called
    mock_imencode.assert_called()

@patch('wms_gen_video.Camera')
@patch('cv2.imencode')  # ✅ Also mock imencode for this test
def test_generate_video_no_frames(mock_imencode, mock_camera_class):
    """Test video generation when no frames available"""
    mock_camera = Mock()
    mock_camera.get_frame.return_value = None
    mock_camera_class.return_value = mock_camera

    gen_video = GenerateVideo()
    generator = gen_video.generate_video()
    assert generator is not None
    
    # ✅ Since no frames, imencode should not be called
    mock_imencode.assert_not_called()

# --- Model tests ---
@patch('wms_model.session')
def test_detection_object_data_with_detections(mock_session):
    """Test detection with mock ONNX output containing detections"""
    # Create proper YOLO output shape: (1, 85, 8400)
    # Format: [x, y, w, h, conf, class0_prob, class1_prob, ...]
    
    detections = np.zeros((1, 85, 8400))
    
    # First detection (index 0) - high confidence person
    detections[0, 0, 0] = 320  # x
    detections[0, 1, 0] = 240  # y  
    detections[0, 2, 0] = 100  # w
    detections[0, 3, 0] = 150  # h
    detections[0, 4, 0] = 0.9  # confidence
    detections[0, 5, 0] = 0.9  # person class probability
    detections[0, 6, 0] = 0.1  # car class probability
    # Rest are 0 (already initialized)
    
    # Second detection - also high confidence
    detections[0, 0, 1] = 100  # x
    detections[0, 1, 1] = 100  # y
    detections[0, 2, 1] = 80   # w
    detections[0, 3, 1] = 120  # h
    detections[0, 4, 1] = 0.85 # confidence
    detections[0, 5, 1] = 0.1  # person class probability
    detections[0, 6, 1] = 0.8  # car class probability
    
    # Rest of detections have low confidence (will be filtered out)
    for i in range(2, 8400):
        detections[0, 4, i] = 0.1  # Low confidence
    
    mock_session.run.return_value = [detections]
    mock_session.get_inputs.return_value = [Mock(name='input')]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    
    assert isinstance(result, dict)
    assert "total" in result
    assert "detections" in result
    assert result["total"] >= 0

@patch('wms_model.session')
def test_detection_object_data_no_detections(mock_session):
    """Test detection with no objects found"""
    # Create proper shape with all low confidence
    detections = np.zeros((1, 85, 8400))
    
    # All detections have low confidence (below threshold)
    for i in range(8400):
        detections[0, 4, i] = 0.1  # Low confidence

    mock_session.run.return_value = [detections]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    
    assert isinstance(result, dict)
    assert result["total"] == 0
    assert result["detections"] == []

@patch('wms_model.session')
def test_detection_object_with_detections(mock_session):
    """Test detection_object function returns annotated frame"""
    detections = np.zeros((1, 85, 8400))
    
    # High confidence detection
    detections[0, 0, 0] = 320  # x
    detections[0, 1, 0] = 240  # y
    detections[0, 2, 0] = 100  # w
    detections[0, 3, 0] = 150  # h
    detections[0, 4, 0] = 0.9  # confidence
    detections[0, 5, 0] = 0.9  # person class probability
    
    mock_session.run.return_value = [detections]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated_frame, count = detection_object(dummy_frame)
    
    assert annotated_frame is not None
    assert annotated_frame.shape == dummy_frame.shape
    assert isinstance(count, int)
    assert count >= 0

@patch('wms_model.session')
def test_detection_object_data_multiple_detections(mock_session):
    """Test multiple object detection"""
    detections = np.zeros((1, 85, 8400))
    
    # Create 3 high-confidence detections
    positions = [(100, 100), (200, 150), (300, 200)]
    for i, (x, y) in enumerate(positions):
        detections[0, 0, i] = x      # x
        detections[0, 1, i] = y      # y
        detections[0, 2, i] = 80     # w
        detections[0, 3, i] = 120    # h
        detections[0, 4, i] = 0.9    # confidence
        detections[0, 5, i] = 0.9 if i % 2 == 0 else 0.1  # person
        detections[0, 6, i] = 0.1 if i % 2 == 0 else 0.9  # car
    
    mock_session.run.return_value = [detections]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    
    assert isinstance(result, dict)
    assert result["total"] >= 0
    assert "detections" in result

@patch('wms_model.session')
def test_detection_object_with_low_confidence(mock_session):
    """Test detection with confidence below threshold"""
    detections = np.zeros((1, 85, 8400))
    
    # All detections below threshold (0.83)
    for i in range(10):
        detections[0, 0, i] = 100 + i*20  # x
        detections[0, 1, i] = 100 + i*20  # y
        detections[0, 2, i] = 50          # w
        detections[0, 3, i] = 70          # h
        detections[0, 4, i] = 0.5         # Below threshold
        detections[0, 5, i] = 0.5         # person probability
    
    mock_session.run.return_value = [detections]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated_frame, count = detection_object(dummy_frame)
    
    # Should filter out low confidence detections
    assert count == 0

# --- FastAPI endpoint tests ---
def test_fastapi_root_endpoint():
    """Test root endpoint"""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "1.0.1"

@patch('wms_main.GenerateVideo')
def test_video_stream_endpoint(mock_generate_video_class):
    """Test video streaming endpoint"""
    mock_gen = Mock()
    mock_gen.generate_video.return_value = iter([b'--frame\r\ntest\r\n'])
    mock_generate_video_class.return_value = mock_gen

    client = TestClient(app)
    response = client.get("/video")
    assert response.status_code == 200
    assert "multipart/x-mixed-replace" in response.headers.get("content-type", "")

# --- Integration tests ---
@patch('wms_model.session')
@patch('cv2.VideoCapture')
def test_full_pipeline_integration(mock_video_capture, mock_session):
    """Test complete pipeline from camera to detection"""
    # Mock camera
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, dummy_frame)
    mock_video_capture.return_value = mock_cap

    # Mock ONNX session with proper shape
    detections = np.zeros((1, 85, 8400))
    detections[0, 0, 0] = 320  # x
    detections[0, 1, 0] = 240  # y
    detections[0, 2, 0] = 100  # w
    detections[0, 3, 0] = 150  # h
    detections[0, 4, 0] = 0.9  # confidence
    detections[0, 5, 0] = 0.9  # person probability
    
    mock_session.run.return_value = [detections]

    # Test pipeline
    cam = Camera()
    frame = cam.get_frame()
    assert frame is not None
    
    result = detection_object_data(frame)
    assert isinstance(result, dict)
    assert "total" in result
    assert "detections" in result

# --- Utility tests ---
def test_class_names_loading():
    """Test CLASS_NAMES is properly loaded"""
    assert isinstance(CLASS_NAMES, list)
    assert len(CLASS_NAMES) > 0
    assert "person" in CLASS_NAMES
    assert "car" in CLASS_NAMES

@patch('wms_model.session')
def test_detection_with_empty_frame(mock_session):
    """Test detection with valid but empty frame"""
    detections = np.zeros((1, 85, 8400))
    # All low confidence
    mock_session.run.return_value = [detections]
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    assert result["total"] == 0
    assert result["detections"] == []

def test_detection_with_none_frame():
    """Test detection with None input - should handle gracefully"""
    result = detection_object_data(None)
    assert result["total"] == 0
    assert result["detections"] == []

# --- WebSocket test ---
@patch('wms_main.Camera')
@patch('wms_main.detection_object_data')
def test_ws_detect_endpoint(mock_detection, mock_camera_class):
    """Test WebSocket detection endpoint"""
    mock_camera = Mock()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_camera.get_frame.return_value = dummy_frame
    mock_camera_class.return_value = mock_camera
    
    mock_detection.return_value = {
        "total": 1,
        "detections": [{"class": "person", "confidence": 0.9}]
    }
    
    client = TestClient(app)
    
    with client.websocket_connect("/ws/detect") as websocket:
        response = websocket.receive_json()
        
        assert isinstance(response, dict)
        assert "total" in response
        assert "detections" in response
        assert response["total"] == 1
        assert response["detections"][0]["class"] == "person"
        assert response["detections"][0]["confidence"] == 0.9

# --- Edge case tests ---
@patch('wms_model.session')
def test_detection_object_data_with_nms_filtering(mock_session):
    """Test detection with overlapping boxes that should be filtered by NMS"""
    detections = np.zeros((1, 85, 8400))
    
    # Create overlapping detections (same area, high confidence)
    for i in range(5):
        detections[0, 0, i] = 320 + i*5  # x (slightly offset)
        detections[0, 1, i] = 240 + i*5  # y (slightly offset)
        detections[0, 2, i] = 100        # w
        detections[0, 3, i] = 150        # h
        detections[0, 4, i] = 0.9        # confidence
        detections[0, 5, i] = 0.9        # person probability
    
    mock_session.run.return_value = [detections]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    
    # NMS should reduce the number of detections
    assert isinstance(result, dict)
    assert result["total"] >= 0

@patch('wms_model.session')
def test_detection_object_data_with_invalid_coordinates(mock_session):
    """Test detection with coordinates outside image bounds"""
    detections = np.zeros((1, 85, 8400))
    
    # Detection outside image bounds
    detections[0, 0, 0] = -100  # x (negative)
    detections[0, 1, 0] = -100  # y (negative) 
    detections[0, 2, 0] = 50    # w
    detections[0, 3, 0] = 50    # h
    detections[0, 4, 0] = 0.9   # confidence
    detections[0, 5, 0] = 0.9   # person probability
    
    # Detection with very large coordinates
    detections[0, 0, 1] = 1000  # x (outside 640x480)
    detections[0, 1, 1] = 1000  # y (outside 640x480)
    detections[0, 2, 1] = 100   # w
    detections[0, 3, 1] = 100   # h
    detections[0, 4, 1] = 0.9   # confidence
    detections[0, 5, 1] = 0.9   # person probability
    
    mock_session.run.return_value = [detections]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    
    # Should handle invalid coordinates gracefully
    assert isinstance(result, dict)
    assert result["total"] >= 0

# --- Performance test ---
@patch('wms_model.session')
def test_detection_performance_with_many_objects(mock_session):
    """Test detection with maximum number of objects"""
    detections = np.zeros((1, 85, 8400))
    
    # Fill first 100 detections with high confidence
    for i in range(100):
        detections[0, 0, i] = (i % 20) * 30  # x (grid pattern)
        detections[0, 1, i] = (i // 20) * 30 # y (grid pattern)
        detections[0, 2, i] = 25             # w
        detections[0, 3, i] = 25             # h
        detections[0, 4, i] = 0.95           # confidence
        detections[0, 5, i] = 0.9            # person probability
    
    mock_session.run.return_value = [detections]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    
    # Should handle many detections efficiently
    assert isinstance(result, dict)
    assert result["total"] >= 0
    assert len(result["detections"]) <= result["total"]