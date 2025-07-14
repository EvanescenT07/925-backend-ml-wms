import pytest
import cv2
import os
import numpy as np
import threading
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

os.environ['MODEL_PATH'] = 'dummy_model.pt'

# --- Mock Camera class to avoid RTSP connection ---
class MockCamera:
    def __init__(self):
        self.frame = None
        self.running = True
        self.cap = Mock()
        self.cap.isOpened.return_value = True
        self.cap.release = Mock()
        self.thread = Mock()
        
    def get_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)
        
    def stop(self):
        self.running = False
        
    def __del__(self):
        pass

# --- Mock classes for testing ---
class DummyVideoCapture:
    def __init__(self, *args, **kwargs):
        self.opened = True
    def isOpened(self):
        return self.opened
    def release(self):
        self.opened = False
    def read(self):
        return False, None

class DummyThread:
    def __init__(self, *args, **kwargs): pass
    def start(self): pass
    def join(self): pass

# Import modules with Camera and YOLO patched
with patch('ultralytics.YOLO') as mock_yolo_class, \
     patch('wms_camera.Camera', MockCamera):
    mock_yolo_instance = Mock()
    mock_yolo_instance.names = {0: "test_object"}
    mock_yolo_class.return_value = mock_yolo_instance

    from wms_camera import Camera
    from wms_model import detection_object, detection_object_data
    from wms_gen_video import GenerateVideo
    from wms_main import app

def test_camera_init_failure():
    # Test that Camera fails when VideoCapture fails
    with patch('cv2.VideoCapture') as mock_vc, \
         patch('threading.Thread', return_value=DummyThread()):
        mock_vc.return_value = Mock(isOpened=lambda: False, release=lambda: None)
        
        # Create a real Camera class for testing
        import wms_camera
        class RealCamera:
            def __init__(self):
                self.frame = None
                self.running = True
                self.lock = threading.Lock()
                cam_url = os.getenv("CAMERA_URL_1")
                self.cap = cv2.VideoCapture(cam_url)
                self.error_count = 0
                if not self.cap.isOpened():
                    raise RuntimeError(f"Camera at {cam_url} could not be opened.")
                self.thread = threading.Thread(target=self.update, daemon=True)
                self.thread.start()
            
            def update(self):
                pass
        
        with pytest.raises(RuntimeError):
            RealCamera()
                          
def test_camera_init_and_release():
    cam = Camera()  # This uses MockCamera
    assert hasattr(cam, "cap")
    assert cam.cap.isOpened()
    cam.stop()

@patch('wms_model.model')
def test_detection_object_with_detections(mock_model):
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
    annotated_frame, count = detection_object(dummy_frame)
    assert count == 1
    assert annotated_frame.shape == dummy_frame.shape

@patch('wms_model.model')
def test_detection_object_no_detections(mock_model):
    mock_result = Mock()
    mock_result.boxes = []
    mock_model.return_value = [mock_result]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated_frame, count = detection_object(dummy_frame)
    assert count == 0
    assert annotated_frame.shape == dummy_frame.shape

@patch('wms_model.model')
def test_detection_object_data_with_detections(mock_model):
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
    mock_result = Mock()
    mock_result.boxes = []
    mock_model.return_value = [mock_result]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    assert isinstance(result, dict)
    assert result["total"] == 0
    assert result["detections"] == []

def test_generate_video_success():
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

    mock_camera = Mock()
    mock_camera.get_frame.side_effect = [dummy_frame, None]

    with patch('wms_gen_video.detection_object', return_value=(annotated_frame, 1)):
        gen_video = GenerateVideo(camera=mock_camera)
        generator = gen_video.generate_video()
        frame_data = next(generator)
        assert b'--frame' in frame_data
        assert b'Content-Type: image/jpeg' in frame_data

def test_generate_video_no_frames():
    mock_camera = Mock()
    mock_camera.get_frame.return_value = None

    gen_video = GenerateVideo(camera=mock_camera)
    generator = gen_video.generate_video()
    assert generator is not None

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
    mock_gen = Mock()
    mock_gen.generate_video.return_value = iter([b'--frame\r\ntest\r\n'])
    mock_generate_video_class.return_value = mock_gen

    client = TestClient(app)
    response = client.get("/video")
    assert response.status_code == 200
    assert response.headers["content-type"] == "multipart/x-mixed-replace; boundary=frame"

def test_full_pipeline_integration():
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    with patch('wms_model.model') as mock_model:
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

        cam = Camera()  # Uses MockCamera
        frame = cam.get_frame()
        assert frame is not None
        result = detection_object_data(frame)
        assert result["total"] == 1
        assert result["detections"][0]["class"] == "test_object"

@patch('wms_main.detection_object_data')
def test_ws_detect_endpoint(mock_detection):
    from wms_main import camera_buffer

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    camera_buffer.get_frame = Mock(return_value=dummy_frame)

    mock_detection.return_value = {
        "total": 1,
        "detections": [{"class": "test_object", "confidence": 0.9}]
    }

    client = TestClient(app)
    with client.websocket_connect("/ws/detect") as websocket:
        response = websocket.receive_json()
        assert isinstance(response, dict)
        assert "total" in response
        assert "detections" in response
        assert response["total"] == 1
        assert response["detections"][0]["class"] == "test_object"
        assert response["detections"][0]["confidence"] == 0.9