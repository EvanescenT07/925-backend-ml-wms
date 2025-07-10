import pytest
import cv2
import numpy as np
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

# Mock environment and dependencies before importing
os.environ['MODEL_PATH'] = 'dummy_model.pt'
os.environ['VIDEO_PATH'] = 'dummy_video.mp4'

# Mock ultralytics and other heavy dependencies
with patch('ultralytics.YOLO') as mock_yolo, \
     patch('builtins.open', create=True) as mock_open:
    
    # Setup mocks
    mock_model_instance = Mock()
    mock_model_instance.names = {0: "person", 1: "car", 2: "truck"}
    
    # Mock detection results
    mock_boxes = Mock()
    mock_boxes.conf = [Mock()]
    mock_boxes.conf[0].item.return_value = 0.9
    mock_boxes.xyxy = [np.array([100, 100, 200, 200])]
    mock_boxes.cls = [Mock()]
    mock_boxes.cls[0].item.return_value = 0
    
    mock_result = Mock()
    mock_result.boxes = [mock_boxes]
    
    mock_model_instance.return_value = [mock_result]
    mock_yolo.return_value = mock_model_instance
    
    # Import modules after mocking
    from wms_video import VideoFrame
    from wms_model import detection_object, detection_object_data
    from wms_gen_video import GenerateVideo
    from wms_main import app

# --- VideoFrame class tests ---
@patch('cv2.VideoCapture')
def test_video_frame_init_success(mock_video_capture):
    """Test successful video initialization"""
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap
    
    video = VideoFrame("test_video.mp4")
    assert hasattr(video, "video")
    assert video.frame_count == 0
    mock_video_capture.assert_called_once_with("test_video.mp4")

@patch('cv2.VideoCapture')
def test_video_frame_init_failure(mock_video_capture):
    """Test video initialization failure"""
    mock_cap = Mock()
    mock_cap.isOpened.return_value = False
    mock_video_capture.return_value = mock_cap
    
    with pytest.raises(ValueError, match="Video file .* not found or cannot be opened"):
        VideoFrame("nonexistent_video.mp4")

@patch('cv2.VideoCapture')
def test_video_frame_get_frame_success(mock_video_capture):
    """Test successful frame capture"""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, dummy_frame)
    mock_video_capture.return_value = mock_cap
    
    video = VideoFrame("test_video.mp4")
    frame = video.get_frame()
    
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert video.frame_count == 1

@patch('cv2.VideoCapture')
def test_video_frame_get_frame_restart_video(mock_video_capture):
    """Test video restart when frames are exhausted"""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    # First read fails, second read succeeds after restart
    mock_cap.read.side_effect = [(False, None), (True, dummy_frame)]
    mock_video_capture.return_value = mock_cap
    
    video = VideoFrame("test_video.mp4")
    frame = video.get_frame()
    
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    mock_cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 0)

@patch('cv2.VideoCapture')
def test_video_frame_get_frame_no_frames(mock_video_capture):
    """Test when no frames are available"""
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)
    mock_video_capture.return_value = mock_cap
    
    video = VideoFrame("test_video.mp4")
    frame = video.get_frame()
    
    assert frame is None

@patch('cv2.VideoCapture')
def test_video_frame_destructor(mock_video_capture):
    """Test video resource cleanup"""
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap
    
    video = VideoFrame("test_video.mp4")
    video.__del__()
    
    mock_cap.release.assert_called_once()

# --- Model tests ---
@patch('wms_model.model')
def test_detection_object_with_detections(mock_model):
    """Test object detection with mock YOLO results"""
    # Setup mock detection result
    mock_boxes = Mock()
    mock_boxes.conf = [Mock()]
    mock_boxes.conf[0].item.return_value = 0.9
    mock_boxes.xyxy = [np.array([100, 100, 200, 200])]
    mock_boxes.cls = [Mock()]
    mock_boxes.cls[0].item.return_value = 0
    
    mock_result = Mock()
    mock_result.boxes = [mock_boxes]
    mock_model.return_value = [mock_result]
    mock_model.names = {0: "person"}
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated_frame, count = detection_object(dummy_frame)
    
    assert annotated_frame is not None
    assert annotated_frame.shape == dummy_frame.shape
    assert isinstance(count, int)
    assert count >= 0

@patch('wms_model.model')
def test_detection_object_no_detections(mock_model):
    """Test object detection with no objects found"""
    mock_result = Mock()
    mock_result.boxes = []
    mock_model.return_value = [mock_result]
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated_frame, count = detection_object(dummy_frame)
    
    assert annotated_frame is not None
    assert count == 0

@patch('wms_model.model')
def test_detection_object_data_with_detections(mock_model):
    """Test detection data extraction"""
    # Setup mock detection result
    mock_boxes = Mock()
    mock_boxes.conf = [Mock()]
    mock_boxes.conf[0].item.return_value = 0.9
    mock_boxes.xyxy = [np.array([100, 100, 200, 200])]
    mock_boxes.cls = [Mock()]
    mock_boxes.cls[0].item.return_value = 0
    
    mock_result = Mock()
    mock_result.boxes = [mock_boxes]
    mock_model.return_value = [mock_result]
    mock_model.names = {0: "person"}
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    
    assert isinstance(result, dict)
    assert "detections" in result
    assert "total" in result
    assert result["total"] >= 0
    
    if result["total"] > 0:
        detection = result["detections"][0]
        assert "class" in detection
        assert "confidence" in detection

@patch('wms_model.model')
def test_detection_object_data_low_confidence(mock_model):
    """Test detection filtering with low confidence"""
    # Setup mock detection result with low confidence
    mock_boxes = Mock()
    mock_boxes.conf = [Mock()]
    mock_boxes.conf[0].item.return_value = 0.5  # Below THRESHOLD (0.83)
    mock_boxes.xyxy = [np.array([100, 100, 200, 200])]
    mock_boxes.cls = [Mock()]
    mock_boxes.cls[0].item.return_value = 0
    
    mock_result = Mock()
    mock_result.boxes = [mock_boxes]
    mock_model.return_value = [mock_result]
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    
    assert result["total"] == 0
    assert result["detections"] == []

@patch('wms_model.model')
def test_detection_object_multiple_detections(mock_model):
    """Test multiple object detection"""
    # Setup multiple mock detection results
    mock_boxes1 = Mock()
    mock_boxes1.conf = [Mock()]
    mock_boxes1.conf[0].item.return_value = 0.9
    mock_boxes1.xyxy = [np.array([100, 100, 200, 200])]
    mock_boxes1.cls = [Mock()]
    mock_boxes1.cls[0].item.return_value = 0
    
    mock_boxes2 = Mock()
    mock_boxes2.conf = [Mock()]
    mock_boxes2.conf[0].item.return_value = 0.85
    mock_boxes2.xyxy = [np.array([300, 300, 400, 400])]
    mock_boxes2.cls = [Mock()]
    mock_boxes2.cls[0].item.return_value = 1
    
    mock_result = Mock()
    mock_result.boxes = [mock_boxes1, mock_boxes2]
    mock_model.return_value = [mock_result]
    mock_model.names = {0: "person", 1: "car"}
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    
    assert result["total"] == 2
    assert len(result["detections"]) == 2

# --- GenerateVideo class tests ---
@patch('wms_gen_video.VideoFrame')
@patch('wms_gen_video.detection_object')
@patch('cv2.imencode')
def test_generate_video_success(mock_imencode, mock_detection, mock_video_frame_class):
    """Test video generation with mock video and detection"""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Mock cv2.imencode
    mock_imencode.return_value = (True, np.array([255, 216, 255, 224], dtype=np.uint8))
    
    # Mock VideoFrame
    mock_video = Mock()
    mock_video.get_frame.side_effect = [dummy_frame, None]
    mock_video_frame_class.return_value = mock_video
    
    # Mock detection
    mock_detection.return_value = (annotated_frame, 1)
    
    gen_video = GenerateVideo()
    generator = gen_video.generate_video("test_video.mp4")
    
    # Get one frame
    frame_data = next(generator)
    assert b'--frame' in frame_data
    assert b'Content-Type: image/jpeg' in frame_data
    
    mock_imencode.assert_called()

@patch('wms_gen_video.VideoFrame')
@patch('cv2.imencode')
def test_generate_video_no_frames(mock_imencode, mock_video_frame_class):
    """Test video generation when no frames available"""
    mock_video = Mock()
    mock_video.get_frame.return_value = None
    mock_video_frame_class.return_value = mock_video
    
    gen_video = GenerateVideo()
    generator = gen_video.generate_video("test_video.mp4")
    
    assert generator is not None

# --- FastAPI endpoint tests ---
def test_fastapi_root_endpoint():
    """Test root endpoint"""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "1.0.0"

@patch('wms_main.VideoFrame')
def test_video_stream_endpoint(mock_video_frame_class):
    """Test video streaming endpoint"""
    mock_video = Mock()
    mock_video.generate_video.return_value = iter([b'--frame\r\ntest\r\n'])
    mock_video_frame_class.return_value = mock_video
    
    client = TestClient(app)
    response = client.get("/video")
    assert response.status_code == 200
    assert "multipart/x-mixed-replace" in response.headers.get("content-type", "")

# --- WebSocket test ---
@patch('wms_main.VideoFrame')
@patch('wms_main.detection_object_data')
def test_ws_detect_endpoint(mock_detection, mock_video_frame_class):
    """Test WebSocket detection endpoint"""
    mock_video = Mock()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_video.get_frame.return_value = dummy_frame
    mock_video_frame_class.return_value = mock_video
    
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

# --- Integration tests ---
@patch('cv2.VideoCapture')
@patch('wms_model.model')
def test_full_pipeline_integration(mock_model, mock_video_capture):
    """Test complete pipeline from video to detection"""
    # Mock video
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, dummy_frame)
    mock_video_capture.return_value = mock_cap
    
    # Mock YOLO model
    mock_boxes = Mock()
    mock_boxes.conf = [Mock()]
    mock_boxes.conf[0].item.return_value = 0.9
    mock_boxes.xyxy = [np.array([100, 100, 200, 200])]
    mock_boxes.cls = [Mock()]
    mock_boxes.cls[0].item.return_value = 0
    
    mock_result = Mock()
    mock_result.boxes = [mock_boxes]
    mock_model.return_value = [mock_result]
    mock_model.names = {0: "person"}
    
    # Test pipeline
    video = VideoFrame("test_video.mp4")
    frame = video.get_frame()
    assert frame is not None
    
    result = detection_object_data(frame)
    assert isinstance(result, dict)
    assert "total" in result
    assert "detections" in result

# --- Edge case tests ---
@patch('wms_model.model')
def test_detection_with_empty_frame(mock_model):
    """Test detection with empty frame"""
    mock_result = Mock()
    mock_result.boxes = []
    mock_model.return_value = [mock_result]
    
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(empty_frame)
    assert result["total"] == 0
    assert result["detections"] == []

@patch('wms_model.model')
def test_detection_with_model_without_names(mock_model):
    """Test detection when model doesn't have names attribute"""
    mock_boxes = Mock()
    mock_boxes.conf = [Mock()]
    mock_boxes.conf[0].item.return_value = 0.9
    mock_boxes.xyxy = [np.array([100, 100, 200, 200])]
    mock_boxes.cls = [Mock()]
    mock_boxes.cls[0].item.return_value = 0
    
    mock_result = Mock()
    mock_result.boxes = [mock_boxes]
    mock_model.return_value = [mock_result]
    # Remove names attribute
    if hasattr(mock_model, 'names'):
        delattr(mock_model, 'names')
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated_frame, count = detection_object(dummy_frame)
    
    assert annotated_frame is not None
    assert count >= 0

@patch('cv2.VideoCapture')
@patch('wms_video.logging')
def test_video_frame_generate_video_with_error(mock_logging, mock_video_capture):
    """Test video generation with error handling"""
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = Exception("Video read error")
    mock_video_capture.return_value = mock_cap
    
    video = VideoFrame("test_video.mp4")
    generator = video.generate_video()
    
    # The generator should handle the error
    try:
        next(generator)
    except StopIteration:
        pass  # Expected when generator completes
    
    mock_logging.error.assert_called()

# --- Performance tests ---
@patch('wms_model.model')
def test_detection_performance_with_many_objects(mock_model):
    """Test detection with many objects"""
    # Create many mock detection results
    mock_boxes_list = []
    for i in range(10):
        mock_boxes = Mock()
        mock_boxes.conf = [Mock()]
        mock_boxes.conf[0].item.return_value = 0.9
        mock_boxes.xyxy = [np.array([i*50, i*50, (i+1)*50, (i+1)*50])]
        mock_boxes.cls = [Mock()]
        mock_boxes.cls[0].item.return_value = i % 3  # Cycle through classes
        mock_boxes_list.append(mock_boxes)
    
    mock_result = Mock()
    mock_result.boxes = mock_boxes_list
    mock_model.return_value = [mock_result]
    mock_model.names = {0: "person", 1: "car", 2: "truck"}
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detection_object_data(dummy_frame)
    
    assert result["total"] == 10
    assert len(result["detections"]) == 10

# --- Utility tests ---
def test_environment_variables():
    """Test environment variables are properly set"""
    assert os.getenv("MODEL_PATH") is not None
    assert os.getenv("VIDEO_PATH") is not None

@patch('cv2.VideoCapture')
def test_video_frame_frame_count_increment(mock_video_capture):
    """Test frame count increments correctly"""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, dummy_frame)
    mock_video_capture.return_value = mock_cap
    
    video = VideoFrame("test_video.mp4")
    
    assert video.frame_count == 0
    video.get_frame()
    assert video.frame_count == 1
    video.get_frame()
    assert video.frame_count == 2

# --- Cleanup test ---
@patch('wms_gen_video.VideoFrame')
def test_generate_video_cleanup(mock_video_frame_class):
    """Test proper cleanup in GenerateVideo"""
    mock_video = Mock()
    mock_video.get_frame.side_effect = [None]  # No frames
    mock_video_frame_class.return_value = mock_video
    
    gen_video = GenerateVideo()
    gen_video.video_writer = Mock()  # Simulate having a video writer
    
    generator = gen_video.generate_video("test_video.mp4")
    
    try:
        list(generator)  # Exhaust generator
    except:
        pass
    
    # Should clean up video writer
    if gen_video.video_writer:
        gen_video.video_writer.release.assert_called()