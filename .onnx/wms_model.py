from dotenv import load_dotenv
import onnxruntime as ort
import multiprocessing as mp
import numpy as np
import cv2
import os
import yaml
import time
import logging

load_dotenv()

THRESHOLD = 0.83
NMS_IOU_THRESHOLD = 0.5
MODEL_PATH = os.getenv("MODEL_PATH")
if MODEL_PATH is None:
    raise ValueError("Environment variable 'MODEL_PATH' is not set.")


# Load Class Names
MODEL_CLASSNAME_PATH = os.getenv("CLASS_NAMES_PATH")
if MODEL_CLASSNAME_PATH is None:
    raise ValueError("Environment variable 'CLASS_NAMES_PATH' is not set.")

with open(MODEL_CLASSNAME_PATH, 'r') as f:
    data = yaml.safe_load(f)
    if isinstance(data.get("class_names"), dict):
        CLASS_NAMES = [data["class_names"][i] for i in sorted(data["class_names"].keys(), key=int)]
        logging.info(f"Loaded class names: {CLASS_NAMES}")
    else:
        CLASS_NAMES = []
        logging.error("Invalid class names format.")
        
# Optimize ONNX session for performance
def get_optimal_execution_providers():
    providers = ort.get_available_providers()
    execution_providers = []
    
    # Check for available execution providers
    if "TensorrtExecutionProvider" in providers: # TensorRT for NVIDIA GPUs
        execution_providers.append("TensorrtExecutionProvider")
        logging.info("✅ Using TensorRT for maximum NVIDIA performance")
    
    if "CUDAExecutionProvider" in providers: # CUDA for NVIDIA GPUs
        execution_providers.append("CUDAExecutionProvider")
        logging.info("✅ Using CUDA for NVIDIA GPU acceleration")
    
    elif "DmlExecutionProvider" in providers: # DirectML for AMD and Intel GPUs
        execution_providers.append("DmlExecutionProvider")
        logging.info("✅ Using DirectML for GPU acceleration")
        
    elif "ROCMExecutionProvider" in providers: # ROCm for AMD GPUs
        execution_providers.append("ROCMExecutionProvider")
        logging.info("✅ Using ROCm for AMD GPU acceleration")
 
    execution_providers.append("CPUExecutionProvider") 
    logging.info("Using CPUExecutionProvider for CPU inference.")
    
    return execution_providers

def get_session_option():
    # Create session with optimized providers
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    
    cpu_count = mp.cpu_count()
    session_options.intra_op_num_threads = min(8, cpu_count)
    session_options.inter_op_num_threads = min(4, cpu_count // 2)
    
    logging.info(f"Using {cpu_count} CPU threads for inference.")
    
    return session_options

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=get_session_option(),
    providers=get_optimal_execution_providers()
)

# Check if the providers are active
active_providers = session.get_providers()
if not active_providers:
    logging.error("No active execution providers found. Please check your ONNX Runtime installation and model compatibility.")
logging.info(f"Active execution providers: {active_providers}")

# Pre-allocate input tensors for performance
input_name = session.get_inputs()[0].name

# Optimized preprocessing function
def preprocess(frame):
    if frame.shape[:2] != (640, 640):
        img = cv2.resize(frame, (640, 640))
    else:
        img = frame
        
# Convert and normalize in one step
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    return img

# Function to calculate Intersection over Union (IoU)
def iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (box1_area + box2_area - inter_area)

# Non-Maximum Suppression (NMS) function
def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current_idx = indices[0]
        keep.append(current_idx)
        indices = indices[1:]
        if len(indices) == 0:
            break
    
        # Vectorized IoU calculation
        current_box = boxes[current_idx]
        remaining_boxes = boxes[indices]
        
        # Calculate IoU for all remaining boxes
        ious = np.array([
            iou(current_box, box) for box in remaining_boxes
        ])
        indices = indices[ious < iou_threshold]
        
    return keep 

def postprocess(outputs, orig_shape):
    try:
        # Validate model output
        if not outputs or len(outputs) == 0:
            logging.warning("Model returned empty outputs")
            return []
        
        output = outputs[0]
        if output is None or output.size == 0:
            logging.warning("Model output is empty")
            return []
            
        output = output.squeeze().T  # Shape: (8400, 5) or (8400, 85)
        
        # FIXED: Handle different model output formats
        if len(output.shape) != 2:
            logging.warning(f"Invalid output dimensions: {output.shape}")
            return []
            
        h0, w0 = orig_shape[:2]
        
        # Check if this is a single-class model (5 values) or multi-class (85+ values)
        if output.shape[1] == 5:
            # Single-class model: [x, y, w, h, confidence]
            logging.debug("Processing single-class model output")
            
            # Vectorized confidence filtering
            confidences = output[:, 4]
            high_conf_indices = confidences >= THRESHOLD

            if not np.any(high_conf_indices):
                return []
            
            # Filter detections based on confidence
            filtered_output = output[high_conf_indices]
            boxes, scores, class_ids = [], [], []
            
            for det in filtered_output:
                x, y, w, h, conf = det[:5]
                
                # For single-class model, class_id is always 0
                class_id = 0
                
                # Convert to xyxy format and scale to original image size
                x1 = int((x - w / 2) * w0 / 640)
                y1 = int((y - h / 2) * h0 / 640)
                x2 = int((x + w / 2) * w0 / 640)
                y2 = int((y + h / 2) * h0 / 640)

                # Clamp coordinate to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w0, x2), min(h0, y2)
                
                # Validate bounding box
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(conf)
                    class_ids.append(class_id)
                    
        elif output.shape[1] >= 85:
            # Multi-class model: [x, y, w, h, confidence, class1, class2, ...]
            logging.debug("Processing multi-class model output")
            
            # Vectorized confidence filtering
            confidences = output[:, 4]
            high_conf_indices = confidences >= THRESHOLD

            if not np.any(high_conf_indices):
                return []
            
            # Filter detections based on confidence
            filtered_output = output[high_conf_indices]
            boxes, scores, class_ids = [], [], []
            
            for det in filtered_output:
                if len(det) < 6:  # Validate detection format
                    continue
                    
                x, y, w, h = det[:4]
                conf = det[4]
                class_scores = det[5:]
                
                # Validate class_scores before argmax
                if len(class_scores) == 0:
                    logging.warning("Empty class scores detected")
                    continue
                    
                class_id = int(np.argmax(class_scores))
                
                # Convert to xyxy format and scale to original image size
                x1 = int((x - w / 2) * w0 / 640)
                y1 = int((y - h / 2) * h0 / 640)
                x2 = int((x + w / 2) * w0 / 640)
                y2 = int((y + h / 2) * h0 / 640)

                # Clamp coordinate to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w0, x2), min(h0, y2)
                
                # Validate bounding box
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(conf)
                    class_ids.append(class_id)
        else:
            logging.warning(f"Unsupported model output shape: {output.shape}")
            return []
        
        if len(boxes) == 0:
            return []
        
        # Apply Non-Maximum Suppression
        keep = nms(boxes, scores=scores, iou_threshold=NMS_IOU_THRESHOLD)
        
        return [
            (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i], class_ids[i]) for i in keep
        ]
        
    except Exception as e:
        logging.error(f"Error in postprocess: {e}")
        return []

def detection_object_data(frame):
    try:
        # Validate frame input
        if frame is None:
            logging.warning("Received None frame")
            return {"total": 0, "detections": []}
        
        if frame.size == 0:
            logging.warning("Received empty frame")
            return {"total": 0, "detections": []}
            
        # Validate frame shape
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            logging.warning(f"Invalid frame shape: {frame.shape}")
            return {"total": 0, "detections": []}
        
        start_time = time.time()
        
        # preprocess the frame
        img = preprocess(frame)
        
        # Validate preprocessed input
        if img is None or img.size == 0:
            logging.warning("Preprocessing failed")
            return {"total": 0, "detections": []}

        # run inference
        outputs = session.run(None, {input_name: img})
        
        # postprocess the outputs
        detections = postprocess(outputs=outputs, orig_shape=frame.shape)
        
        # Prepare the result
        result = []
        for x1, y1, x2, y2, conf, class_id in detections:
            # FIXED: Handle single-class model
            if class_id >= len(CLASS_NAMES):
                logging.warning(f"Invalid class_id: {class_id}, using first class")
                class_id = 0
                
            class_name = CLASS_NAMES[class_id] if len(CLASS_NAMES) > 0 else "object"
            result.append({
                "class_id": int(class_id),
                "class_name": class_name,
                "confidence": float(conf),
                "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
            })
            
        inference_time = time.time() - start_time
        
        # Log performance periodically
        if len(result) > 0:
            logging.debug(f"Inference time: {inference_time:.4f}s, detections: {len(result)}")
                        
        return {
            "total": len(result),
            "detections": result
        }
        
    except Exception as e:
        logging.error(f"Error in detection_object_data: {e}")
        return {"total": 0, "detections": []}

def detection_object(frame):
    try:
        if frame is None or frame.size == 0:
            logging.warning("Received invalid frame for detection_object")
            return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8), 0
        
        detections = postprocess(
            session.run(None, {input_name: preprocess(frame)}), orig_shape=frame.shape
        )
        annotated_frame = frame.copy()
        
        # Draw bounding boxes and labels on the frame
        for x1, y1, x2, y2, conf, class_id in detections:
            # FIXED: Handle single-class model
            if class_id >= len(CLASS_NAMES):
                class_id = 0
                
            class_name = CLASS_NAMES[class_id] if len(CLASS_NAMES) > 0 else "object"
            label = f"{class_name} {conf:.2f}"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Total number of detections
        cv2.putText(annotated_frame, f"Total: {len(detections)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame, len(detections)
        
    except Exception as e:
        logging.error(f"Error in detection_object: {e}")
        return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8), 0