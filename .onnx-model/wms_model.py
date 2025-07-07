import onnxruntime as ort
import cv2
import numpy as np
import os
import yaml
from dotenv import load_dotenv
import logging

load_dotenv()

THRESHOLD = 0.83
NMS_IOU_THRESHOLD = 0.5
MODEL_PATH = os.getenv("MODEL_PATH")

# Load class names from YAML file
with open("class.yaml", "r") as f:
    data = yaml.safe_load(f)
    # Support both list and dict format
    if isinstance(data.get("names"), dict):
        CLASS_NAMES = [data["names"][k] for k in sorted(data["names"].keys(), key=int)]
    else:
        CLASS_NAMES = data.get("names", [])

# Check available providers
providers = ort.get_available_providers()
if "CUDAExecutionProvider" in providers:
    exec_providers = ['CUDAExecutionProvider']
    logging.info("Using CUDAExecutionProvider for ONNX Runtime.")
else:
    exec_providers = ['CPUExecutionProvider']
    logging.info("Using CPUExecutionProvider for ONNX Runtime.")
session = ort.InferenceSession(MODEL_PATH, providers=exec_providers)

# Preprocess function to prepare the input frame for the model
def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# Function to calculate Intersection over Union (IoU) between two bounding boxes
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    logging.debug(f"IoU: {iou} for boxes {box1} and {box2}")
    return iou

# Function to perform Non-Maximum Suppression (NMS) on the detected boxes, This function filters out overlapping boxes based on IoU threshold
def nms(boxes, scores, iou_threshold):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        idxs = idxs[1:]
        idxs = [i for i in idxs if iou(boxes[current], boxes[i]) < iou_threshold]
    return keep

# Postprocess function to extract bounding boxes, scores, and class IDs from the model outputs
def postprocess(outputs, orig_shape):
    boxes, scores, class_ids = [], [], []
    output = outputs[0]  # (1, 5, 8400)
    output = np.squeeze(output)  # (5, 8400)
    output = output.T  # (8400, 5)
    h0, w0 = orig_shape[:2]
    
    for det in output:
        conf = det[4]
        if conf >= THRESHOLD:
            x, y, w, h = det[0:4]
            class_ids = int(det[5]) if len(det) > 5 else 0
            # Convert to xyxy and scale to original image
            x1 = int((x - w / 2) * w0 / 640)
            y1 = int((y - h / 2) * h0 / 640)
            x2 = int((x + w / 2) * w0 / 640)
            y2 = int((y + h / 2) * h0 / 640)
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            class_ids.append(class_ids)
    # NMS
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    keep = nms(boxes, scores, NMS_IOU_THRESHOLD)
    result = []
    for i in keep:
        result.append((*boxes[i], scores[i], class_ids[i]))
    return result

# Function to detect objects in a given frame using the ONNX model
def detection_object(frame):
    orig_shape = frame.shape
    img = preprocess(frame)
    outputs = session.run(None, {session.get_inputs()[0].name: img})
    detection = postprocess(outputs, orig_shape)
    annotated_frame = frame.copy()
    count = 0
    for x1, y1, x2, y2, conf, class_id in detection:
        label = f"{CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else class_id} {conf:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        count += 1
        cv2.putText(annotated_frame, f"Total: {count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        logging.debug(f"Detected {label} at ({x1}, {y1}, {x2}, {y2}) with confidence {conf:.2f}")
    return annotated_frame, count

# Function to detect objects in a frame and return the data in a structured format
def detection_object_data(frame):
    orig_shape = frame.shape
    img = preprocess(frame)
    ort_inputs = {
        session.get_inputs()[0].name: img
    }
    ort_outs = session.run(None, ort_inputs)
    detection = postprocess(ort_outs, orig_shape)
    result = []
    count = 0
    for x1, y1, x2, y2, conf, class_id in detection:
        label = f"{CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)} {conf:.2f}"
        result.append({
            "class_id": int(class_id),
            "class": label,
            "confidence": float(conf),
            "box": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            },
        })
        count += 1
        logging.debug(f"Detected {label} at ({x1}, {y1}, {x2}, {y2}) with confidence {conf:.2f}")
    # Always return a result, even if empty
    return {
        "total": count,
        "detection": result
    }