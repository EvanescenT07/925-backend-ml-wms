from ultralytics import YOLO
from dotenv import load_dotenv
import cv2
import os
import logging

load_dotenv()

# Set up Threshold for detection confidence and load the YOLO model
THRESHOLD = 0.83
model = YOLO(os.getenv("MODEL_PATH"))

if not model:
    logging.error("Model not loaded. Please check the MODEL_PATH environment variable.")

# Function to perform object detection on a given frame
def detection_object(frame):
    try:
        result = model(frame)[0]
    except Exception as e:
        logging.exception("Model inference failed during object detection.")
        return None
    annotated_frame = frame.copy()
    count = 0
    
    for box in result.boxes:
        conf = box.conf[0].item()
        if conf >= THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id] if hasattr(model, "names") else str(class_id)
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            count += 1
            
    cv2.putText(annotated_frame, f"Total: {count}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    logging.debug(f"Detected {count} objects with confidence >= {THRESHOLD}")
    return annotated_frame, count

# Function to perform object detection and return structured data
def detection_object_data(frame):
    result = model(frame)[0]
    detections = []
    count = 0
    
    for box in result.boxes:
        conf = box.conf[0].item()
        if conf >= THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id] if hasattr(model, "names") else str(class_id)
            detections.append({
                "class": class_name,
                "confidence": conf,
            })
            count += 1
            logging.debug(f"Detected {class_name} at ({x1}, {y1}, {x2}, {y2}) with confidence {conf:.2f}")
    return {
        "total": count,
        "detections": detections
    }