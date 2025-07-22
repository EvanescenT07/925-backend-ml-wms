from ultralytics import YOLO
from dotenv import load_dotenv
import cv2
import os

load_dotenv()

THRESHOLD = 0.83
model_path = os.getenv("MODEL_PATH")
if model_path is None:
    raise ValueError("Environment variable MODEL_PATH is not set.")
model = YOLO(model_path)

def detection_object(frame):
    frame = cv2.flip(frame, 1)
    result = model(frame)[0]
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
            
    h, _ = annotated_frame.shape[:2]
    text = f"Total: {count}"
    font_scale = 1.5
    thickness = 2
    margin = 10
    x = margin
    y = h - margin
    cv2.putText(
        annotated_frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
    )
    return annotated_frame, count

def detection_object_data(frame):
    frame = cv2.flip(frame, 1)
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
            count += 15
    return {
        "detections": detections,
        "total": count
    }