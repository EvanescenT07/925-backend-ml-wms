import argparse
import cv2
from ultralytics import YOLO

THRESHOLD = 0.85
VIDEO_PATH = "video/warehouse.mp4"
MODEL_PATH = "model/model-WMS.pt"


def detection(video_path, model_path, conf_threshold=THRESHOLD):
    model = YOLO(model_path)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_width = 640
    frame_height = 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    out = cv2.VideoWriter(
        'test/result_video.mp4',
        fourcc,
        int(capture.get(cv2.CAP_PROP_FPS)),
        (frame_width, frame_height)
    )
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (frame_width, frame_height))
        result = model(frame)[0]
        count = 0
        for box in result.boxes:
            conf = float(box.conf)
            if conf >= conf_threshold:
                count += 15
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # label = f"{model.names[int(box.cls)]} {conf:.2f}"
                label = f"{conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 155, 0), 1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 155, 0), 1)
        
        font_scale = 0.6  # Smaller text
        thickness = 2
        margin = 10
        text = f"total box : {count}"
        x = margin
        y = frame_height - margin

        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        out.write(frame)
        cv2.imshow("Model Testing - Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    out.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    detection(VIDEO_PATH, MODEL_PATH, THRESHOLD)