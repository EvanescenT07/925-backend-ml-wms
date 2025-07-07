import argparse
import cv2
from ultralytics import YOLO

THRESHOLD = 0.85

def detection(video_path, model_path, conf_threshold=THRESHOLD):
    model = YOLO(model_path)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_width = 640
    frame_height = 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # label = f"{model.names[int(box.cls)]} {conf:.2f}"
                label = f"{conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        print(f"total box : {count}")
        
        cv2.putText(frame, f"total box : {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        out.write(frame)
        cv2.imshow("Model Testing - Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    out.release()
    cv2.destroyAllWindows()
    print("Result video saved to test/result_video.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOv8 model on a video.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("model_path", type=str, help="Path to the YOLOv8 model file.")
    args = parser.parse_args()
    detection(args.video_path, args.model_path, THRESHOLD)