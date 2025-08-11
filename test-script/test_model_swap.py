# Simple webcam brightness check and dummy detection function

import cv2

def is_dark(frame, threshold=50):
    # Simple brightness check (mean pixel value)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    print(f"Frame brightness: {brightness:.2f}")
    return brightness < threshold

def detection_object_data(frame):
    # Dummy function to show which model would be used
    if is_dark(frame):
        print("Dark condition detected: Would use IR model for detection.")
    else:
        print("Normal lighting detected: Would use normal model for detection.")

def main():
    cap = cv2.VideoCapture(0)  # Open default webcam
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        detection_object_data(frame)

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()