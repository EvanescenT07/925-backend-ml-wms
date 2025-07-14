import cv2

# Replace this with your RTSP URL
rtsp_url = "rtsp://rtsp:admin@192.168.1.11:1935/h264_ulaw.sdp"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the camera opened successfully
if not cap.isOpened():
    print("❌ Failed to connect to the RTSP stream.")
    exit()

print("✅ Connected to RTSP stream. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    

    if not ret:
        print("⚠️ Failed to retrieve frame. Retrying...")
        continue

    # Display the resulting frame
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    cv2.imshow("RTSP Stream", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
