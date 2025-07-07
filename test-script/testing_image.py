import cv2
import numpy as np
from ultralytics import YOLO

AREA_A = np.array([
    [184, 170],   # kiri atas
    [307, 170],  # kanan atas
    [279, 442],  # kanan bawah
    [132, 432]    # kiri bawah
])

AREA_B = np.array([
    [339, 181],   # kiri atas
    [562, 191],   # kanan atas
    [593, 409],   # kanan bawah
    [344, 445]    # kiri bawah
])

def point_in_poly(point, poly):
    return cv2.pointPolygonTest(poly, point, False) >= 0

def detection(image_path, model_path, conf_threshold=0.85):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    frame = cv2.resize(image, (640, 480))
    result = model(frame)[0]

    count_a, count_b = 0, 0
    counted = set()

    for box in result.boxes:
        conf = float(box.conf)
        if conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            label = f"{conf:.2f}"

            # Cek area
            in_a = point_in_poly((cx, cy), AREA_A)
            in_b = point_in_poly((cx, cy), AREA_B)

            if in_a and (cx, cy) not in counted:
                count_a += 1
                counted.add((cx, cy))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif in_b and (cx, cy) not in counted:
                count_b += 1
                counted.add((cx, cy))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)

            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Gambar area A dan B
    cv2.polylines(frame, [AREA_A], True, (0, 255, 0), 2)
    cv2.polylines(frame, [AREA_B], True, (255, 0, 0), 2)

    cv2.putText(frame, f"Area A: {count_a}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Area B: {count_b}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Total: {count_a + count_b}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Detection Area", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detection("plating_store_kamera2-26/test/images/image_048_jpg.rf.a510c692a1c581fb42353654e93b68c4.jpg", "backend/model/production/best.pt")
# Contoh pemanggilan
# detection("test.jpg", "model.pt")