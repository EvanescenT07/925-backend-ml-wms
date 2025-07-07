import cv2

# Load image
image = cv2.imread('plating_store_kamera2-26/test/images/image_048_jpg.rf.a510c692a1c581fb42353654e93b68c4.jpg')
image = cv2.resize(image, (640, 480))

# Fungsi untuk mencetak koordinat saat diklik
def draw_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        # Gambar titik (opsional)
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", image)

cv2.imshow("Image", image)
cv2.setMouseCallback("Image", draw_point)
cv2.waitKey(0)
cv2.destroyAllWindows()
