import cv2

FILE_PATH = "images/images-test.jpg"

# Load image
image = cv2.imread(FILE_PATH)
if image is None:
    raise FileNotFoundError("Image not found or unable to load.")
image = cv2.resize(image, (640, 480))

# Fungsi untuk mencetak koordinat saat diklik
def draw_point(event, x, y, flags, param):
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        # Gambar titik (opsional)
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", img)

cv2.imshow("Image", image)
cv2.setMouseCallback("Image", draw_point, image)
cv2.waitKey(0)
cv2.destroyAllWindows()
