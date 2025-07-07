from ultralytics import YOLO

model = YOLO("backend/model/testing/best.pt")
model.export(format="onnx")