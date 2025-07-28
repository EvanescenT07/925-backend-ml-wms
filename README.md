# Warehouse Management System (WMS) - Video Object Detection Backend Service

## 🚀 Project Overview

This project is a **Warehouse Management System (WMS)** for Capstone Project that leverages **YOLO-based object detection** on video feeds. It features:

- Real-time object detection from warehouse video or camera streams
- FastAPI backend with REST and WebSocket endpoints
- Live video streaming and detection results to a web frontend
- Modular structure for ONNX, Ultralytics, and custom video feed backends
- Comprehensive unit tests for robust development

---

## 🛠️ Getting Started

### 1. **Clone the Repository**

```bash
git clone https://github.com/EvanescenT07/925-backend-ml-wms.git
cd warehouse-management-system/backend-ml-system
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. **Install Dependencies**

Choose your backend implementation and install requirements:

#### Ultralytics

```bash
cd .ultralytics
pip install --upgrade pip
pip install -r requirements.txt
```

#### ONNX Runtime

```bash
cd .onnx
pip install --upgrade pip
pip install -r requirements.txt
```

#### Custom Video Feed

```bash
cd .video_feed
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. **Prepare Environment Variables**

Rename the example `.env.example` into `.env` file and adjust as needed

### 4. **Start the FastAPI Server**

```bash
# From the desired backend folder
uvicorn wms_main:app --reload
```

- The API endpoint will be available at: **http://localhost:8000/**
- The Video Stream endpoint: **http://localhost:8000/video**
- WebSocket detection endpoint: **ws://localhost:8000/ws/detect**

### 📝 Notes

- **Virtual Environment**: Always activate your virtual environment before installing dependencies or running the app.
- **Logging**: Logs are saved to `wms.log` and can also be configured to show in the terminal.
- **Configuration**: Change model/video paths and other settings in the `.env` file.
- **Testing**: Run `pytest test_wms.py -v` in your backend folder to execute all unit tests.
- **Docker Support**: See the provided Dockerfiles for containerized deployment.

### ✨ Highlights

- **Modular**: Supports ONNX, Ultralytics, and custom video feed backends.
- **Real-Time**: Live video and detection streaming to the web frontend.
- **Tested**: Comprehensive unit tests for reliability and maintainability.

### 📂 Project Structure

```
backend-ml-system/
│
├── .onnx/                # ONNX runtime backend
│   ├── wms_main.py
│   ├── wms_model.py
│   ├── wms_camera.py
│   ├── wms_gen_video.py
│   ├── wms_logging.py
│   ├── requirements.txt
│   ├── dockerfile
│   ├── docker-compose.yml
│   ├── .env
│   ├── .env.example
│   ├── model/            # Place ONNX model here
│   └── ...
│
├── .ultralytics/         # Ultralytics YOLO backend
│   ├── wms_main.py
│   ├── wms_model.py
│   ├── ...
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── .env
│   ├── .env.example
│   └── model/            # Place YOLOv11 model here
│
├── .video_feed/          # Custom video feed backend
│   ├── wms_main.py
│   ├── wms_model.py
│   ├── wms_video.py
│   ├── wms_gen_video.py
│   ├── wms_logging.py
│   ├── requirements.txt
│   ├── .env
│   ├── .env.example
│   └── ...
│
├── frontend/             # Simple Test HTML frontend
│   └── main.html
│
├── .model-train/         # Model training scripts and notebooks
│   ├── 925_yolov11_train.ipynb
│   ├── yolo11n.pt
│   ├── dataset/
│   ├── runs/
│   └── ...
│
├── test-script/          # Test scripts for model functionality
│   └── ...
│
├── utils/                # Helper scripts/utilities
│   └── ...
│
├── README.md
├── .gitignore
└── ...
```

---

### 🤝 Credits

Developed for Capstone Project, President University.

`Still on DEVELOPMENT Stage`
