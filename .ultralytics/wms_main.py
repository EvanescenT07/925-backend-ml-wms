from fastapi import FastAPI, WebSocket, WebSocketException
from fastapi.responses import StreamingResponse
from wms_model import detection_object_data
from wms_gen_video import GenerateVideo
from wms_camera import Camera 
from wms_logging import LoggingConfig
import cv2
import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketException
from fastapi.responses import StreamingResponse

# App metadata
title = "Warehouse Management System"
description = "PT. Akebono Brake Astra Indonesia Warehouse Management System"
version = "1.0.1"

# Initialize FastAPI app
app = FastAPI(
    title=title,
    description=description,
    version=version,
)

LoggingConfig()

camera_buffer = Camera()

# API root endpoint
@app.get("/")
def root():
    return {
        "message": "Welcome to the Warehouse Management System API",
        "version": version,
        "description": description,
    }

# API endpoint to stream video frames
@app.get("/video")
async def video_stream():
    stream = GenerateVideo(camera=camera_buffer)
    return StreamingResponse(
        stream.generate_video(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

# WebSocket endpoint for real-time object detection
@app.websocket("/ws/detect")
async def websocket_detection(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            frame = camera_buffer.get_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))
            data = detection_object_data(frame=frame)
            await websocket.send_json(data=data)
            await asyncio.sleep(0.05)  # Adjust the sleep time as needed
    except WebSocketException:
        logging.info("WebSocket client disconnected")
    finally:
        await websocket.close()
        logging.info("WebSocket connection closed")

# Run the application with: uvicorn wms_main:app --host 0.0.0.0 --port 8000
# Note: Ensure that the necessary imports and configurations are in place for the Camera and detection_object_data functions.