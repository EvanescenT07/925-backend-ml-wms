from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from wms_model import detection_object_data
from wms_video import VideoFrame
from wms_logging import LoggingConfig
import cv2
import os
import logging
import asyncio

# APP Metadata
title = "Warehouse Management System"
description = "PT. Akebono Brake Astra Indonesia Warehouse Management System"
version = "1.0.0"

video = os.getenv("VIDEO_PATH")

app = FastAPI(
    title=title,
    description=description,
    version=version,
    debug=True,
)

LoggingConfig()

# API root endpoint
@app.get("/")
async def root():
    logging.info("Root endpoint accessed")
    return {
        "message": "Welcome to the Warehouse Management System",
        "version": version,
        "description": description
    }
    
# Video feed endpoint
@app.get("/video")
async def video_feed():
    logging.info("Video feed endpoint accessed")
    video_stream = VideoFrame(video)
    
    return StreamingResponse(
        video_stream.generate_video(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
    
# WebSocket endpoint for real-time object detection
@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    video_stream = VideoFrame(video)
    try:
        while True:
            frame = video_stream.get_frame()
            if frame is None:
                await asyncio.sleep(1)
                continue
            data = detection_object_data(frame)
            await websocket.send_json(data)
            await asyncio.sleep(2)
    finally:
        del video_stream
        await websocket.close()