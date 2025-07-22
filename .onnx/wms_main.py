from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from wms_camera import Camera
from wms_model import detection_object_data  # Update to the correct symbol name, or fix as needed
from wms_gen_video import GenerateVideo
from wms_logging import LoggingConfig
from contextlib import asynccontextmanager
import cv2
import asyncio
import json
import time
import logging

LoggingConfig()

# App metadata
title = "WMS Backend ML System"
description = "Backend system for Warehouse Management System (WMS) using Machine Learning with ONNX runtime."
version = "1.0.1"

# Global resources
resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting WMS Backend ML System...")
    try:
        # Initialize camera instance
        resources["camera"] = Camera()
        await asyncio.sleep(0.5)  # Simulate async startup delay

        # verify camera connection
        test_frame = resources["camera"].get_frame()
        if test_frame is not None:
            logging.info("Camera initialized successfully.")
        else:
            logging.warning("Camera initialization failed, no frame received.")
        logging.info("WMS Backend ML System started successfully.")
    
    except Exception as e:
        logging.error(f"Error during startup: {e}")
        resources["camera"] = None
        
    yield
    
    # Cleanup resources
    logging.info("Shutting down WMS Backend ML System...")
    if "camera" in resources and resources["camera"]:
        resources["camera"].stop()
        logging.info("Camera stopped.")
    resources.clear()
    
# Function to get camera instance
def get_camera():
    camera = resources.get("camera")
    if camera is None:
        logging.error("Camera instance is not available.")
        raise HTTPException(status_code=503, detail="Camera service unavailable")
    return camera


app = FastAPI(
    title=title,
    description=description,
    version=version,
    lifespan=lifespan
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the WMS Backend ML System",
        "version": version,
        "description": description,
        "endpoints": {
            "check": "/",
            "video": "/video",
            "websocket_detection": "/ws/detect",
            "health status": "/health"
        }
    }

# Video streaming endpoint
@app.get("/video")
async def video_stream():
    try:
        camera = get_camera()
        video_generator = GenerateVideo(camera)
        
        return StreamingResponse(
            video_generator.generate_video(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except HTTPException as http_exc:
        logging.error(f"HTTP error: {http_exc.detail}")
        return {"error": "HTTP error occurred"}
    except Exception as e:
        logging.error(f"Error in video streaming: {e}")
        return {"error": "Failed to stream video"}
    finally:
        logging.info("Video streaming stopped.")

# WebSocket endpoint for real-time object detection
@app.websocket("/ws/detect")
async def websocket_detection(websocket: WebSocket):
    await websocket.accept()
    
    try:
        camera = get_camera()
    except HTTPException as http_exc:
        await websocket.close(code=http_exc.status_code, reason=http_exc.detail)
        return
    
    try:
        last_detection_time = 0
        target_fps = 24
        detection_interval = 1.0 / target_fps
        
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "WebSocket connection established for object detection",
        })
        
        while True:
            current_time = time.time()
            
            #  Maintain target FPS
            if current_time - last_detection_time < detection_interval:
                await asyncio.sleep(0.005)
                continue
            
            # Get frame from shared camera instance
            frame = camera.get_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            
            # Optimize frame size for processing
            if frame.shape[0] > 640:
                frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)
            
            # Perform object detection
            data= detection_object_data(frame=frame)
            
            # Send detection data over WebSocket
            await websocket.send_json({
                "type": "detection",
                "data": data,
                "timestamp": current_time
            })
            
            last_detection_time = current_time
            
    except WebSocketDisconnect:
        logging.info("WebSocket connection closed by client.")
        return {"status": "disconnected"}
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        await websocket.close(code=1000, reason="Internal Server Error")
        return {"error": "WebSocket connection error"}
    finally:
        try:
            await websocket.close()
            logging.info("WebSocket connection closed.")
        except Exception as e:
            pass
            logging.error(f"Error closing WebSocket: {e}")
        
    
# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check with FPS monitoring"""
    try:
        camera = get_camera()
        
        # Test frame retrieval
        start_time = time.time()
        frame = camera.get_frame()
        frame_time = time.time() - start_time
        
        return {
            "status": "healthy" if frame is not None else "degraded",
            "camera_connected": frame is not None,
            "frame_retrieval_time": f"{frame_time*1000:.1f}ms",
            "estimated_max_fps": f"{1/frame_time:.1f}" if frame_time > 0 else "unknown",
            "frame_shape": frame.shape if frame is not None else None,
            "timestamp": time.time()
        }
    except:
        return {
            "status": "unhealthy",
            "camera_connected": False,
            "timestamp": time.time()
        }
        
# Run the FastAPI app
# uvicorn main:app --host 0.0.0.0 --port 8000