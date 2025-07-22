from wms_model import detection_object
import cv2
import time
import logging

class GenerateVideo:
    def __init__(self, camera):
        self.cam = camera
        
    def generate_video(self):
        try:
            frame_count = 0
            start_time = time.time()
            
            while True:
                frame = self.cam.get_frame()
                if frame is None:
                    # Handle the case where no frame is returned
                    logging.warning("No frame received from camera.")
                    time.sleep(0.001)
                    continue
                
                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Perform object detection
                result, _ = detection_object(frame=frame)
                
                # Optimized JPEG encoding for faster processing
                encode_params = [
                    cv2.IMWRITE_JPEG_QUALITY, 80,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1
                ]
                
                success, jpeg = cv2.imencode('.jpg', result, encode_params)
                if not success:
                    logging.error("Failed to encode frame to JPEG.")
                    continue
                
                frame_bytes = jpeg.tobytes()
                
                # Yield the frame as a multipart response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # FPS Monitoring (Optional)
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    logging.debug(f"FPS: {fps:.2f}")
                
        except Exception as e:
            logging.error(f"Error in video generation: {e}")
        finally:
            logging.info("Video generation stopped.")