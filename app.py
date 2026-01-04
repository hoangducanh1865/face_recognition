import os
import cv2
import asyncio
import httpx
import threading
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from datetime import datetime
from typing import Optional
import shutil

# Import existing modules
from src.config import Config
from src.model import SiameseModel
from src.verifier import FaceVerifier
from src.utils import setup_gpu

# Configuration
BACKEND_URL = "http://localhost:8080/api/face-access"
ROOM_NAME = os.getenv("ROOM_NAME", "Main Entrance")
CAMERA_INDEX = 0

app = FastAPI(title="Face Recognition Access Control")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
camera = None
verifier = None
latest_frame = None
camera_lock = threading.Lock()
known_faces = {} # userId -> name

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        global latest_frame
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with camera_lock:
                    latest_frame = frame
            else:
                # Reconnect logic could go here
                pass

    def stop(self):
        self.running = False
        self.cap.release()

@app.on_event("startup")
async def startup_event():
    global camera, verifier
    setup_gpu()
    
    # Initialize Model and Verifier
    # Note: We assume the model is already trained and saved at the path specified in Config
    config = Config()
    config.create_directories()
    model = SiameseModel()
    # Load weights (assuming they exist, otherwise this might fail)
    # Use the model path from config, which points to models/siamesemodelv2.keras
    model_path = config.paths.model_path
    if os.path.exists(model_path):
        model.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: No model found at {model_path}. Verification will fail.")

    verifier = FaceVerifier(model, config)
    
    # Start Camera
    camera = Camera()
    
    # Start Sync Task
    asyncio.create_task(sync_faces_loop())

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.on_event("shutdown")
def shutdown_event():
    if camera:
        camera.stop()

async def sync_faces_loop():
    """Periodically sync faces from backend"""
    while True:
        try:
            print("Syncing faces from backend...")
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{BACKEND_URL}/users")
                if response.status_code == 200:
                    users = response.json()
                    config = Config()
                    save_dir = config.paths.verification_images_path
                    
                    current_files = set(os.listdir(save_dir))
                    
                    for user in users:
                        user_id = user["userId"]
                        image_url = user["imageUrl"]
                        # Assuming image_url is relative to backend: /uploads/faces/filename.jpg
                        # We need to construct full URL: http://localhost:8080/uploads/faces/filename.jpg
                        # But wait, the backend returns what we saved.
                        
                        # For simplicity, we'll use the filename from the URL as the local filename
                        filename = os.path.basename(image_url)
                        if filename not in current_files:
                            # Download
                            full_url = f"http://localhost:8080{image_url}"
                            print(f"Downloading face for {user['name']} from {full_url}")
                            img_resp = await client.get(full_url)
                            if img_resp.status_code == 200:
                                with open(os.path.join(save_dir, filename), "wb") as f:
                                    f.write(img_resp.content)
                            else:
                                print(f"Failed to download image: {img_resp.status_code}")
                        
                        known_faces[filename] = {"userId": user_id, "name": user["name"]}
                        
        except Exception as e:
            print(f"Sync error: {e}")
        
        await asyncio.sleep(60) # Sync every minute

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

def generate_frames():
    while True:
        with camera_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()
        
        # Draw rectangle if face detected (optional, requires detection logic here)
        # For now just stream raw video
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.post("/verify")
async def verify_face():
    """Manually trigger verification on current frame"""
    global latest_frame
    with camera_lock:
        if latest_frame is None:
            return {"success": False, "message": "No camera frame"}
        frame = latest_frame.copy()

    # Save frame temporarily for verifier
    config = Config()
    input_path = os.path.join(config.paths.input_image_path, "input_image.jpg")
    cv2.imwrite(input_path, frame)

    try:
        # Run verification
        # The verifier.verify() method compares input_image.jpg against ALL images in verification_images
        # It returns a list of results. We need to find the best match.
        # Note: The existing verifier.verify() returns (results, is_verified) but logic seems to be 1:1 or 1:N?
        # Let's look at verifier.py again. It iterates all images.
        
        results = []
        verification_path = config.paths.verification_images_path
        valid_images = [img for img in os.listdir(verification_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        best_score = 0
        best_match = None
        
        # We need to access the model directly or modify verifier to return scores
        # The existing verifier.verify() returns a list of predictions.
        # Let's use the verifier's internal logic but adapted here for 1:N
        
        input_img = verifier.preprocessor.preprocess(input_path)
        
        for image_name in valid_images:
            val_img_path = os.path.join(verification_path, image_name)
            val_img = verifier.preprocessor.preprocess(val_img_path)
            
            # Predict
            # model.predict expects list of [input_batch, validation_batch]
            input_batch = np.expand_dims(input_img, axis=0)
            val_batch = np.expand_dims(val_img, axis=0)
            score = verifier.model.predict([input_batch, val_batch], verbose=0)[0][0]
            
            if score > 0.5: # Threshold
                if score > best_score:
                    best_score = score
                    best_match = image_name

        if best_match:
            user_info = known_faces.get(best_match, {"name": "Unknown", "userId": "unknown"})
            
            # Log to backend
            try:
                async with httpx.AsyncClient() as client:
                    # Convert frame to bytes for upload
                    _, img_encoded = cv2.imencode('.jpg', frame)
                    files = {"snapshot": ("snapshot.jpg", img_encoded.tobytes(), "image/jpeg")}
                    data = {
                        "accessPointName": ROOM_NAME,
                        "userId": user_info["userId"],
                        "accessType": "IN" # Default to IN
                    }
                    # Fire and forget logging or wait? Let's wait to ensure it's logged
                    await client.post(f"{BACKEND_URL}/log", data=data, files=files)
            except Exception as log_error:
                print(f"Failed to log access: {log_error}")
            
            return {
                "success": True, 
                "match": True, 
                "user": {
                    "name": user_info["name"],
                    "id": user_info["userId"]
                }, 
                "score": float(best_score)
            }
        else:
            return {"success": True, "match": False}

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
