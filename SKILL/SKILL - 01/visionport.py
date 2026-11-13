import io
import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

# --------------------------------------------------
# 1. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(title="Indian Traffic Detection API")

# --------------------------------------------------
# 2. Enable CORS for VisionPort HTML frontend
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "null",
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --------------------------------------------------
# 3. Load YOLOv8 Nano model
# --------------------------------------------------

print("Loading YOLOv8 Nano model...")
model = YOLO("yolov8n.pt")
print("YOLO model loaded successfully!")

# --------------------------------------------------
# 4. Home endpoint
# --------------------------------------------------

@app.get("/")
async def home():
    return {
        "status": "Active",
        "message": "Indian Traffic Detection API is running!"
    }

# --------------------------------------------------
# 5. Health check endpoint
# --------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "Active",
        "model": "YOLOv8 Nano",
        "device": "CPU"
    }

# --------------------------------------------------
# 6. Object Detection endpoint
# --------------------------------------------------

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):

    try:
        # Read uploaded image
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        nparr = np.frombuffer(contents, np.uint8)

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file."
            )

        # Run YOLO inference on CPU
        results = model.predict(
            source=img,
            imgsz=416,
            conf=0.25,
            device="cpu",
            verbose=False
        )

        # Annotate image with bounding boxes
        annotated_img = results[0].plot()

        # Encode image as JPEG
        success, encoded_img = cv2.imencode(
            ".jpg",
            annotated_img
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Image encoding failed."
            )

        # Return processed image
        return StreamingResponse(
            io.BytesIO(encoded_img.tobytes()),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store"
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        print("Detection Error:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# --------------------------------------------------
# 7. Start server
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
