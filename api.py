from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io
import numpy as np
from typing import Dict, Any
import os
import time
from concurrent.futures import ThreadPoolExecutor
import functools
import asyncio

# Import the model loader
from model_loader import get_solver, CaptchaSolver

app = FastAPI(title="Captcha Solver API")

# Global model instance
solver = None

# Configure thread pool for handling model inference
# Number of workers should be adjusted based on your hardware
inference_pool = ThreadPoolExecutor(max_workers=4)

@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    global solver
    print("Loading CAPTCHA solver model...")
    start_time = time.time()
    solver = get_solver()
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when shutting down"""
    print("Shutting down and cleaning up resources...")
    # Any cleanup needed for the model

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_html():
    """Serve the HTML upload page"""
    return FileResponse(os.path.join(static_dir, "index.html"))

# This function now uses the globally loaded model
async def solve_captcha(image) -> Dict[str, Any]:
    """
    Takes an image and returns the predicted text and confidence score
    using the PyTorch CAPTCHA solver model.
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Use the global solver instance
    global solver
    
    # Define a synchronous function for the worker pool
    def predict_sync(img):
        start_time = time.time()
        prediction, confidence = solver.predict(img)
        inference_time = time.time() - start_time
        return prediction, confidence, inference_time
    
    # Run inference in the thread pool
    loop = asyncio.get_running_loop()
    prediction, confidence, inference_time = await loop.run_in_executor(
        inference_pool, 
        functools.partial(predict_sync, image)
    )
    
    return {
        "text": prediction,
        "confidence": float(confidence),
        "inference_time": inference_time
    }

@app.post("/predict/", response_model=Dict[str, Any])
async def upload_captcha(file: UploadFile = File(...)):
    """
    Endpoint to solve a captcha image.
    Accepts an image file and returns the predicted text and confidence.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Process with the CAPTCHA solving model
        result = await solve_captcha(image)
        
        return {
            "success": True,
            "prediction": result["text"],
            "confidence": result["confidence"],
            "inference_time": result["inference_time"]
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)