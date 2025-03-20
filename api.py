from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io
import numpy as np
from typing import Dict, Any
import os

# Import the model loader
from model_loader import get_solver

app = FastAPI(title="Captcha Solver API")

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_html():
    """Serve the HTML upload page"""
    return FileResponse(os.path.join(static_dir, "index.html"))

# This function now uses the PyTorch model
async def solve_captcha(image) -> Dict[str, Any]:
    """
    Takes an image and returns the predicted text and confidence score
    using the PyTorch CAPTCHA solver model.
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Get the solver instance
    solver = get_solver()
    
    # Get prediction and confidence
    prediction, confidence = solver.predict(image)
    
    return {
        "text": prediction,
        "confidence": float(confidence)  # Ensure it's a Python float
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
            "confidence": result["confidence"]
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)