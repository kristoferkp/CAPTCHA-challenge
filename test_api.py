import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
from PIL import Image
import numpy as np
import os
import random

from api import app

# Create a test client
client = TestClient(app)

@pytest.fixture
def test_image_with_value():
    """
    Load an actual CAPTCHA image from data/images folder.
    Return the image and the expected value (extracted from filename).
    """
    # Path to images directory
    image_dir = os.path.join(os.path.dirname(__file__), "data", "images")
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) 
                  if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        pytest.skip("No test images found in data/images directory")
    
    # Select a random image
    image_file = random.choice(image_files)
    
    # Extract expected value from filename (assuming filename format like "ABC123.png")
    expected_value = os.path.splitext(image_file)[0]
    
    # Load the image
    image_path = os.path.join(image_dir, image_file)
    img = Image.open(image_path)
    
    # Convert to bytes for request
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=img.format or 'PNG')
    img_byte_arr.seek(0)
    
    return {
        "image": img_byte_arr,
        "expected_value": expected_value,
        "filename": image_file,
        "pil_image": img
    }

@patch('api.solve_captcha')
def test_predict_endpoint_success(mock_solve_captcha, test_image_with_value):
    """Test successful prediction with a real CAPTCHA image."""
    # Get image data
    img_data = test_image_with_value
    expected_value = img_data["expected_value"]
    
    # Configure the mock to return the expected value from filename
    mock_solve_captcha.return_value = {
        "text": expected_value,
        "confidence": 0.95,
        "inference_time": 0.1
    }
    
    response = client.post(
        "/predict/",
        files={"file": (img_data["filename"], img_data["image"], "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["prediction"] == expected_value
    assert data["confidence"] == 0.95
    assert "inference_time" in data

def test_predict_endpoint_invalid_file_type():
    """Test error when non-image file is uploaded."""
    text_file = io.BytesIO(b"This is not an image")
    response = client.post(
        "/predict/",
        files={"file": ("file.txt", text_file, "text/plain")}
    )
    
    assert response.status_code == 400
    assert "File must be an image" in response.text

@patch('api.solve_captcha')
def test_predict_endpoint_model_error(mock_solve_captcha, test_image_with_value):
    """Test error handling when model inference fails."""
    # Get image data
    img_data = test_image_with_value
    
    mock_solve_captcha.side_effect = Exception("Model error")
    
    response = client.post(
        "/predict/",
        files={"file": (img_data["filename"], img_data["image"], "image/png")}
    )
    
    assert response.status_code == 500
    data = response.json()
    assert data["success"] is False
    assert "error" in data

@patch('api.solver')
def test_solve_captcha_function(mock_solver, test_image_with_value):
    """Test the solve_captcha function directly."""
    from api import solve_captcha
    import asyncio
    
    # Get image data
    img_data = test_image_with_value
    expected_value = img_data["expected_value"]
    
    # Configure the mock
    mock_solver.predict.return_value = (expected_value, 0.85)
    
    # Run the function with synchronous execution
    result = asyncio.run(solve_captcha(img_data["pil_image"]))
    
    # Verify the result
    assert result["text"] == expected_value
    assert result["confidence"] == 0.85
    assert "inference_time" in result
