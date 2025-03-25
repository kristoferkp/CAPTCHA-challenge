# CAPTCHA solver

## Making and running the Docker container

To build and run the Docker image for the CAPTCHA solver, use the following commands:

```bash
# Build the container
docker build -t captcha-solver:latest .

# Run the container
docker-compose up -d
```

To change the host or port, change the API_HOST or API_PORT environment variables in `docker-compose.yml`

## Example API calls

```bash
curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@5X97.png;type=image/png'
```

```python
import requests

headers = {
    'accept': 'application/json',
}

files = {
    'file': ('CAPTCHA.png', open('./data/images/5X97.png', 'rb'), 'image/png'),
}

response = requests.post('http://localhost:8000/predict/', headers=headers, files=files)

print(response.json())
```

```
{'success': True, 'prediction': '5X97', 'confidence': 0.9999451637268066, 'inference_time': 0.03582620620727539}
```

## Changes made to the model

### Model Architecture Improvements

The CAPTCHA recognition model has been significantly enhanced with the following architectural improvements:

1. **Enhanced CNN Feature Extraction**

   - Increased depth with multiple convolutional layers (1→32→64 channels vs. original 1→12)
   - Added residual connections to improve gradient flow
   - Incorporated additional BatchNorm and ReLU activations for better training stability

2. **Attention Mechanism**

   - Added an attention layer to focus on the most relevant features of the image
   - Helps the model distinguish between similar characters

3. **Improved Sequence Processing**

   - Replaced LSTM with bidirectional GRU (similar performance but faster inference)
   - Increased hidden dimensions from 48 to 128 for more expressive feature representation

4. **Regularization Techniques**
   - Implemented strategic dropout rates to prevent overfitting
   - Added layer normalization before classification for more stable training

These enhancements have resulted in:

- Higher accuracy on CAPTCHA images
- Improved resistance to noise and distortion
- Faster inference times with comparable model size

### Preprocessor

The project now incorporates an advanced image preprocessing pipeline that improves CAPTCHA recognition:

1. **Adaptive Image Upscaling**

   - Configurable upscaling factor and algorithm selection (bicubic, lanczos, bilinear)
   - Preserves character details that might be lost in low-resolution images

2. **Intelligent Image Enhancement**

   - Dynamic contrast and brightness adjustment
   - Customizable sharpening with blend control for optimal character definition
   - Edge enhancement using unsharp masking technique to highlight character boundaries

3. **Parameter Tuning System**
   - All preprocessing parameters can be fine-tuned for specific CAPTCHA types
   - Enables optimization for different visual styles and obfuscation techniques

The preprocessor works by:

1. Converting images to grayscale for consistent processing
2. Applying resolution enhancement through intelligent upscaling
3. Adjusting contrast and brightness to improve feature visibility
4. Enhancing edges using computer vision techniques
5. Optimizing the final image dimensions for the neural network

These preprocessing improvements have proven particularly effective for:

- Low-quality or deliberately blurred CAPTCHAs
- CAPTCHAs with overlapping characters
- Images with noise or distortion effects

---

These changes have improved the model precision from 40%-60% to 90%-95%

## Any assumptions made

1. Character set: The model is trained on the specific character set `2345789ABCDEFHKLMNPRTUVWXYZ` and may not recognize characters outside this set
2. CAPTCHA length: The model expects 4-character CAPTCHAs, as in the training examples
3. Image characteristics: The model works best with similar visual characteristics (font style, distortion patterns) to the training data
4. Image format: PNG format is expected for the images

## Overall impressions

This was a really great challenge and I really enjoyed it. I firstly I added a preprocessor, but the preprocessor didn't produce good enough output, something around 70%. Adding more epochs eventually improved the model to around 80%, but that still didn't feel enough improvement. So, I chose to modify the model itself. Modifying the model itself was a great challenge, but with a great amount of googling, I managed to up the prediction accuracy to around 95%.
I learned a lot about CAPTCHAs and Pytorch in general.
