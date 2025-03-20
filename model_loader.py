import os
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
import numpy as np
from typing import Dict, Any, List, Tuple

# Configuration
IMG_WIDTH = 65
IMG_HEIGHT = 25
BLANK_TOKEN = 0

class Preprocessor:
    """Preprocessor with upscaling for improved CAPTCHA recognition"""
    
    def __init__(self, 
                 upscale_factor=2.0, 
                 upscale_method='bicubic',
                 contrast_factor=0.95,
                 brightness_factor=1.10,
                 sharpen_after_upscale=True,
                 sharpen_amount=0.68,
                 sharpen_blend=0.66): 
        self.upscale_factor = upscale_factor
        self.upscale_method = upscale_method
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor
        self.sharpen_after_upscale = sharpen_after_upscale
        self.sharpen_amount = sharpen_amount
        self.sharpen_blend = sharpen_blend
    
    def preprocess(self, pil_img):
        """Apply preprocessing with parameters learned from trainable model"""
        # Ensure grayscale
        img = pil_img.convert('L')
        
        # Calculate upscaled dimensions while maintaining aspect ratio
        orig_width, orig_height = img.size
        upscaled_width = int(orig_width * self.upscale_factor)
        upscaled_height = int(orig_height * self.upscale_factor)
        
        # Map string method names to PIL constants
        resize_methods = {
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS,
            'bilinear': Image.BILINEAR,
            'nearest': Image.NEAREST
        }
        
        # Apply upscaling with selected method
        resize_method = resize_methods.get(self.upscale_method.lower(), Image.BICUBIC)
        img = img.resize((upscaled_width, upscaled_height), resize_method)
        
        # Apply brightness adjustment
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(self.brightness_factor)
        
        # Apply sharpening with learned parameters
        if self.sharpen_after_upscale:
            # Create a sharpened version
            enhancer = ImageEnhance.Sharpness(img)
            sharpened_img = enhancer.enhance(1.0 + self.sharpen_amount)
            
            # Create a new image that blends original and sharpened based on sharpen_blend
            if hasattr(Image, 'blend'):  # For newer PIL versions
                img = Image.blend(img, sharpened_img, self.sharpen_blend)
            else:
                # Manual blending using pixel manipulation for older PIL versions
                img_data = np.array(img).astype(float)
                sharp_data = np.array(sharpened_img).astype(float)
                blended_data = img_data * (1 - self.sharpen_blend) + sharp_data * self.sharpen_blend
                img = Image.fromarray(np.clip(blended_data, 0, 255).astype(np.uint8))
        
        # Apply contrast with learned parameter
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.contrast_factor)
        
        # Resize to the dimensions expected by the model
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        
        return img
class CaptchaModel(nn.Module):
    def __init__(self, num_chars, dropout_rate=0.3):
        super().__init__()

        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(2, 2)
        )

        cnn_output_height = IMG_HEIGHT // 2

        # Reshaping layer
        self.reshape = nn.Linear(12 * cnn_output_height, 48)
        self.dropout = nn.Dropout(dropout_rate)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=48,
            hidden_size=48,
            bidirectional=True,
            batch_first=True
        )

        # Classification layer
        self.classifier = nn.Linear(96, num_chars + 1)  # +1 for blank token

    def forward(self, x):
        # Feature extraction
        x = self.cnn(x)

        # Reshape for sequence processing
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).reshape(batch_size, width, channels * height)

        # Sequence processing
        x = self.reshape(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)

        # Classification
        x = self.classifier(x)
        return nn.functional.log_softmax(x, dim=2)


def decode_predictions(outputs, idx_to_char):
    """Convert model outputs to text predictions"""
    predictions = []
    output_args = torch.argmax(outputs.detach().cpu(), dim=2)

    for pred in output_args:
        text = ''
        prev_char = None

        for p in pred:
            p_item = p.item()
            # Only add character if it's not blank and not a repeat
            if p_item != BLANK_TOKEN and p_item != prev_char:
                if p_item in idx_to_char:
                    text += idx_to_char[p_item]
            prev_char = p_item

        predictions.append(text)

    return predictions


class CaptchaSolver:
    def __init__(self, model_path="captcha_model.pth", device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Try to load the model first to determine the correct number of characters
        model_exists = os.path.exists(model_path)
        
        # Default character set - customize based on your CAPTCHA requirements
        # For this example, we're using a reduced set that matches the model
        chars = "2345789ABCDEFHKLMNPRTUVWXYZ"  # Changed to match the model (27 chars)
        self.idx_to_char = {idx+1: char for idx, char in enumerate(chars)}
        num_chars = len(self.idx_to_char)
        
        # Create model with the correct number of output classes
        self.model = CaptchaModel(num_chars)
        
        # Load model weights if available
        if model_exists:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Loading model with strict=False to inspect state_dict...")
                # Try loading non-strictly to get the actual state_dict
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Check the classifier shape to determine the actual number of classes
                if 'classifier.weight' in checkpoint:
                    classifier_shape = checkpoint['classifier.weight'].shape
                    actual_classes = classifier_shape[0]
                    print(f"Model has {actual_classes-1} characters (plus blank token)")
                    
                    # Recreate the model with the correct number of classes
                    adjusted_chars = chars[:actual_classes-1]  # Adjust character set to match model
                    self.idx_to_char = {idx+1: char for idx, char in enumerate(adjusted_chars)}
                    self.model = CaptchaModel(actual_classes-1)
                    
                    # Try loading again
                    try:
                        self.model.load_state_dict(checkpoint)
                        print("Model loaded successfully with adjusted character set")
                    except Exception as e2:
                        print(f"Still could not load model: {e2}")
        else:
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create preprocessor
        self.preprocessor = Preprocessor(
            upscale_factor=2.0,
            upscale_method='bicubic',
            contrast_factor=0.95,
            brightness_factor=1.10,
            sharpen_after_upscale=True,
            sharpen_amount=0.68,
            sharpen_blend=0.66
        )
    
    def preprocess_image(self, image):
        """Preprocess the image for the model"""
        # Use the preprocessor
        processed_img = self.preprocessor.preprocess(image)
        
        # Convert to tensor and normalize
        img_array = np.array(processed_img) / 255.0
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        img_tensor = (img_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
        
        return img_tensor.to(self.device)
    
    def predict(self, image) -> Tuple[str, float]:
        """
        Predict the text in the image and return confidence
        
        Args:
            image: PIL Image object
            
        Returns:
            tuple: (predicted_text, confidence)
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
        # Decode prediction
        predictions = decode_predictions(outputs, self.idx_to_char)
        predicted_text = predictions[0]  # Get the first prediction (batch size is 1)
        
        # Calculate confidence
        # For a simple approach, we'll use the average softmax probability of the predicted characters
        probs = torch.exp(outputs[0])  # Convert log_softmax to probabilities
        argmax_indices = torch.argmax(outputs[0], dim=1)
        
        # Get the probabilities of the selected characters
        char_probs = []
        for i, idx in enumerate(argmax_indices):
            prob = probs[i, idx].item()
            if idx != BLANK_TOKEN and idx.item() in self.idx_to_char:
                char_probs.append(prob)
        
        # Calculate average confidence (or 0 if no characters were recognized)
        confidence = sum(char_probs) / len(char_probs) if char_probs else 0.0
        
        return predicted_text, confidence

# Create a singleton instance
captcha_solver = None

def get_solver():
    global captcha_solver
    if captcha_solver is None:
        captcha_solver = CaptchaSolver()
    return captcha_solver