import os
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
import numpy as np
from typing import Dict, Any, List, Tuple
import torch.nn.functional as F
import cv2

# Configuration
IMG_WIDTH = 65
IMG_HEIGHT = 25
BLANK_TOKEN = 0
DROPOUT_RATE = 0.3

class Preprocessor:
    """Preprocessor with upscaling for improved CAPTCHA recognition
    with parameters optimized through training"""
    
    def __init__(self, 
                 upscale_factor=1, 
                 upscale_method='bicubic',
                 contrast_factor=1,
                 brightness_factor=1,
                 sharpen_after_upscale=True,
                 sharpen_amount=1,
                 sharpen_blend=1,
                 edge_enhancement=True): 
        self.upscale_factor = upscale_factor
        self.upscale_method = upscale_method
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor
        self.sharpen_after_upscale = sharpen_after_upscale
        self.sharpen_amount = sharpen_amount
        self.sharpen_blend = sharpen_blend
        self.edge_enhancement = edge_enhancement
        
    def enhance_edges(self, pil_img):
        # Convert to numpy array for OpenCV processing
        img = np.array(pil_img)
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Apply unsharp masking for edge enhancement
        gaussian_blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
        enhanced = cv2.addWeighted(gray, 2.0, gaussian_blur, -1.0, 0)
        
        # Return as PIL Image
        return Image.fromarray(enhanced)
    
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
        
        # Apply brightness adjustment (new from trained model)
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

        if hasattr(self, 'edge_enhancement'):
            img = self.enhance_edges(img)
        
        # Resize to the dimensions expected by the model
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        
        return img
    
class CaptchaModel(nn.Module):
    def __init__(self, num_chars, dropout_rate=DROPOUT_RATE):
        super().__init__()
        
        # Enhanced CNN feature extraction with residual connections
        self.cnn_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout_rate/2),
            
            # Second conv block with residual connection
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout_rate/2)
        )
        
        # Calculate the output dimensions after CNN layers
        cnn_output_height = IMG_HEIGHT // 4  # After two max-pooling layers
        cnn_output_width = IMG_WIDTH // 4
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(64 * cnn_output_height, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Bidirectional GRU (faster than LSTM with similar performance)
        self.rnn = nn.GRU(
            input_size=64 * cnn_output_height,
            hidden_size=128,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if dropout_rate > 0 else 0
        )
        
        # Classification layer with layer normalization
        self.layer_norm = nn.LayerNorm(256)  # 2 * hidden_size due to bidirectional
        self.classifier = nn.Linear(256, num_chars + 1)  # +1 for blank token
        
    def forward(self, x):
        # Extract features with CNN
        batch_size = x.size(0)
        x = self.cnn_layers(x)
        
        # Reshape for sequence processing: [batch, channels, height, width] -> [batch, width, channels*height]
        x = x.permute(0, 3, 1, 2)
        _, seq_len, channels, height = x.size()
        x = x.reshape(batch_size, seq_len, channels * height)
        
        # Apply attention
        e = self.attention(x).squeeze(-1)
        alpha = F.softmax(e, dim=1).unsqueeze(1)
        
        # RNN processing
        rnn_out, _ = self.rnn(x)
        
        # Apply layer normalization
        x = self.layer_norm(rnn_out)
        
        # Classification
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=2)


def decode_predictions(outputs, idx_to_char):
    """Convert model outputs to text predictions with fixed length of 4 characters"""
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

        # Ensure exactly 4 characters
        if len(text) < 4:
            # Pad with placeholder characters if less than 4
            text = text + 'A' * (4 - len(text))
        elif len(text) > 4:
            # Truncate to 4 characters if longer
            text = text[:4]

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
            upscale_factor=2.0,             # Increased for better detail
            upscale_method='lanczos',       # Changed from 'bicubic' for better edge preservation
            contrast_factor=4,              # Increased for more defined lines
            brightness_factor=1.1,          # Slightly reduced to prevent washing out details
            sharpen_after_upscale=True,     
            sharpen_amount=2.3,             # Significantly increased for sharper lines
            sharpen_blend=0.8,              # Increased blend factor for stronger effect
            edge_enhancement=True           # Enable additional edge enhancement
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
    
    def predict(self, image):
            """
            Predict the text in a CAPTCHA image
            
            Args:
                image: PIL Image object
                
            Returns:
                tuple: (predicted_text, confidence_score)
            """
            try:
                # First preprocess the image using our trained preprocessor
                processed_img = self.preprocessor.preprocess(image)
                
                # Convert to tensor and normalize to [0, 1]
                img_tensor = torch.FloatTensor(np.array(processed_img)).unsqueeze(0).unsqueeze(0) / 255.0
                
                # Move to the same device as model
                img_tensor = img_tensor.to(self.device)
                
                # Get predictions from model
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                
                # Decode predictions
                predictions = decode_predictions(outputs, self.idx_to_char)
                prediction = predictions[0] if predictions else ""
                
                # Calculate confidence score
                # Use the highest softmax probability for each character as confidence
                probs = torch.exp(outputs)
                max_probs, _ = torch.max(probs, dim=2)
                confidence = float(max_probs.mean().item())
                
                print(f"Predicted: {prediction}, Confidence: {confidence:.4f}")
                
                return prediction, confidence
            
            except Exception as e:
                print(f"Error in prediction: {e}")
                return "", 0.0

# Create a singleton instance
captcha_solver = None

def get_solver():
    global captcha_solver
    if captcha_solver is None:
        captcha_solver = CaptchaSolver()
    return captcha_solver