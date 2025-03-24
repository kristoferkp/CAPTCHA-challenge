FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies needed for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt /app/

# Install Python packages from requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and Python files
COPY captcha_model.pth /app/
COPY model_loader.py /app/
COPY api.py /app/

# Expose port for the API
EXPOSE 8000

# Command to run the API
CMD ["python", "api.py"]
