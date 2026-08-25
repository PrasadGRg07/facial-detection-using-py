FROM python:3.11-slim

# Install system dependencies required for OpenCV and dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Install face_recognition_models directly if it's missing in pip dependencies
# RUN pip install git+https://github.com/ageitgey/face_recognition_models

# Copy source code
COPY . .

# Run the app using gunicorn - Railway injects $PORT automatically
CMD gunicorn --bind 0.0.0.0:$PORT --threads 4 app:app
