# Server1.Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Download YOLO models trained on artwork
RUN mkdir -p /models && \
    curl -L https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8n.pt -o /models/yolov8n-art.pt && \
    curl -L https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8s.pt -o /models/yolov8s-art.pt

# Install deps
COPY server1/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY server1/ .

# Flask runs on port 5000
EXPOSE 5050

CMD ["python", "app.py"]
