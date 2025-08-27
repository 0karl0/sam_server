FROM intel/intel-optimized-pytorch

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Download YOLO models trained on artwork
RUN mkdir -p /models && \
    curl -L https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8n.pt -o /models/yolov8n-art.pt && \
    curl -L https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8s.pt -o /models/yolov8s-art.pt

# Copy and install Python dependencies
COPY server2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server2/ .

CMD ["python3", "worker.py"]
