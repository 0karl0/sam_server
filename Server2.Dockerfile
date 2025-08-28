FROM intel/intel-optimized-pytorch

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY server2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server2/ .

# Download example detection models (no-clobber) into /models
# before launching the worker.
CMD bash -c 'mkdir -p /models && \
    wget -nc -P /models https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8n.pt && \
    wget -nc -P /models https://github.com/lyuwenyu/rf-detr/releases/download/v0.1/rf_detr_r50.pth && \
    wget -nc -P /models https://github.com/lyuwenyu/RT-DETR/releases/download/v0.1/rt_detr_r50.pth && \
    wget -nc -P /models https://github.com/lyuwenyu/D-FINE/releases/download/v0.1/dfine_r18.pth && \
    python3 worker.py'
