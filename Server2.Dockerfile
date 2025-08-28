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

# Copy and install Python dependencies
COPY server2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server2/ .

CMD ["python3", "worker.py"]
