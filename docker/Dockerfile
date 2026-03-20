FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Python and OpenCV system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    libgl1-mesa-glx libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch (CUDA 12.4) then remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY weights/ ./weights/
COPY .env.example .env

CMD ["python3", "src/main.py"]
