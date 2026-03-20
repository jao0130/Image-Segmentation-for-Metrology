FROM python:3.10-slim

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .env.example .env

CMD ["python", "src/main.py"]
