# =====================================================
# MEDISPACE OCR Service
# Port: 8001
# Stack: Python 3.11 + FastAPI + PaddleOCR + VietOCR
# =====================================================

FROM python:3.14-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt .

# 1. Torch CPU (CUDA version không dùng được trong Docker)
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu

# 2. OpenCV headless 4.9.x — thỏa mãn albumentations (>=4.9.0)
#    paddleocr 2.7.3 declare <=4.6.0.66 nhưng code thực tế tương thích 4.9.x
RUN pip install --no-cache-dir "opencv-python-headless==4.9.0.80"

# 3. PaddleOCR --no-deps (bỏ qua opencv constraint quá strict)
RUN pip install --no-cache-dir --no-deps paddleocr==2.7.3

# 4. VietOCR --no-deps (tránh kéo albumentations tự động)
RUN pip install --no-cache-dir --no-deps vietocr==0.3.13

# 5. Albumentations + imgaug --no-deps (tránh check opencv >=4.9.0)
RUN pip install --no-cache-dir --no-deps albumentations==1.4.2 imgaug==0.4.0

# 6. Phần còn lại của requirements
RUN pip install --no-cache-dir -r requirements.docker.txt

# Copy source code
COPY . .

RUN mkdir -p /app/logs /root/.paddleocr /root/.cache/vietocr

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=120s \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
