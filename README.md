# MEDISPACE Python Services

> Monorepo chứa các Python microservices cho dự án MediSpace E-Commerce.

## Cấu Trúc

```
MEDISPACE_Python_Services/
├── MEDISPACE_OCR_Service/    # OCR Service (port 8001)
├── MEDISPACE_ML_Service/     # ML Recommendation Service (port 8002)
├── venv/                     # Shared venv cho ML Service
├── start_ocr.ps1             # Script khởi động OCR
├── start_ml.ps1              # Script khởi động ML
└── README.md
```

## Khởi Động Nhanh

### OCR Service (port 8001)
```powershell
# Chạy từ thư mục MEDISPACE_Python_Services/
.\start_ocr.ps1
```

### ML Recommendation Service (port 8002)
```powershell
# Chạy từ thư mục MEDISPACE_Python_Services/
.\start_ml.ps1
```

## Lưu Ý Venv

| Service | Venv Location |
|---|---|
| OCR Service | `../MEDISPACE_OCR_Service/venv/` (path cũ, có PaddleOCR/VietOCR/Torch) |
| ML Service | `./venv/` (shared venv, có scikit-learn/mlxtend/apscheduler) |

Monorepo chứa các Python microservices cho dự án MediSpace E-Commerce.

## Services

### 📄 MEDISPACE_OCR_Service (Port 8001)
Dịch vụ nhận dạng và trích xuất thông tin đơn thuốc Tiếng Việt.
- **Pipeline**: PaddleOCR → VietOCR → Ollama/Gemini LLM → JSON
- **Endpoint chính**: `POST /api/ocr/extract-prescription`

### 🤖 MEDISPACE_ML_Service (Port 8002)
Dịch vụ gợi ý sản phẩm dựa trên Machine Learning.
- **Algorithms**: TF-IDF + Cosine Similarity, FP-Growth, NMF Trending, SVD
- **Endpoints**: `/recommend/related`, `/recommend/bought-together`, `/recommend/trending`, `/recommend/for-you`

## Khởi Động

```bash
# OCR Service
cd MEDISPACE_OCR_Service
uvicorn main:app --reload --port 8001

# ML Service
cd MEDISPACE_ML_Service
uvicorn main:app --reload --port 8002
```

## Tech Stack
- Python 3.10+
- FastAPI + Uvicorn
- scikit-learn, scipy, pandas, numpy
- mlxtend (FP-Growth)
- APScheduler
- MongoDB (pymongo)
- PaddleOCR, VietOCR (OCR service only)
