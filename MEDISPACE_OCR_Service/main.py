"""
main.py - FastAPI Entry Point cho MEDISPACE OCR Service
Pipeline: Ảnh đơn thuốc → PaddleOCR (detect) → VietOCR (recognize) → Custom LLM API → JSON
"""
# Load .env đầu tiên
import os
from dotenv import load_dotenv
load_dotenv()

import cv2
import time
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.services.detector import detect_text_regions, get_paddle_ocr
from src.services.recognizer import extract_full_text, get_vietocr, move_vietocr_to_cpu, move_vietocr_to_gpu
from src.services.extractor import extract_prescription_info
from src.utils.debug_logger import save_debug_log

app = FastAPI(
    title="MEDISPACE OCR Service",
    description="API nhận diện và trích xuất thông tin đơn thuốc Tiếng Việt",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def preload_models():
    """Pre-load tat ca AI models khi server khoi dong, khong cho request dau tien."""
    print("\n[Startup] Dang pre-load cac AI models...")
    t_start = time.time()
    get_paddle_ocr()      # Pre-load PaddleOCR
    get_vietocr()         # Pre-load VietOCR len GPU

    # Custom LLM API la remote service, khong can pre-load o day.
    base_url = os.getenv("CUSTOM_LLM_BASE_URL", "https://llm.datateam.space")
    model_name = os.getenv("CUSTOM_LLM_MODEL", "gemma-4-e4b-it.gguf")
    print(f"[Startup] Custom LLM API: {model_name} tai {base_url}")

    t_end = time.time()
    print(f"[Startup] Tat ca models da san sang! (Mat {t_end - t_start:.2f}s)")
    print(f"[Startup] Server san sang nhan request!\n")


def read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không thể đọc file ảnh. Hãy thử định dạng JPG/PNG/WEBP.")
    return image


@app.get("/")
async def root():
    base_url = os.getenv("CUSTOM_LLM_BASE_URL", "https://llm.datateam.space")
    model_name = os.getenv("CUSTOM_LLM_MODEL", "gemma-4-e4b-it.gguf")
    engine_name = f"PaddleOCR + VietOCR + Custom LLM ({model_name} @ {base_url})"

    return {
        "service": "MEDISPACE OCR Service",
        "status": "running",
        "version": "1.0.0",
        "engine": engine_name,
        "extractor_backend": "custom"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/ocr/extract-prescription")
async def extract_prescription(file: UploadFile = File(...)):
    """
    ★ ENDPOINT CHÍNH ★
    Pipeline đầy đủ:
    - Trạm 1: PaddleOCR phát hiện vùng chữ
    - Trạm 2: VietOCR đọc text Tiếng Việt
    - Trạm 3: Local LLM (Ollama) / Cloud LLM (Gemini) trích xuất thành JSON
    """
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không hỗ trợ: {file.content_type}. Chỉ nhận: JPG, PNG, WEBP"
        )

    try:
        file_bytes = await file.read()
        image = read_image_from_upload(file_bytes)

        print(f"\n{'='*50}")
        print(f"[Pipeline] Bắt đầu: {file.filename} ({image.shape})")

        # Trạm 1: PaddleOCR
        print("[Pipeline] Trạm 1: PaddleOCR phát hiện vùng chữ...")
        t1_start = time.time()
        boxes = detect_text_regions(image)
        t1_end = time.time()
        print(f"[Pipeline] Tìm thấy {len(boxes)} vùng chữ (Xong Trạm 1 mất: {t1_end - t1_start:.2f}s)")

        if not boxes:
            return {
                "success": False,
                "message": "Không tìm thấy chữ trong ảnh.",
                "rawText": "",
                "data": None
            }

        # Trạm 2: VietOCR
        print("[Pipeline] Trạm 2: VietOCR nhận diện text...")
        t2_start = time.time()
        raw_text = extract_full_text(image, boxes)
        t2_end = time.time()
        print(f"[Pipeline] Text ({len(raw_text)} ký tự - Xong Trạm 2 mất: {t2_end - t2_start:.2f}s):\n{raw_text[:400]}")

        # === Dynamic VRAM Swap: Now handled inside extractor.py when LLM is needed ===
        backend = os.getenv("EXTRACTOR_BACKEND", "ollama").lower()

        # Trạm 3: Extractor (Hybrid Regex + LLM)
        print(f"[Pipeline] Trạm 3: Đang trích xuất JSON (Hybrid)...")
        t3_start = time.time()
        extracted_data = extract_prescription_info(raw_text)
        t3_end = time.time()
        
        method = extracted_data.get('_extraction_method', 'unknown')
        print(f"[Pipeline] Hoàn thành! Method: {method}, Confidence: {extracted_data.get('confidence')} (Xong Trạm 3 mất: {t3_end - t3_start:.2f}s)")
        
        total_time = t3_end - t1_start
        print(f"[Pipeline] ★ Tổng thời gian pipeline: {total_time:.2f}s ★")

        timing_data = {
            "station1_PaddleOCR_seconds": round(t1_end - t1_start, 2),
            "station2_VietOCR_seconds": round(t2_end - t2_start, 2),
            "station3_Extractor_seconds": round(t3_end - t3_start, 2),
            "total_pipeline_seconds": round(total_time, 2),
            "extractor_backend": backend.lower()
        }

        # ★ Debug log — ghi ra file để dễ debug
        try:
            save_debug_log(
                filename=file.filename,
                raw_text=raw_text,
                extracted_data=extracted_data,
                timing=timing_data,
                image_shape=image.shape,
                num_boxes=len(boxes),
            )
        except Exception as log_err:
            print(f"[Pipeline] Debug log error (non-fatal): {log_err}")

        return {
            "success": True,
            "message": "Trích xuất đơn thuốc thành công",
            "rawText": raw_text,
            "data": extracted_data,
            "timing": timing_data
        }

    except ValueError as e:
        print(f"[Pipeline] ValueError: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"[Pipeline] Lỗi: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")


@app.post("/api/ocr/extract-text")
async def extract_text_only(file: UploadFile = File(...)):
    """Chỉ OCR text (PaddleOCR + VietOCR), không gọi LLM. Dùng để debug."""
    try:
        file_bytes = await file.read()
        image = read_image_from_upload(file_bytes)
        
        t1_start = time.time()
        boxes = detect_text_regions(image)
        t1_end = time.time()
        
        t2_start = time.time()
        raw_text = extract_full_text(image, boxes)
        t2_end = time.time()
        
        return {
            "success": True,
            "totalRegions": len(boxes),
            "time_PaddleOCR_seconds": round(t1_end - t1_start, 2),
            "time_VietOCR_seconds": round(t2_end - t2_start, 2),
            "time_total_OCR_seconds": round(t2_end - t1_start, 2),
            "rawText": raw_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")


# ─── Debug: Test LLM trực tiếp qua Swagger ───────────────────────────────────

class TestLLMRequest(BaseModel):
    raw_text: str = """PHÒNG KHÁM ĐA KHOA ABC\nBác sĩ: Nguyễn Văn Bình\nBệnh nhân: Trần Thị Mai, 45 tuổi, Nữ\nSĐT: 0901234567\nNgày: 15/04/2025\nChẩn đoán: Viêm họng cấp\n\nĐơn thuốc:\n1. Amoxicillin 500mg - Sáng 1 viên, Tối 1 viên - 10 viên\n2. Paracetamol 500mg - Uống khi sốt - 20 viên"""

    class Config:
        json_schema_extra = {
            "example": {
                "raw_text": "Bác sĩ: Nguyễn Văn A\nBệnh nhân: Lê Thị B, 30 tuổi\nThuốc: Paracetamol 500mg - 2 lần/ngày - 10 viên"
            }
        }


@app.post(
    "/api/ocr/test-llm",
    summary="[DEBUG] Test Custom LLM API bằng raw text",
    description=(
        "Gọi trực tiếp Custom LLM API với raw text bạn tự nhập, **bỏ qua** bước OCR ảnh.\n\n"
        "Dùng để:\n"
        "- Xác nhận LLM API có hoạt động không\n"
        "- Kiểm tra chất lượng trích xuất JSON\n"
        "- Debug prompt khi regex không đủ dữ liệu\n\n"
        f"**Endpoint LLM:** `{os.getenv('CUSTOM_LLM_BASE_URL', 'https://llm.datateam.space')}/v1/chat/completions`\n\n"
        f"**Model:** `{os.getenv('CUSTOM_LLM_MODEL', 'gemma-4-e4b-it.gguf')}`"
    ),
    tags=["Debug"]
)
async def test_llm_direct(body: TestLLMRequest):
    """
    [DEBUG] Gửi raw_text trực tiếp đến Custom LLM API và trả về JSON đơn thuốc.
    Không cần upload ảnh, không qua PaddleOCR / VietOCR.
    """
    if not body.raw_text or not body.raw_text.strip():
        raise HTTPException(status_code=400, detail="raw_text không được để trống")

    from src.services.extractor import _extract_with_custom_api

    t_start = time.time()
    result = _extract_with_custom_api(body.raw_text)
    elapsed = round(time.time() - t_start, 2)

    return {
        "success": "error" not in result,
        "message": "Trích xuất thành công" if "error" not in result else result.get("error"),
        "llm_seconds": elapsed,
        "model": os.getenv("CUSTOM_LLM_MODEL", "gemma-4-e4b-it.gguf"),
        "base_url": os.getenv("CUSTOM_LLM_BASE_URL", "https://llm.datateam.space"),
        "data": result
    }
