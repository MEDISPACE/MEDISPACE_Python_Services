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
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, wait
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.services.detector import detect_text_regions_with_image, get_paddle_ocr
from src.services.recognizer import extract_full_text, get_vietocr, move_vietocr_to_cpu, move_vietocr_to_gpu
from src.services.extractor import extract_prescription_info
from src.services.quality import merge_candidates, normalize_prescription_data, score_candidate
from src.services.vision_extractor import extract_prescription_from_image
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

_liveness_server_started = False

def start_liveness_server() -> None:
    """Serve Docker liveness checks outside the FastAPI event loop."""
    global _liveness_server_started
    if _liveness_server_started:
        return

    class LivenessHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            body = b'{"status":"healthy"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    port = int(os.getenv("OCR_LIVENESS_PORT", "8002"))
    server = ThreadingHTTPServer(("0.0.0.0", port), LivenessHandler)
    thread = threading.Thread(target=server.serve_forever, name="ocr-liveness", daemon=True)
    thread.start()
    _liveness_server_started = True
    print(f"[Startup] Liveness server ready on port {port}")


@app.on_event("startup")
async def preload_models():
    """Pre-load tat ca AI models khi server khoi dong, khong cho request dau tien."""
    start_liveness_server()
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


def analyze_image_quality(image: np.ndarray) -> dict:
    """Return advisory image-quality signals before OCR continues."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    short_side = min(h, w)
    long_side = max(h, w)

    flags = []
    if blur_score < float(os.getenv("OCR_BLUR_WARN_THRESHOLD", "80")):
        flags.append("blurry")
    if brightness < float(os.getenv("OCR_DARK_WARN_THRESHOLD", "70")):
        flags.append("too_dark")
    if brightness > float(os.getenv("OCR_BRIGHT_WARN_THRESHOLD", "220")):
        flags.append("too_bright")
    if contrast < float(os.getenv("OCR_LOW_CONTRAST_WARN_THRESHOLD", "35")):
        flags.append("low_contrast")
    if short_side < int(os.getenv("OCR_MIN_SHORT_SIDE_WARN", "720")):
        flags.append("low_resolution")
    if long_side / max(short_side, 1) > float(os.getenv("OCR_EXTREME_ASPECT_WARN", "3.2")):
        flags.append("extreme_aspect_ratio")

    level = "good"
    if flags:
        level = "review"
    if len(flags) >= 3 or ("blurry" in flags and "low_resolution" in flags):
        level = "poor"

    return {
        "level": level,
        "flags": flags,
        "width": int(w),
        "height": int(h),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blurScore": round(blur_score, 2),
    }

def attach_image_quality(response: dict, image_quality: dict) -> dict:
    response["imageQuality"] = image_quality
    quality = response.get("quality")
    if isinstance(quality, dict):
        quality["imageQuality"] = image_quality
        if image_quality.get("flags"):
            flags = set(quality.get("flags") or [])
            flags.update(f"image_{flag}" for flag in image_quality.get("flags") or [])
            quality["flags"] = sorted(flags)
            quality["canEarlyReturn"] = False
            if image_quality.get("level") == "poor":
                quality["level"] = "low"
                quality["score"] = min(int(quality.get("score", 0) or 0), 54)
    return response

def run_traditional_pipeline(image: np.ndarray, filename: str, allow_llm_fallback: bool = True) -> dict:
    print(f"[Pipeline] Traditional OCR start: {filename} ({image.shape})")
    t1_start = time.time()
    boxes, ocr_image = detect_text_regions_with_image(image)
    t1_end = time.time()
    print(f"[Pipeline] Traditional found {len(boxes)} text boxes in {t1_end - t1_start:.2f}s")

    if not boxes:
        return {
            "success": False,
            "message": "No text found in image.",
            "rawText": "",
            "data": None,
            "timing": {
                "station1_PaddleOCR_seconds": round(t1_end - t1_start, 2),
                "station2_VietOCR_seconds": 0,
                "station3_Extractor_seconds": 0,
                "total_pipeline_seconds": round(t1_end - t1_start, 2),
                "extractor_backend": "none",
            },
            "numBoxes": 0,
        }

    t2_start = time.time()
    raw_text = extract_full_text(ocr_image, boxes)
    t2_end = time.time()
    print(f"[Pipeline] Traditional raw text length: {len(raw_text)} in {t2_end - t2_start:.2f}s")

    backend = os.getenv("EXTRACTOR_BACKEND", "custom").lower()
    t3_start = time.time()
    extracted_data = extract_prescription_info(raw_text, allow_llm_fallback=allow_llm_fallback)
    t3_end = time.time()

    extracted_data = normalize_prescription_data(extracted_data, "traditional")
    extracted_data["_extraction_method"] = extracted_data.get("_extraction_method") or "traditional_ocr"
    total_time = t3_end - t1_start

    return {
        "success": True,
        "message": "Prescription extracted successfully",
        "rawText": raw_text,
        "data": extracted_data,
        "timing": {
            "station1_PaddleOCR_seconds": round(t1_end - t1_start, 2),
            "station2_VietOCR_seconds": round(t2_end - t2_start, 2),
            "station3_Extractor_seconds": round(t3_end - t3_start, 2),
            "total_pipeline_seconds": round(total_time, 2),
            "extractor_backend": backend if allow_llm_fallback else "regex_only_no_llm_fallback",
        },
        "numBoxes": len(boxes),
    }


def run_vision_pipeline(file_bytes: bytes, content_type: str) -> dict:
    t_start = time.time()
    data = extract_prescription_from_image(file_bytes, content_type)
    elapsed = time.time() - t_start
    timing = {"vision_llm_seconds": round(elapsed, 2)}
    if isinstance(data, dict) and isinstance(data.get("_vision_timing"), dict):
        timing.update(data.get("_vision_timing") or {})
    return {
        "success": "error" not in data,
        "message": "Vision OCR completed" if "error" not in data else data.get("error"),
        "data": data,
        "timing": timing,
    }


def build_ocr_response(
    traditional_result: dict | None,
    vision_result: dict | None,
    mode: str,
    include_candidates: bool,
    started_at: float,
) -> dict:
    raw_text = traditional_result.get("rawText", "") if traditional_result else ""
    traditional_data = traditional_result.get("data") if traditional_result else None
    vision_data = vision_result.get("data") if vision_result else None

    traditional_quality = score_candidate(traditional_data, "traditional", bool(raw_text), raw_text) if traditional_result else None
    vision_quality = score_candidate(vision_data, "vision", False) if vision_result else None
    merged, merged_quality = merge_candidates(traditional_data, vision_data, traditional_quality, vision_quality, raw_text)

    timing = {"total_pipeline_seconds": round(time.time() - started_at, 2), "mode": mode}
    if traditional_result:
        timing.update({f"traditional_{k}": v for k, v in traditional_result.get("timing", {}).items()})
        timing["traditional_numBoxes"] = traditional_result.get("numBoxes", 0)
    if vision_result:
        timing.update(vision_result.get("timing", {}))

    has_successful_candidate = bool(merged_quality.get("usableMedicationCandidate"))

    response = {
        "success": has_successful_candidate,
        "message": "Prescription extracted successfully" if has_successful_candidate else "Prescription extraction needs manual review",
        "rawText": raw_text,
        "data": merged,
        "quality": merged_quality,
        "timing": timing,
    }

    if include_candidates:
        response["candidates"] = {
            "traditional": {
                "data": traditional_data,
                "quality": traditional_quality,
                "timing": traditional_result.get("timing", {}) if traditional_result else None,
                "success": traditional_result.get("success") if traditional_result else False,
            },
            "vision": {
                "data": vision_data,
                "quality": vision_quality,
                "timing": vision_result.get("timing", {}) if vision_result else None,
                "success": vision_result.get("success") if vision_result else False,
            },
        }

    return response


def run_parallel_pipeline(image: np.ndarray, file_bytes: bytes, filename: str, content_type: str, mode: str, started_at: float) -> dict:
    vision_timeout = int(os.getenv("VISION_LLM_TIMEOUT_SECONDS", "45"))
    vision_strategy = os.getenv("VISION_EXTRACTION_STRATEGY", "structured").strip().lower()
    structured_fallback = os.getenv("VISION_STRUCTURED_FALLBACK_TWO_STAGE", "true").lower() != "false"
    default_branch_timeout = vision_timeout * (2 if vision_strategy == "two_stage" or (vision_strategy == "structured" and structured_fallback) else 1) + 30
    timeout = int(os.getenv("VISION_BRANCH_TIMEOUT_SECONDS", str(default_branch_timeout)))
    response_budget = int(os.getenv("OCR_RESPONSE_BUDGET_SECONDS", "120"))
    if mode != "parallel_benchmark":
        timeout = min(timeout, response_budget)
    include_candidates = mode == "parallel_benchmark" or os.getenv("OCR_INCLUDE_CANDIDATES", "false").lower() == "true"
    executor = ThreadPoolExecutor(max_workers=1)
    vision_future = executor.submit(run_vision_pipeline, file_bytes, content_type)
    results: dict[str, dict | None] = {"traditional": None, "vision": None}
    deadline = time.time() + timeout

    def timeout_vision_result() -> dict:
        return {
            "success": False,
            "message": f"Vision branch timed out after {timeout}s",
            "data": {"medications": [], "confidence": "low", "error": f"Vision branch timed out after {timeout}s"},
            "timing": {"vision_llm_seconds": timeout, "visionTimedOut": True},
        }

    def candidate_quality(source: str, result: dict) -> dict:
        data = result.get("data") or {}
        raw_text = result.get("rawText", "") if source == "traditional" else ""
        has_raw_text = bool(raw_text) if source == "traditional" else False
        return score_candidate(data, source, has_raw_text, raw_text)

    def candidate_can_return(source: str, result: dict) -> bool:
        quality = candidate_quality(source, result)
        critical_flags = quality.get("criticalFlags") or []
        can_return = bool(quality.get("canEarlyReturn"))
        if source == "traditional" and not can_return:
            can_return = bool(
                quality.get("score", 0) >= int(os.getenv("TRADITIONAL_EARLY_RETURN_MIN_SCORE", "88"))
                and quality.get("medicationCount", 0) >= int(os.getenv("TRADITIONAL_EARLY_RETURN_MIN_MEDICATIONS", "2"))
                and quality.get("medicationNameRatio", 0) >= 0.8
                and quality.get("medicationQuantityUnitRatio", 0) >= 0.7
                and quality.get("medicationUnitRatio", 0) >= 0.5
                and not critical_flags
            )
        if not can_return:
            print(
                f"[Pipeline] {source} candidate not enough for early return: "
                f"score={quality.get('score')} flags={quality.get('flags')} "
                f"meds={quality.get('medicationCount')}"
            )
        return can_return

    def best_available_response(early_returned: bool = False, source: str | None = None) -> dict:
        response = build_ocr_response(results["traditional"], results["vision"], mode, include_candidates, started_at)
        response["timing"]["responseBudgetSeconds"] = response_budget
        response["timing"]["visionBranchTimeoutSeconds"] = timeout
        if early_returned:
            response["timing"]["earlyReturned"] = True
            response["timing"]["earlyReturnSource"] = source
        return response

    try:
        # PaddleOCR/VietOCR can fail intermittently when executed from a worker thread
        # inside the Docker runtime, so keep the traditional branch on the request thread.
        results["traditional"] = run_traditional_pipeline(
            image,
            filename,
            allow_llm_fallback=(mode == "parallel_benchmark" or os.getenv("PARALLEL_TRADITIONAL_LLM_FALLBACK", "false").lower() == "true"),
        )

        if mode == "parallel_benchmark":
            try:
                results["vision"] = vision_future.result(timeout=max(0.1, deadline - time.time()))
            except FuturesTimeoutError:
                results["vision"] = timeout_vision_result()
            return best_available_response()

        if vision_future.done():
            results["vision"] = vision_future.result()
            if candidate_can_return("vision", results["vision"]):
                executor.shutdown(wait=False, cancel_futures=True)
                return best_available_response(True, "vision")

        if candidate_can_return("traditional", results["traditional"]):
            executor.shutdown(wait=False, cancel_futures=True)
            return best_available_response(True, "traditional")

        pending = set() if results["vision"] else {vision_future}
        while pending:
            remaining = max(0.1, deadline - time.time())
            done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            if not done:
                results["vision"] = timeout_vision_result()
                pending.clear()
                continue

            for future in done:
                results["vision"] = future.result()
                if candidate_can_return("vision", results["vision"]):
                    executor.shutdown(wait=False, cancel_futures=True)
                    return best_available_response(True, "vision")

        return best_available_response()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


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
async def extract_prescription(file: UploadFile = File(...), mode: str = Form("traditional")):
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
        image_quality = analyze_image_quality(image)

        mode = (mode or "traditional").strip().lower()
        allowed_modes = {"traditional", "vision", "parallel", "parallel_benchmark"}
        if mode not in allowed_modes:
            raise HTTPException(status_code=400, detail=f"Unsupported OCR mode: {mode}")

        started_at = time.time()
        content_type = file.content_type or "image/jpeg"
        filename = file.filename or "prescription"

        print(f"\n{'='*50}")
        print(f"[Pipeline] Start: {filename} ({image.shape}) mode={mode}")

        if mode == "traditional":
            traditional_result = run_traditional_pipeline(image, filename)
            response = build_ocr_response(traditional_result, None, mode, False, started_at)
        elif mode == "vision":
            vision_result = await run_in_threadpool(run_vision_pipeline, file_bytes, content_type)
            response = build_ocr_response(None, vision_result, mode, True, started_at)
        else:
            response = run_parallel_pipeline(image, file_bytes, filename, content_type, mode, started_at)

        response = attach_image_quality(response, image_quality)

        try:
            save_debug_log(
                filename=filename,
                raw_text=response.get("rawText", ""),
                extracted_data=response.get("data") or {},
                timing=response.get("timing") or {},
                image_shape=image.shape,
                num_boxes=response.get("timing", {}).get("traditional_numBoxes", 0),
            )
        except Exception as log_err:
            print(f"[Pipeline] Debug log error (non-fatal): {log_err}")

        return response

        print(f"\n{'='*50}")
        print(f"[Pipeline] Bắt đầu: {file.filename} ({image.shape})")

        # Trạm 1: PaddleOCR
        print("[Pipeline] Trạm 1: PaddleOCR phát hiện vùng chữ...")
        t1_start = time.time()
        boxes, ocr_image = detect_text_regions_with_image(image)
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
        raw_text = extract_full_text(ocr_image, boxes)
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
        boxes, ocr_image = detect_text_regions_with_image(image)
        t1_end = time.time()
        
        t2_start = time.time()
        raw_text = extract_full_text(ocr_image, boxes)
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
