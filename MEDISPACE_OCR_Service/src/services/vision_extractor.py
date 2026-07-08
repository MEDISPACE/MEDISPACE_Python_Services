"""Vision LLM extraction for prescription images."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Any, Dict

import cv2
import numpy as np
import requests

from src.services.quality import empty_prescription_data, normalize_prescription_data, score_candidate


VISION_PROMPT = """You are extracting structured data from a Vietnamese medical prescription image.
The image may be handwritten, tilted, noisy, or partially cropped. Read it visually like a pharmacist reviewing a prescription.
Return only valid JSON. Do not include markdown or explanations.

Expected schema:
{
  "patientName": string|null,
  "patientAge": string|null,
  "patientGender": "male"|"female"|"other"|null,
  "phoneNumber": string|null,
  "doctorName": string|null,
  "hospitalName": string|null,
  "prescriptionDate": "YYYY-MM-DD"|null,
  "diagnosis": string|null,
  "medications": [
    {
      "productName": string,
      "activeIngredient": string|null,
      "dosage": string|null,
      "quantity": number|null,
      "unit": string|null,
      "instructions": string|null,
      "confidence": "high"|"medium"|"low"
    }
  ],
  "specialNotes": string|null,
  "confidence": "high"|"medium"|"low"
}

Rules:
- Focus on the actual medicine rows/numbered lines, especially sections like "CAP TOA", "TOA THUOC", "DON THUOC".
- Preserve the visible handwritten brand/name and strength exactly as best as possible, for example "Sucralin 450mg", "Methicone 10mg", "Esonic 20mg".
- Do not replace an unclear handwritten name with a common product unless the writing strongly supports it.
- If uncertain between similar readings, choose the best visible reading, keep confidence low/medium, and mention uncertainty in instructions if useful.
- For Vietnamese prescription templates, phrases like "Ngay uong 2 lan, moi lan 1 vien" are instructions, not medicine names.
- If a medicine row has no visible quantity but shows frequency/dose, keep quantity null and put the visible usage in instructions.
- Use null for missing values, never empty strings for unknown fields.
- If a value is uncertain, include the best reading and set confidence to low or medium.
- Do not invent medicines that are not visible in the image.
"""

VISION_STRUCTURED_PROMPT = """You are a Vision OCR system for Vietnamese medical prescriptions.
Read the image carefully like a pharmacist, including handwritten medicine rows, printed templates, dosage notes, and quantities.
Think through the image internally, but return only one valid JSON object. Do not include markdown, prose, medical advice, or explanations.

Expected schema:
{
  "patientName": string|null,
  "patientAge": string|null,
  "patientGender": "male"|"female"|"other"|null,
  "phoneNumber": string|null,
  "doctorName": string|null,
  "hospitalName": string|null,
  "prescriptionDate": "YYYY-MM-DD"|null,
  "diagnosis": string|null,
  "medications": [
    {
      "productName": string,
      "activeIngredient": string|null,
      "dosage": string|null,
      "quantity": number|null,
      "unit": string|null,
      "instructions": string|null,
      "confidence": "high"|"medium"|"low",
      "needsReview": boolean,
      "reviewReason": string|null
    }
  ],
  "specialNotes": string|null,
  "confidence": "high"|"medium"|"low"
}

Rules:
- Preserve visible brand/product names from the image as written. Do not replace a handwritten brand with a generic active ingredient unless the image explicitly shows it.
- Separate medicine identity from usage instructions. Phrases like "ngay uong 2 lan, moi lan 1 vien" are instructions, not product names.
- If a value is unclear, provide the best visible reading, set confidence to low or medium, and set needsReview=true with a short reviewReason.
- If total quantity is not visible, use null. Do not treat dose frequency as total quantity.
- Do not invent medicines, strengths, quantities, diagnoses, doctors, or dates that are not visible.
- Use null for unknown values, never an empty string for unknown fields.
- Return only strict JSON that matches the schema.
"""

VISION_FREEFORM_PROMPT = """Bạn là trợ lý nhà thuốc AI đang đọc ảnh đơn thuốc Việt Nam.
Ảnh có thể là chữ viết tay, nghiêng, nhiễu hoặc bị chụp xa. Hãy đọc trực tiếp từ ảnh giống cách dược sĩ đọc đơn.

Nhiệm vụ:
- Xác định đây có phải đơn thuốc/toa thuốc/phiếu khám có danh sách thuốc hay không.
- Liệt kê các thuốc nhìn thấy theo từng dòng, ưu tiên tên thuốc/brand, hàm lượng, số lượng và cách dùng.
- Giữ nguyên cách đọc tên thuốc nhìn thấy trên ảnh; nếu không chắc, ghi rõ "có thể là ...".
- Không tự thay thuốc bằng tên phổ biến nếu nét chữ không ủng hộ rõ.
- Các cụm như "ngày uống 2 lần, mỗi lần 1 viên" là cách dùng, không phải tên thuốc.
- Trả lời ngắn gọn bằng tiếng Việt, chỉ tập trung phần thông tin đọc được từ ảnh. Không giải thích công dụng thuốc.
"""

VISION_JSON_FROM_READING_PROMPT = """Bạn nhận được phần đọc thô từ ảnh đơn thuốc Việt Nam. Hãy chuẩn hóa thành JSON hợp lệ, không markdown, không giải thích.

Schema bắt buộc:
{{
  "patientName": string|null,
  "patientAge": string|null,
  "patientGender": "male"|"female"|"other"|null,
  "phoneNumber": string|null,
  "doctorName": string|null,
  "hospitalName": string|null,
  "prescriptionDate": "YYYY-MM-DD"|null,
  "diagnosis": string|null,
  "medications": [
    {{
      "productName": string,
      "activeIngredient": string|null,
      "dosage": string|null,
      "quantity": number|null,
      "unit": string|null,
      "instructions": string|null,
      "confidence": "high"|"medium"|"low"
    }}
  ],
  "specialNotes": string|null,
  "confidence": "high"|"medium"|"low"
}}

Quy tắc:
- Chỉ dùng thông tin có trong phần đọc thô.
- Nếu tên thuốc có chữ "có thể là" hoặc không chắc, vẫn điền best reading vào productName và đặt confidence low/medium.
- Nếu không thấy tổng số lượng thuốc, để quantity null; đừng lấy số lần uống làm quantity.
- Nếu chỉ thấy "2 lần, mỗi lần 1 viên", đưa vào instructions.
- Unknown dùng null, không dùng chuỗi rỗng.

Phần đọc thô:
{freeform_reading}
"""


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def prepare_image_for_vision(file_bytes: bytes, mime_type: str) -> tuple[bytes, str, Dict[str, Any]]:
    """Prepare an image for the Vision LLM.

    Chat AI sends the original bytes directly, which preserves handwriting strokes.
    Keep that behavior for normal prescription uploads and only downscale very large
    files to stay within the LLM server payload budget.
    """
    max_side = max(256, _env_int("VISION_IMAGE_MAX_SIDE", 2560))
    jpeg_quality = min(95, max(50, _env_int("VISION_IMAGE_JPEG_QUALITY", 92)))
    send_original_under_bytes = max(0, _env_int("VISION_SEND_ORIGINAL_UNDER_BYTES", 8 * 1024 * 1024))
    meta: Dict[str, Any] = {
        "originalBytes": len(file_bytes),
        "originalMimeType": mime_type,
        "maxSide": max_side,
        "jpegQuality": jpeg_quality,
        "sendOriginalUnderBytes": send_original_under_bytes,
    }

    if send_original_under_bytes and len(file_bytes) <= send_original_under_bytes and mime_type in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
        meta["preparedBytes"] = len(file_bytes)
        meta["preparedMimeType"] = mime_type
        meta["preprocessSkipped"] = "original_under_size_limit"
        return file_bytes, mime_type, meta

    try:
        np_buffer = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        if image is None:
            meta["preprocessSkipped"] = "decode_failed"
            return file_bytes, mime_type, meta

        height, width = image.shape[:2]
        meta["originalWidth"] = width
        meta["originalHeight"] = height
        scale = min(1.0, max_side / float(max(height, width)))
        if scale < 1.0:
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            new_width = width
            new_height = height

        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            meta["preprocessSkipped"] = "encode_failed"
            return file_bytes, mime_type, meta

        prepared = encoded.tobytes()
        meta["resizedWidth"] = new_width
        meta["resizedHeight"] = new_height
        meta["preparedBytes"] = len(prepared)
        meta["preparedMimeType"] = "image/jpeg"
        return prepared, "image/jpeg", meta
    except Exception as exc:
        meta["preprocessSkipped"] = f"failed: {exc}"
        return file_bytes, mime_type, meta


def _extract_json_object(response_text: str) -> Dict[str, Any]:
    text = (response_text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))

def _extract_medications_from_freeform(reading: str) -> Dict[str, Any]:
    text = reading or ""
    medications = []
    seen = set()
    stop_prefixes = (
        "luu y", "lưu ý", "tom tat", "tóm tắt", "khuyen cao", "khuyến cáo",
        "lieu luong", "liều lượng", "cach dung", "cách dùng", "loai tai lieu",
        "loại tài liệu", "mo ta", "mô tả", "dua tren", "dựa trên", "don thuoc",
        "đơn thuốc", "cac thuoc", "các thuốc", "san pham", "sản phẩm",
    )

    def clean_name(value: str) -> str:
        value = re.sub(r"^[\s\-*•]+", "", value.strip())
        value = re.sub(r"^\d+[\.)]\s*", "", value)
        value = value.replace("**", "").replace("__", "").strip(" :-–—\t")
        if ":" in value:
            value = value.split(":", 1)[0].strip(" :-–—\t")
        value = re.sub(r"^c[oó]\s+th[eể]\s+l[aà]\s+", "", value, flags=re.IGNORECASE)
        value = re.split(r"\s+\(.*", value, maxsplit=1)[0].strip(" :-–—\t") if "(" in value else value
        return value.strip()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = re.sub(r"^\d+[\.)]\s*", "", re.sub(r"\s+", " ", line.lower())).strip(" :-–—*")
        if any(normalized.startswith(prefix) for prefix in stop_prefixes):
            continue

        candidate = ""
        bullet_match = re.match(r"^(?:[-*•]|\d+[\.)])\s*(.+)$", line)
        if bullet_match:
            candidate = bullet_match.group(1)
        elif ":" in line and len(line.split(":", 1)[0].split()) <= 5:
            candidate = line.split(":", 1)[0]

        name = clean_name(candidate)
        if not name or len(name) < 3:
            continue
        if len(name.split()) > 6:
            continue
        key = re.sub(r"\W+", "", name.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        medications.append({
            "productName": name,
            "activeIngredient": None,
            "dosage": None,
            "quantity": None,
            "unit": None,
            "instructions": None,
            "confidence": "low",
            "needsReview": True,
            "reviewReason": "parsed_from_vision_freeform",
        })

    return {
        "patientName": None,
        "patientAge": None,
        "patientGender": None,
        "phoneNumber": None,
        "doctorName": None,
        "hospitalName": None,
        "prescriptionDate": None,
        "diagnosis": None,
        "medications": medications,
        "specialNotes": None,
        "confidence": "low" if medications else "medium",
        "_extraction_method": "vision_freeform_fallback",
    }

def _post_llm(endpoint: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int, retries: int, retry_backoff: float, label: str) -> str:
    last_error = f"{label} failed"
    response: requests.Response | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError(f"{label} response has no choices")
            return choices[0].get("message", {}).get("content", "").strip()
        except requests.exceptions.Timeout:
            last_error = f"{label} timeout after {timeout}s"
        except requests.exceptions.ConnectionError as exc:
            last_error = f"Cannot connect to {label}: {exc}"
        except requests.exceptions.HTTPError as exc:
            body = exc.response.text[:300] if exc.response is not None else ""
            raise RuntimeError(f"{label} HTTP error: {exc} {body}") from exc
        except requests.exceptions.RequestException as exc:
            last_error = f"{label} request failed: {exc}"

        print(f"[VisionExtractor] Attempt {attempt}/{retries} failed: {last_error}")
        if attempt < retries:
            time.sleep(retry_backoff * attempt)

    raise TimeoutError(last_error)

def extract_prescription_from_image(file_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
    base_url = os.getenv("CUSTOM_LLM_BASE_URL", "https://llm.datateam.space").rstrip("/")
    model_name = os.getenv("VISION_LLM_MODEL") or os.getenv("CUSTOM_LLM_MODEL", "gemma-4-e4b-it.gguf")
    api_key = os.getenv("CUSTOM_LLM_API_KEY", "")
    timeout = _env_int("VISION_LLM_TIMEOUT_SECONDS", 45)
    retries = max(1, _env_int("VISION_LLM_RETRIES", 2))
    json_timeout = max(5, _env_int("VISION_JSON_NORMALIZE_TIMEOUT_SECONDS", min(timeout, 25)))
    json_retries = max(1, _env_int("VISION_JSON_NORMALIZE_RETRIES", 1))
    retry_backoff = max(0.0, _env_float("VISION_LLM_RETRY_BACKOFF_SECONDS", 1.5))
    strategy = os.getenv("VISION_EXTRACTION_STRATEGY", "structured").strip().lower()
    if strategy not in {"direct", "structured", "two_stage"}:
        strategy = "structured"
    allow_structured_fallback = os.getenv("VISION_STRUCTURED_FALLBACK_TWO_STAGE", "true").lower() == "true"

    prepared_bytes, prepared_mime_type, image_meta = prepare_image_for_vision(file_bytes, mime_type)
    image_b64 = base64.b64encode(prepared_bytes).decode("ascii")
    image_url = f"data:{prepared_mime_type};base64,{image_b64}"
    endpoint = f"{base_url}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    direct_payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "Return strict JSON for prescription extraction. No prose.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": VISION_PROMPT},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": _env_int("VISION_LLM_MAX_TOKENS", 4096),
        "stream": False,
        "response_format": {"type": "json_object"},
    }

    structured_payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful prescription OCR engine. Return strict JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": VISION_STRUCTURED_PROMPT},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": _env_int("VISION_LLM_MAX_TOKENS", 4096),
        "stream": False,
        "response_format": {"type": "json_object"},
    }

    freeform_payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "Bạn là Trợ lý Nhà thuốc AI của Medispace. Hãy đọc ảnh đơn thuốc cẩn trọng, không bịa thông tin.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": VISION_FREEFORM_PROMPT},
                ],
            },
        ],
        "temperature": _env_float("VISION_LLM_FREEFORM_TEMPERATURE", 0.30),
        "max_tokens": _env_int("VISION_LLM_FREEFORM_MAX_TOKENS", 4096),
        "stream": False,
    }

    print(
        f"[VisionExtractor] Calling Vision LLM ({model_name}) at {endpoint} "
        f"image={image_meta.get('originalBytes')}B->{image_meta.get('preparedBytes', image_meta.get('originalBytes'))}B "
        f"strategy={strategy} timeout={timeout}s retries={retries}..."
    )

    timing: Dict[str, Any] = {"strategy": strategy}

    def run_two_stage() -> tuple[Dict[str, Any], str | None]:
        stage1_start = time.time()
        reading = _post_llm(endpoint, freeform_payload, headers, timeout, retries, retry_backoff, "Vision LLM freeform read")
        timing["vision_freeform_seconds"] = round(time.time() - stage1_start, 2)
        json_payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "Return strict JSON only. No prose."},
                {"role": "user", "content": VISION_JSON_FROM_READING_PROMPT.format(freeform_reading=reading)},
            ],
            "temperature": 0.0,
            "max_tokens": _env_int("VISION_LLM_MAX_TOKENS", 4096),
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        stage2_start = time.time()
        try:
            text = _post_llm(endpoint, json_payload, headers, json_timeout, json_retries, retry_backoff, "Vision LLM JSON normalize")
            timing["vision_json_seconds"] = round(time.time() - stage2_start, 2)
            return _extract_json_object(text), reading
        except (json.JSONDecodeError, RuntimeError, TimeoutError) as exc:
            timing["vision_json_seconds"] = round(time.time() - stage2_start, 2)
            timing["visionJsonFallbackReason"] = str(exc)
            fallback = _extract_medications_from_freeform(reading)
            fallback["_vision_json_error"] = f"Vision JSON normalize failed; used freeform fallback: {exc}"
            return fallback, reading

    try:
        if strategy == "direct":
            direct_start = time.time()
            response_text = _post_llm(endpoint, direct_payload, headers, timeout, retries, retry_backoff, "Vision LLM direct JSON")
            timing["vision_direct_seconds"] = round(time.time() - direct_start, 2)
            result = _extract_json_object(response_text)
            freeform_reading = None
        elif strategy == "structured":
            structured_start = time.time()
            response_text = _post_llm(endpoint, structured_payload, headers, timeout, retries, retry_backoff, "Vision LLM structured JSON")
            timing["vision_structured_seconds"] = round(time.time() - structured_start, 2)
            result = _extract_json_object(response_text)
            result = normalize_prescription_data(result, "vision")
            quality = score_candidate(result, "vision", False)
            freeform_reading = None
            if allow_structured_fallback and not quality.get("canEarlyReturn"):
                print(f"[VisionExtractor] Structured quality low ({quality.get('score')}); falling back to two_stage...")
                timing["structuredFallbackTwoStage"] = True
                result, freeform_reading = run_two_stage()
                strategy = "two_stage_fallback"
        else:
            result, freeform_reading = run_two_stage()

        result = normalize_prescription_data(result, "vision")
        result["_extraction_method"] = "vision_llm"
        result["_vision_strategy"] = strategy
        result["_vision_image"] = image_meta
        result["_vision_timing"] = timing
        if freeform_reading:
            result["_vision_freeform_reading"] = freeform_reading[:4000]
        print(f"[VisionExtractor] Vision LLM succeeded. Medications: {len(result.get('medications', []))}")
        return result
    except json.JSONDecodeError as exc:
        return empty_prescription_data(f"Vision LLM returned invalid JSON: {exc}")
    except (RuntimeError, TimeoutError) as exc:
        return empty_prescription_data(str(exc))
    except Exception as exc:
        return empty_prescription_data(f"Vision LLM failed: {exc}")
