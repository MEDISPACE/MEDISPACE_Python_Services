"""
src/services/extractor.py
Trạm 3: Structured Information Extraction.
Nhiệm vụ: Chuyển raw text → JSON chuẩn hóa gồm thông tin đơn thuốc.

Pipeline (Hybrid):
  Bước 3A: Regex Parser (nhanh, không cần network)
  Bước 3B: Custom LLM API (llama.cpp / OpenAI-compatible) — chỉ gọi khi Regex thiếu dữ liệu

Cấu hình qua .env:
  CUSTOM_LLM_BASE_URL  - ví dụ: https://llm.datateam.space
  CUSTOM_LLM_MODEL     - ví dụ: gemma-4-e4b-it.gguf
  CUSTOM_LLM_API_KEY   - Bearer token nếu cần (có thể để trống)
  REGEX_MIN_SCORE      - Số trường tối thiểu Regex phải tìm được để skip LLM (mặc định: 5)
  REGEX_REQUIRE_MEDICATIONS - Bắt buộc phải có medications[] để skip LLM (mặc định: true)
"""
import os
import time
import json
import requests
from dotenv import load_dotenv

from src.services.regex_parser import parse_with_regex

load_dotenv()


# Prompt — rõ ràng, cụ thể để LLM trả JSON đúng format
EXTRACTION_PROMPT = """Bạn là chuyên gia y tế, hãy phân tích đoạn văn bản sau được trích xuất từ một đơn thuốc y tế Việt Nam và trả về thông tin dưới dạng JSON hợp lệ.

Nội dung đơn thuốc:
---
{raw_text}
---

Hãy trả về JSON với đúng cấu trúc sau (không thêm bất kỳ text nào khác ngoài JSON):
{{
  "patientName": "tên bệnh nhân hoặc null",
  "patientAge": "tuổi bệnh nhân hoặc null",
  "patientGender": "Nam/Nữ hoặc null",
  "phoneNumber": "số điện thoại liên hệ (10 số) hoặc null",
  "doctorName": "tên bác sĩ hoặc null",
  "hospitalName": "tên bệnh viện/phòng khám hoặc null",
  "prescriptionDate": "ngày kê đơn dạng YYYY-MM-DD hoặc null",
  "diagnosis": "chẩn đoán bệnh hoặc null",
  "medications": [
    {{
      "productName": "tên thuốc",
      "dosage": "liều dùng (ví dụ: Sáng 1 viên, Tối 1 viên sau ăn)",
      "quantity": số lượng dạng số nguyên hoặc null,
      "unit": "đơn vị (viên/gói/chai...) hoặc null",
      "instructions": "hướng dẫn đặc biệt hoặc null"
    }}
  ],
  "specialNotes": "ghi chú đặc biệt hoặc null",
  "confidence": "high/medium/low - mức độ tự tin vào kết quả trích xuất"
}}

Lưu ý quan trọng:
- Nếu không tìm thấy thông tin, điền null (không phải chuỗi rỗng "")
- medications phải là mảng, ngay cả khi chỉ có 1 loại thuốc
- Nếu không thể đọc rõ tên thuốc, hãy ghi lại những gì bạn đọc được dù không chắc chắn
- confidence: "high" nếu thông tin rõ ràng, "medium" nếu còn chỗ mờ nhạt, "low" nếu chữ viết tay khó đọc"""


def extract_prescription_info(raw_text: str) -> dict:
    """
    Trích xuất thông tin có cấu trúc — Hybrid Strategy:
      3A: Regex trước (nhanh, offline)
      3B: Custom LLM API bổ sung khi Regex thiếu dữ liệu
    """
    if not raw_text or not raw_text.strip():
        return _empty_result("Không có text để xử lý")

    # === Bước 3A: Regex Parser ===
    print("[Extractor] Bước 3A: Regex Parser đang trích xuất...")
    t_regex = time.time()
    regex_result = parse_with_regex(raw_text)
    regex_time = time.time() - t_regex
    regex_score = regex_result.pop("_regex_score", 0)

    print(f"[Extractor] Regex xong trong {regex_time:.3f}s (score: {regex_score}/10)")

    min_score = int(os.getenv("REGEX_MIN_SCORE", "5"))
    require_meds = os.getenv("REGEX_REQUIRE_MEDICATIONS", "true").lower() == "true"

    has_medications = len(regex_result.get("medications", [])) > 0
    has_key_fields = regex_score >= min_score

    if (not require_meds or has_medications) and has_key_fields:
        regex_result["confidence"] = "high" if regex_score >= 7 else "medium"
        regex_result["_extraction_method"] = "regex_only"
        print(f"[Extractor] ✅ Regex đủ dữ liệu! Skip LLM. (score: {regex_score}/10)")
        return regex_result

    print(f"[Extractor] ⚠ Regex thiếu (medications={has_medications}, score={regex_score}/10)")
    print("[Extractor] Bước 3B: Gọi Custom LLM API bổ sung...")

    # === Bước 3B: Custom LLM API ===
    llm_result = _extract_with_custom_api(raw_text)

    final_result = _merge_results(regex_result, llm_result)
    final_result["_extraction_method"] = "regex_plus_llm"
    return final_result


def _merge_results(regex_result: dict, llm_result: dict) -> dict:
    """Merge Regex + LLM: Ưu tiên Regex cho thông tin chung, LLM cho medications."""
    merged = {}
    for key in ["patientName", "patientAge", "patientGender", "phoneNumber",
                "doctorName", "hospitalName", "prescriptionDate",
                "diagnosis", "specialNotes"]:
        merged[key] = regex_result.get(key) or llm_result.get(key)

    merged["medications"] = llm_result.get("medications") or regex_result.get("medications", [])
    merged["confidence"] = llm_result.get("confidence", "medium")

    return merged


def _extract_with_custom_api(raw_text: str) -> dict:
    """
    Gọi llama.cpp HTTP server (hoặc bất kỳ OpenAI-compatible server nào)
    qua POST /v1/chat/completions.
    """
    base_url = os.getenv("CUSTOM_LLM_BASE_URL", "https://llm.datateam.space").rstrip("/")
    model_name = os.getenv("CUSTOM_LLM_MODEL", "gemma-4-e4b-it.gguf")
    api_key = os.getenv("CUSTOM_LLM_API_KEY", "")

    prompt = EXTRACTION_PROMPT.format(raw_text=raw_text)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    endpoint = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "Bạn là chuyên gia y tế. Hãy luôn trả lời bằng JSON hợp lệ, không thêm text nào khác."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
        "stream": False,
        "response_format": {"type": "json_object"}
    }

    print(f"[Extractor] Đang gọi Custom LLM ({model_name}) tại {endpoint}...")

    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=180)
        response.raise_for_status()

        response_json = response.json()

        # Parse OpenAI-compatible response format
        choices = response_json.get("choices", [])
        if not choices:
            raise ValueError("Response không có 'choices'")

        message = choices[0].get("message", {})
        response_text = message.get("content", "").strip()

        if not response_text:
            raise ValueError("Response content rỗng")

        # Loại bỏ markdown code block nếu model trả về
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        result = json.loads(response_text)
        print(f"[Extractor] ✅ Custom LLM thành công! Confidence: {result.get('confidence', 'unknown')}")
        return result

    except requests.exceptions.ConnectionError:
        error_msg = f"Không thể kết nối Custom LLM API tại {base_url}."
        print(f"[Extractor] ❌ {error_msg}")
        return _empty_result(error_msg)
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error từ Custom LLM API: {e.response.status_code} - {e.response.text[:200]}"
        print(f"[Extractor] ❌ {error_msg}")
        return _empty_result(error_msg)
    except json.JSONDecodeError as e:
        print(f"[Extractor] ❌ Lỗi parse JSON từ Custom LLM: {e}")
        raw_preview = response_text[:300] if 'response_text' in locals() and response_text else "(empty)"
        print(f"[Extractor] Raw response: {raw_preview}")
        return _empty_result(f"Lỗi parse JSON (Custom LLM): {str(e)}")
    except Exception as e:
        print(f"[Extractor] ❌ Lỗi khi gọi Custom LLM API: {e}")
        return _empty_result(f"Lỗi hệ thống Custom LLM: {str(e)}")


def _empty_result(error_msg: str) -> dict:
    """Trả về cấu trúc rỗng khi có lỗi."""
    return {
        "patientName": None,
        "patientAge": None,
        "patientGender": None,
        "phoneNumber": None,
        "doctorName": None,
        "hospitalName": None,
        "prescriptionDate": None,
        "diagnosis": None,
        "medications": [],
        "specialNotes": None,
        "confidence": "low",
        "error": error_msg
    }
