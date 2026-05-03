"""
src/services/extractor.py
Trạm 3: Structured Information Extraction bằng Google Gemini Flash.
Nhiệm vụ: Chuyển raw text → JSON chuẩn hóa gồm thông tin đơn thuốc.
"""
import os
import time
import json
import requests
from dotenv import load_dotenv

from src.services.regex_parser import parse_with_regex

load_dotenv()

# Khởi tạo Gemini client (lazy load)
_gemini_client = None


def get_gemini_client():
    """Lazy initialization Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        try:
            from google import genai
        except ImportError:
            raise ValueError("[Extractor] Gói google-genai chưa được cài đặt. Hãy pip install google-genai để dùng backend gemini.")
            
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError(
                "[Extractor] GEMINI_API_KEY chưa được cấu hình! "
                "Vui lòng tạo file .env và điền API Key từ aistudio.google.com"
            )
        _gemini_client = genai.Client(api_key=api_key)
        print("[Extractor] Gemini client đã sẵn sàng!")
    return _gemini_client


# Prompt mẫu - rõ ràng, cụ thể để Gemini trả JSON đúng format
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
    Trích xuất thông tin có cấu trúc - Hybrid Strategy: Regex trước, LLM bổ sung khi cần.
    """
    if not raw_text or not raw_text.strip():
        return _empty_result("Không có text để xử lý")
        
    backend = os.getenv("EXTRACTOR_BACKEND", "ollama").lower()
    
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
    print(f"[Extractor] Bước 3B: Gọi LLM ({backend}) bổ sung...")
    
    # === Dynamic VRAM Swap: CPU ===
    if backend == "ollama":
        try:
            from src.services.recognizer import move_vietocr_to_cpu
            move_vietocr_to_cpu()
        except ImportError:
            pass
            
    if backend == "ollama":
        llm_result = _extract_with_ollama(raw_text)
    elif backend == "gemini":
        llm_result = _extract_with_gemini(raw_text)
    else:
        print(f"[Extractor] Lỗi: Backend không hợp lệ ({backend}). Hỗ trợ: ollama, gemini")
        llm_result = _extract_with_ollama(raw_text) # Fallback to ollama
        
    # === Dynamic VRAM Swap: GPU ===
    if backend == "ollama":
        try:
            from src.services.recognizer import move_vietocr_to_gpu
            move_vietocr_to_gpu()
        except ImportError:
            pass
            
    final_result = _merge_results(regex_result, llm_result)
    final_result["_extraction_method"] = "regex_plus_llm"
    return final_result

def _merge_results(regex_result: dict, llm_result: dict) -> dict:
    """Merge regex + LLM: Ưu tiên regex cho thông tin chung, LLM cho thuốc."""
    merged = {}
    for key in ["patientName", "patientAge", "patientGender", "phoneNumber",
                "doctorName", "hospitalName", "prescriptionDate",
                "diagnosis", "specialNotes"]:
        merged[key] = regex_result.get(key) or llm_result.get(key)
        
    merged["medications"] = llm_result.get("medications") or regex_result.get("medications", [])
    merged["confidence"] = llm_result.get("confidence", "medium")
    
    return merged


def _extract_with_ollama(raw_text: str) -> dict:
    """Gọi Local LLM qua Ollama REST API để trích xuất JSON."""
    prompt = EXTRACTION_PROMPT.format(raw_text=raw_text)
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.0,
            "num_ctx": 2048
        }
    }
    
    print(f"[Extractor] Đang gọi Ollama ({model_name}) để trích xuất JSON...")
    
    try:
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=180)
        response.raise_for_status()
        
        result_json = response.json()
        response_text = result_json.get("response", "").strip()
        
        result = json.loads(response_text)
        print(f"[Extractor] Trích xuất thành công (Ollama)! Confidence: {result.get('confidence', 'unknown')}")
        return result
        
    except requests.exceptions.ConnectionError:
        error_msg = f"Không thể kết nối Ollama tại {base_url}. Vui lòng đảm bảo bạn đã cài và chạy Ollama."
        print(f"[Extractor] Lỗi: {error_msg}")
        return _empty_result(error_msg)
    except json.JSONDecodeError as e:
        print(f"[Extractor] Lỗi parse JSON từ Ollama: {e}")
        return _empty_result(f"Lỗi parse JSON (Ollama): {str(e)}")
    except Exception as e:
        print(f"[Extractor] Lỗi khi gọi Ollama API: {e}")
        return _empty_result(f"Lỗi hệ thống Ollama: {str(e)}")


def _extract_with_gemini(raw_text: str) -> dict:
    """Gọi Gemini Cloud API qua google.genai để trích xuất JSON."""
    client = get_gemini_client()
    prompt = EXTRACTION_PROMPT.format(raw_text=raw_text)

    try:
        from google.genai import types
        print("[Extractor] Đang gọi Cloud Gemini Flash để trích xuất thông tin...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2048,
                response_mime_type="application/json",
            )
        )

        response_text = response.text.strip()
        result = json.loads(response_text)
        print(f"[Extractor] Trích xuất thành công (Gemini)! Confidence: {result.get('confidence', 'unknown')}")
        return result

    except json.JSONDecodeError as e:
        print(f"[Extractor] Lỗi parse JSON từ Gemini: {e}")
        raw_preview = response_text[:200] if 'response_text' in locals() and response_text else "(empty)"
        print(f"[Extractor] Raw response: {raw_preview}")
        return _empty_result(f"Lỗi parse JSON (Gemini): {str(e)}")

    except Exception as e:
        print(f"[Extractor] Lỗi khi gọi Gemini API: {e}")
        return _empty_result(f"Lỗi hệ thống Gemini: {str(e)}")


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
