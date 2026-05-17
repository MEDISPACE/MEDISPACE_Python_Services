"""
scripts/test_llm_api.py
Test nhanh Custom LLM API tại https://llm.datateam.space/

Chạy từ thư mục gốc MEDISPACE_OCR_Service:
  python scripts/test_llm_api.py
"""
import os
import sys
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL  = os.getenv("CUSTOM_LLM_BASE_URL", "https://llm.datateam.space").rstrip("/")
MODEL     = os.getenv("CUSTOM_LLM_MODEL",    "gemma-4-e4b-it.gguf")
API_KEY   = os.getenv("CUSTOM_LLM_API_KEY",  "")

SEP = "=" * 60

def _headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


# ──────────────────────────────────────────────────────────────
# Test 1: GET /v1/models — kiểm tra server có online không
# ──────────────────────────────────────────────────────────────
def test_models_endpoint():
    print(f"\n{SEP}")
    print("TEST 1: GET /v1/models — Kiểm tra server online")
    print(SEP)
    url = f"{BASE_URL}/v1/models"
    try:
        t0 = time.time()
        r = requests.get(url, headers=_headers(), timeout=10)
        elapsed = time.time() - t0
        print(f"  Status : {r.status_code} ({elapsed:.2f}s)")
        if r.status_code == 200:
            data = r.json()
            models = data.get("data", data.get("models", []))
            print(f"  Models  : {[m.get('id') or m.get('model') for m in models]}")
            print("  ✅ Server ONLINE")
            return True
        else:
            print(f"  ❌ Server trả lỗi: {r.text[:200]}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Không kết nối được tới {BASE_URL}")
        return False
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# Test 2: POST /v1/chat/completions — prompt đơn giản
# ──────────────────────────────────────────────────────────────
def test_chat_simple():
    print(f"\n{SEP}")
    print("TEST 2: POST /v1/chat/completions — Prompt đơn giản")
    print(SEP)
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Trả lời đúng 1 từ: thủ đô của Việt Nam là gì?"}
        ],
        "temperature": 0.0,
        "max_tokens": 50,
        "stream": False,
    }
    try:
        t0 = time.time()
        r = requests.post(url, json=payload, headers=_headers(), timeout=60)
        elapsed = time.time() - t0
        print(f"  Status  : {r.status_code} ({elapsed:.2f}s)")
        if r.status_code == 200:
            data = r.json()
            content = data["choices"][0]["message"]["content"].strip()
            print(f"  Trả lời : {content!r}")
            print("  ✅ Chat endpoint hoạt động")
            return True
        else:
            print(f"  ❌ Lỗi: {r.text[:300]}")
            return False
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# Test 3: POST /v1/chat/completions — JSON mode (giống extractor)
# ──────────────────────────────────────────────────────────────
SAMPLE_PRESCRIPTION = """
PHÒNG KHÁM ĐA KHOA ABC
Bác sĩ: Nguyễn Văn Bình
Bệnh nhân: Trần Thị Mai, 45 tuổi, Nữ
SĐT: 0901234567
Ngày: 15/04/2025
Chẩn đoán: Viêm họng cấp

Đơn thuốc:
1. Amoxicillin 500mg - Uống sáng 1 viên, tối 1 viên - 10 viên
2. Paracetamol 500mg - Uống khi sốt, mỗi lần 1 viên - 20 viên
"""

def test_chat_json_mode():
    print(f"\n{SEP}")
    print("TEST 3: JSON mode — Giống luồng extractor thực tế")
    print(SEP)
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Bạn là chuyên gia y tế. Hãy luôn trả lời bằng JSON hợp lệ, không thêm text nào khác."
            },
            {
                "role": "user",
                "content": (
                    "Trích xuất thông tin từ đơn thuốc sau và trả về JSON với các trường: "
                    "patientName, patientAge, patientGender, phoneNumber, doctorName, "
                    "diagnosis, medications (mảng gồm productName, dosage, quantity, unit).\n\n"
                    f"{SAMPLE_PRESCRIPTION}"
                )
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "stream": False,
        "response_format": {"type": "json_object"}
    }
    try:
        print(f"  Đang gọi {url} ...")
        t0 = time.time()
        r = requests.post(url, json=payload, headers=_headers(), timeout=120)
        elapsed = time.time() - t0
        print(f"  Status  : {r.status_code} ({elapsed:.2f}s)")
        if r.status_code == 200:
            data = r.json()
            content = data["choices"][0]["message"]["content"].strip()
            # Thử parse JSON
            try:
                parsed = json.loads(content)
                print("  JSON parse: ✅ Hợp lệ")
                print(f"  patientName  : {parsed.get('patientName')}")
                print(f"  doctorName   : {parsed.get('doctorName')}")
                print(f"  diagnosis    : {parsed.get('diagnosis')}")
                meds = parsed.get('medications', [])
                print(f"  medications  : {len(meds)} thuốc")
                for i, m in enumerate(meds, 1):
                    print(f"    [{i}] {m.get('productName')} — {m.get('dosage')} — SL: {m.get('quantity')} {m.get('unit') or ''}")
                print("  ✅ JSON mode hoạt động đúng")
                return True
            except json.JSONDecodeError as e:
                print(f"  ⚠ JSON parse thất bại: {e}")
                print(f"  Raw content: {content[:400]}")
                return False
        else:
            print(f"  ❌ HTTP {r.status_code}: {r.text[:300]}")
            return False
    except requests.exceptions.Timeout:
        print("  ❌ Timeout — model quá chậm hoặc server bận")
        return False
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print("  MEDISPACE — Custom LLM API Test")
    print(f"  URL   : {BASE_URL}")
    print(f"  Model : {MODEL}")
    print(f"  APIKey: {'(có)' if API_KEY else '(trống — public)'}")
    print(f"{'#'*60}")

    results = {
        "server_online":   test_models_endpoint(),
        "chat_basic":      test_chat_simple(),
        "chat_json_mode":  test_chat_json_mode(),
    }

    print(f"\n{SEP}")
    print("KẾT QUẢ TỔNG HỢP:")
    all_pass = True
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    print(SEP)
    if all_pass:
        print("🎉 Tất cả test PASS — API sẵn sàng dùng cho OCR Service!")
    else:
        print("⚠  Một số test thất bại — kiểm tra lại URL, model name, hoặc kết nối mạng.")
    print()
    sys.exit(0 if all_pass else 1)
