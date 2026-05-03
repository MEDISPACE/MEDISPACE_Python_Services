"""
scripts/benchmark.py
Benchmark script: đo accuracy per-field của OCR pipeline.

Cách dùng:
  1. Đặt ảnh đơn thuốc vào test_images/
  2. Cập nhật ground truth trong test_images/ground_truth.json
  3. Chạy: python -m scripts.benchmark

Output: Bảng accuracy per-field + tổng kết.
"""
import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional

# ── Config ───────────────────────────────────────────────────────────────────
OCR_URL = os.getenv("OCR_URL", "http://localhost:8001/api/ocr/extract-prescription")
TEST_DIR = Path(__file__).parent.parent / "test_images"
GT_FILE = TEST_DIR / "ground_truth.json"


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize(s: Optional[str]) -> str:
    """Chuẩn hoá chuỗi để so sánh."""
    if s is None:
        return ""
    return s.strip().lower()


def fuzzy_match(expected: str, actual: str, threshold: float = 0.6) -> bool:
    """So sánh mềm: chứa nhau hoặc tỷ lệ trùng từ >= threshold."""
    e, a = normalize(expected), normalize(actual)
    if not e or not a:
        return e == a  # cả 2 rỗng → match

    # Exact
    if e == a:
        return True

    # Contains
    if e in a or a in e:
        return True

    # Word overlap
    e_words = set(e.split())
    a_words = set(a.split())
    if not e_words:
        return False
    overlap = len(e_words & a_words) / len(e_words)
    return overlap >= threshold


def compare_medications(gt_meds: List[Dict], ocr_meds: List[Dict]) -> Dict[str, Any]:
    """So sánh danh sách thuốc: kiểm tra tên, số lượng, đơn vị."""
    result = {
        "count_expected": len(gt_meds),
        "count_actual": len(ocr_meds),
        "count_match": len(gt_meds) == len(ocr_meds),
        "name_matches": 0,
        "qty_matches": 0,
        "unit_matches": 0,
        "details": [],
    }

    for gt in gt_meds:
        gt_name = normalize(gt.get("productName", ""))
        best_match = None
        best_score = 0

        for ocr in ocr_meds:
            ocr_name = normalize(ocr.get("productName", ""))
            if fuzzy_match(gt_name, ocr_name, 0.5):
                # Tính điểm match
                score = 1
                if gt.get("quantity") is not None and ocr.get("quantity") == gt["quantity"]:
                    score += 1
                if fuzzy_match(str(gt.get("unit", "")), str(ocr.get("unit", ""))):
                    score += 1
                if score > best_score:
                    best_score = score
                    best_match = ocr

        detail = {
            "expected_name": gt.get("productName"),
            "expected_qty": gt.get("quantity"),
            "expected_unit": gt.get("unit"),
        }

        if best_match:
            detail["actual_name"] = best_match.get("productName")
            detail["actual_qty"] = best_match.get("quantity")
            detail["actual_unit"] = best_match.get("unit")
            detail["name_match"] = True
            result["name_matches"] += 1
            detail["qty_match"] = (best_match.get("quantity") == gt.get("quantity"))
            if detail["qty_match"]:
                result["qty_matches"] += 1
            detail["unit_match"] = fuzzy_match(
                str(gt.get("unit", "")), str(best_match.get("unit", ""))
            )
            if detail["unit_match"]:
                result["unit_matches"] += 1
        else:
            detail["actual_name"] = None
            detail["name_match"] = False
            detail["qty_match"] = False
            detail["unit_match"] = False

        result["details"].append(detail)

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def run_benchmark():
    """Chạy benchmark trên tất cả ảnh trong test_images/."""
    if not GT_FILE.exists():
        print(f"❌ Không tìm thấy ground truth: {GT_FILE}")
        print("   Hãy tạo file test_images/ground_truth.json trước.")
        sys.exit(1)

    with open(GT_FILE, encoding="utf-8") as f:
        gt_data = json.load(f)

    images = gt_data.get("images", [])
    if not images:
        print("❌ Ground truth rỗng — không có ảnh nào để test.")
        sys.exit(1)

    print("=" * 70)
    print("  MEDISPACE OCR BENCHMARK")
    print("=" * 70)
    print(f"  Test images: {len(images)}")
    print(f"  OCR endpoint: {OCR_URL}")
    print()

    # Accumulate results
    all_results = []
    field_scores = {
        "patientName": [], "patientAge": [], "patientGender": [],
        "doctorName": [], "hospitalName": [], "prescriptionDate": [],
        "diagnosis": [],
    }
    med_scores = {"name": [], "qty": [], "unit": [], "count": []}
    timings = []

    for entry in images:
        filename = entry["filename"]
        gt = entry["ground_truth"]
        image_path = TEST_DIR / filename

        if not image_path.exists():
            print(f"⚠  SKIP: {filename} — file không tồn tại")
            continue

        print(f"📸 Testing: {filename}")
        print("-" * 50)

        # Call OCR API
        t_start = time.time()
        try:
            with open(image_path, "rb") as img_file:
                resp = requests.post(
                    OCR_URL,
                    files={"file": (filename, img_file, "image/jpeg")},
                    timeout=180,
                )
            resp_data = resp.json()
        except Exception as e:
            print(f"  ❌ Lỗi khi gọi API: {e}")
            continue
        t_elapsed = time.time() - t_start
        timings.append(t_elapsed)

        if not resp_data.get("success"):
            print(f"  ❌ OCR thất bại: {resp_data.get('message')}")
            continue

        ocr = resp_data.get("data", {})

        # Compare simple fields
        for field in field_scores:
            expected = gt.get(field)
            actual = ocr.get(field)
            is_match = fuzzy_match(str(expected or ""), str(actual or ""))
            field_scores[field].append(is_match)
            status = "✅" if is_match else "❌"
            print(f"  {status} {field:20s} | Expected: {expected!r:30s} | Got: {actual!r}")

        # Compare medications
        gt_meds = gt.get("medications", [])
        ocr_meds = ocr.get("medications", [])
        med_result = compare_medications(gt_meds, ocr_meds)

        med_scores["count"].append(med_result["count_match"])
        if med_result["count_expected"] > 0:
            med_scores["name"].append(med_result["name_matches"] / med_result["count_expected"])
            med_scores["qty"].append(med_result["qty_matches"] / med_result["count_expected"])
            med_scores["unit"].append(med_result["unit_matches"] / med_result["count_expected"])

        count_status = "✅" if med_result["count_match"] else "❌"
        print(f"  {count_status} {'medications_count':20s} | Expected: {med_result['count_expected']} | Got: {med_result['count_actual']}")

        for d in med_result["details"]:
            n_s = "✅" if d["name_match"] else "❌"
            q_s = "✅" if d["qty_match"] else "❌"
            u_s = "✅" if d["unit_match"] else "❌"
            print(f"      {n_s} {d['expected_name']!r:25s} → {d.get('actual_name', 'MISS')!r:25s} | qty {q_s} {d.get('actual_qty', '?')} | unit {u_s} {d.get('actual_unit', '?')}")

        print(f"  ⏱  Time: {t_elapsed:.2f}s")
        print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print(f"\n  {'Field':<25s} {'Accuracy':>10s} {'Correct':>10s} {'Total':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")

    for field, scores in field_scores.items():
        if not scores:
            continue
        acc = sum(scores) / len(scores) * 100
        correct = sum(scores)
        print(f"  {field:<25s} {acc:>9.1f}% {correct:>10d} {len(scores):>8d}")

    print(f"\n  {'Medications':>25s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")

    for key in ["count", "name", "qty", "unit"]:
        scores = med_scores[key]
        if not scores:
            continue
        acc = sum(scores) / len(scores) * 100
        label = f"med_{key}"
        print(f"  {label:<25s} {acc:>9.1f}%")

    if timings:
        print(f"\n  ⏱  Avg time: {sum(timings)/len(timings):.2f}s")
        print(f"  ⏱  Total time: {sum(timings):.2f}s")

    print()


if __name__ == "__main__":
    run_benchmark()
