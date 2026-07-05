"""Benchmark MEDISPACE prescription OCR modes.

Examples:
  python -m scripts.benchmark
  python -m scripts.benchmark --modes traditional,vision,parallel_benchmark --repeats 5
  python -m scripts.benchmark --modes traditional,parallel_benchmark --repeats 1 --files Donthuoc3.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


OCR_URL = os.getenv("OCR_URL", "http://localhost:8001/api/ocr/extract-prescription")
TEST_DIR = Path(__file__).parent.parent / "test_images"
GT_FILE = TEST_DIR / "ground_truth.json"
REPORT_DIR = Path(os.getenv("OCR_BENCHMARK_REPORT_DIR", TEST_DIR / "benchmark_reports"))


def normalize(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def fuzzy_match(expected: str, actual: str, threshold: float = 0.6) -> bool:
    exp, act = normalize(expected), normalize(actual)
    if not exp or not act:
        return exp == act
    if exp == act or exp in act or act in exp:
        return True
    exp_words = set(exp.split())
    act_words = set(act.split())
    if not exp_words:
        return False
    return len(exp_words & act_words) / len(exp_words) >= threshold


def compare_medications(gt_meds: List[Dict[str, Any]], ocr_meds: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "count_expected": len(gt_meds),
        "count_actual": len(ocr_meds),
        "count_match": len(gt_meds) == len(ocr_meds),
        "name_matches": 0,
        "qty_matches": 0,
        "unit_matches": 0,
        "qty_expected_count": sum(1 for med in gt_meds if med.get("quantity") is not None),
        "unit_expected_count": sum(1 for med in gt_meds if med.get("unit") not in (None, "")),
        "details": [],
    }

    for gt in gt_meds:
        best_match = None
        best_score = 0
        gt_name = normalize(gt.get("productName", ""))
        for ocr in ocr_meds:
            ocr_name = normalize(ocr.get("productName", ""))
            if fuzzy_match(gt_name, ocr_name, 0.5):
                score = 1
                if gt.get("quantity") is not None and ocr.get("quantity") == gt.get("quantity"):
                    score += 1
                if gt.get("unit") not in (None, "") and fuzzy_match(str(gt.get("unit", "")), str(ocr.get("unit", ""))):
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
            detail["qty_match"] = None if gt.get("quantity") is None else best_match.get("quantity") == gt.get("quantity")
            detail["unit_match"] = None if gt.get("unit") in (None, "") else fuzzy_match(str(gt.get("unit", "")), str(best_match.get("unit", "")))
            result["name_matches"] += 1
            if detail["qty_match"] is True:
                result["qty_matches"] += 1
            if detail["unit_match"] is True:
                result["unit_matches"] += 1
        else:
            detail["actual_name"] = None
            detail["name_match"] = False
            detail["qty_match"] = False
            detail["unit_match"] = False
        result["details"].append(detail)
    return result


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def call_ocr(image_path: Path, mode: str) -> Dict[str, Any]:
    with open(image_path, "rb") as image_file:
        response = requests.post(
            OCR_URL,
            data={"mode": mode},
            files={"file": (image_path.name, image_file, "image/jpeg")},
            timeout=240,
        )
    response.raise_for_status()
    return response.json()


def run_benchmark(modes: List[str], repeats: int, files: Optional[List[str]] = None) -> None:
    if not GT_FILE.exists():
        raise SystemExit(f"Missing ground truth file: {GT_FILE}")
    with open(GT_FILE, encoding="utf-8") as f:
        gt_data = json.load(f)

    images = gt_data.get("images", [])
    if files:
        wanted = {filename.strip() for filename in files if filename.strip()}
        images = [entry for entry in images if entry.get("filename") in wanted]
        missing = sorted(wanted - {entry.get("filename") for entry in images})
        if missing:
            raise SystemExit(f"Files not found in ground_truth.json: {', '.join(missing)}")
    if not images:
        raise SystemExit("ground_truth.json has no images")

    print("=" * 70)
    print("MEDISPACE OCR BENCHMARK")
    print("=" * 70)
    print(f"OCR endpoint: {OCR_URL}")
    print(f"Images: {len(images)} | Modes: {', '.join(modes)} | Repeats: {repeats}")
    if files:
        print(f"Files: {', '.join(entry['filename'] for entry in images)}")

    rows: List[Dict[str, Any]] = []
    simple_fields = ["patientName", "patientAge", "patientGender", "doctorName", "hospitalName", "prescriptionDate", "diagnosis"]

    for mode in modes:
        for entry in images:
            filename = entry["filename"]
            image_path = TEST_DIR / filename
            gt = entry.get("ground_truth", {})
            if not image_path.exists():
                print(f"SKIP missing file: {filename}")
                continue

            print(f"\nTesting {filename} mode={mode}")
            for repeat_idx in range(repeats):
                started = time.time()
                row: Dict[str, Any] = {"mode": mode, "filename": filename, "repeat": repeat_idx + 1}
                try:
                    payload = call_ocr(image_path, mode)
                    elapsed = time.time() - started
                    row["elapsed_seconds"] = round(elapsed, 3)
                    row["success"] = bool(payload.get("success"))
                    row["message"] = payload.get("message")

                    timing = payload.get("timing", {}) or {}
                    quality = payload.get("quality", {}) or {}
                    row["quality_score"] = quality.get("score")
                    row["quality_level"] = quality.get("level")
                    row["conflict_rate"] = quality.get("conflictRate", 0)
                    row["vision_timed_out"] = bool(timing.get("visionTimedOut"))
                    row["traditional_seconds"] = timing.get("traditional_total_pipeline_seconds")
                    row["vision_seconds"] = timing.get("vision_llm_seconds")

                    ocr = payload.get("data", {}) or {}
                    for field in simple_fields:
                        expected_value = gt.get(field)
                        if expected_value in (None, ""):
                            row[f"field_{field}_match"] = None
                            continue
                        match = fuzzy_match(str(expected_value), str(ocr.get(field) or ""))
                        row[f"field_{field}_match"] = match

                    med_result = compare_medications(gt.get("medications", []), ocr.get("medications", []) or [])
                    row["med_count_match"] = med_result["count_match"]
                    expected_count = med_result["count_expected"] or 1
                    qty_expected_count = med_result["qty_expected_count"] or 1
                    unit_expected_count = med_result["unit_expected_count"] or 1
                    row["med_name_score"] = round(med_result["name_matches"] / expected_count, 3)
                    row["med_qty_score"] = round(med_result["qty_matches"] / qty_expected_count, 3)
                    row["med_unit_score"] = round(med_result["unit_matches"] / unit_expected_count, 3)
                    print(f"  run {repeat_idx + 1}/{repeats}: {elapsed:.2f}s quality={row.get('quality_score')} med_name={row['med_name_score']}")
                except Exception as exc:
                    row["success"] = False
                    row["elapsed_seconds"] = round(time.time() - started, 3)
                    row["message"] = str(exc)
                    print(f"  run {repeat_idx + 1}/{repeats}: ERROR {exc}")
                rows.append(row)

    print_summary(rows)
    write_reports(rows, modes, repeats)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode and row.get("success")]
        times = [float(row["elapsed_seconds"]) for row in mode_rows]
        timeouts = sum(1 for row in mode_rows if row.get("vision_timed_out"))
        if not mode_rows:
            print(f"{mode}: no successful runs")
            continue
        print(
            f"{mode}: runs={len(mode_rows)} avg={sum(times)/len(times):.2f}s "
            f"p50={percentile(times, 50):.2f}s p75={percentile(times, 75):.2f}s "
            f"p90={percentile(times, 90):.2f}s p95={percentile(times, 95):.2f}s "
            f"p99={percentile(times, 99):.2f}s timeouts={timeouts}"
        )


def write_reports(rows: List[Dict[str, Any]], modes: List[str], repeats: int) -> None:
    if not rows:
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = REPORT_DIR / f"ocr_benchmark_{ts}"
    with open(f"{base}.json", "w", encoding="utf-8") as f:
        json.dump({"modes": modes, "repeats": repeats, "rows": rows}, f, ensure_ascii=False, indent=2)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(f"{base}.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Reports written: {base}.json and {base}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MEDISPACE OCR modes")
    parser.add_argument("--modes", default="traditional,vision,parallel_benchmark")
    parser.add_argument("--repeats", type=int, default=int(os.getenv("OCR_BENCHMARK_REPEATS", "5")))
    parser.add_argument("--files", default="", help="Comma-separated image filenames from ground_truth.json")
    args = parser.parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    files = [filename.strip() for filename in args.files.split(",") if filename.strip()]
    run_benchmark(modes, max(1, args.repeats), files or None)
