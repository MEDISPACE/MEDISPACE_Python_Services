"""
src/utils/debug_logger.py
Ghi log debug OCR ra file cho mỗi request.
Giúp nhanh chóng xác định lỗi ở tầng nào (OCR text sai hay Regex parse sai).
"""
import os
import json
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def save_debug_log(
    filename: str,
    raw_text: str,
    extracted_data: dict,
    timing: dict,
    image_shape: tuple = None,
    num_boxes: int = 0,
):
    """
    Lưu log debug OCR cho 1 request.

    Output file: logs/ocr_debug_{timestamp}_{filename}.txt
    Nội dung:
      - Timestamp
      - Image info (shape, num_boxes)
      - Raw OCR text (toàn bộ)
      - Extracted data (JSON)
      - Timing
    """
    ensure_log_dir()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in (filename or "unknown"))
    log_filename = f"ocr_debug_{ts}_{safe_name}.txt"
    log_path = os.path.join(LOG_DIR, log_filename)

    # Tách medications ra riêng cho dễ đọc
    meds = extracted_data.get("medications", [])
    meds_summary = []
    for i, m in enumerate(meds):
        meds_summary.append(
            f"  [{i+1}] {m.get('productName', '?')} | "
            f"SL: {m.get('quantity', '?')} {m.get('unit', '?')} | "
            f"Liều: {m.get('dosage', '?')}"
        )

    content = f"""════════════════════════════════════════════════════════════════
  MEDISPACE OCR Debug Log
  Time: {datetime.now().isoformat()}
  File: {filename}
════════════════════════════════════════════════════════════════

── Image Info ──────────────────────────────────────────────────
  Shape:     {image_shape}
  Boxes:     {num_boxes}

── Timing ─────────────────────────────────────────────────────
  PaddleOCR: {timing.get('station1_PaddleOCR_seconds', '?')}s
  VietOCR:   {timing.get('station2_VietOCR_seconds', '?')}s
  Extractor: {timing.get('station3_Extractor_seconds', '?')}s
  Total:     {timing.get('total_pipeline_seconds', '?')}s
  Backend:   {timing.get('extractor_backend', '?')}

── Extraction Method ──────────────────────────────────────────
  Method:     {extracted_data.get('_extraction_method', '?')}
  Confidence: {extracted_data.get('confidence', '?')}

── Extracted Fields ───────────────────────────────────────────
  Patient:    {extracted_data.get('patientName', None)}
  Age:        {extracted_data.get('patientAge', None)}
  Gender:     {extracted_data.get('patientGender', None)}
  Doctor:     {extracted_data.get('doctorName', None)}
  Hospital:   {extracted_data.get('hospitalName', None)}
  Date:       {extracted_data.get('prescriptionDate', None)}
  Diagnosis:  {extracted_data.get('diagnosis', None)}
  Notes:      {extracted_data.get('specialNotes', None)}

── Medications ({len(meds)}) ──────────────────────────────────
{chr(10).join(meds_summary) if meds_summary else '  (none)'}

── Raw OCR Text ───────────────────────────────────────────────
{raw_text}

── Full JSON ──────────────────────────────────────────────────
{json.dumps(extracted_data, ensure_ascii=False, indent=2)}
"""

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[DebugLog] Saved → {log_path}")
    except Exception as e:
        print(f"[DebugLog] Lỗi ghi log: {e}")

    return log_path
"""
Utility for saving debug logs for OCR requests.
"""
