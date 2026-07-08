import json
import os
from pathlib import Path

import pytest

from scripts.benchmark import OCR_URL, call_ocr, compare_medications


IMAGE_NAMES = ["Donthuoc9.jpg", "Donthuoc10.jpg", "Donthuoc11.jpg"]
TEST_IMAGES_DIR = Path(__file__).resolve().parents[1] / "test_images"
GROUND_TRUTH_FILE = TEST_IMAGES_DIR / "ground_truth.json"


def _load_ground_truth() -> dict:
    data = json.loads(GROUND_TRUTH_FILE.read_text(encoding="utf-8"))
    return {entry["filename"]: entry["ground_truth"] for entry in data.get("images", [])}


@pytest.mark.parametrize("image_name", IMAGE_NAMES)
def test_named_prescription_images_have_ground_truth(image_name: str) -> None:
    ground_truth = _load_ground_truth()
    image_path = TEST_IMAGES_DIR / image_name

    assert image_path.exists(), f"Missing prescription fixture image: {image_path}"
    assert image_path.stat().st_size > 0
    assert image_name in ground_truth, f"Missing ground truth entry for {image_name}"

    expected_meds = ground_truth[image_name].get("medications") or []
    assert len(expected_meds) == ground_truth[image_name].get("medications_count")
    assert expected_meds, f"Ground truth for {image_name} must include medications"
    assert all(med.get("productName") for med in expected_meds)


@pytest.mark.integration
@pytest.mark.parametrize("image_name", IMAGE_NAMES)
def test_ocr_api_extracts_required_image_fixtures(image_name: str) -> None:
    if os.getenv("RUN_OCR_IMAGE_TESTS") != "1":
        pytest.skip("Set RUN_OCR_IMAGE_TESTS=1 to run OCR image integration tests")

    ground_truth = _load_ground_truth()[image_name]
    image_path = TEST_IMAGES_DIR / image_name
    payload = call_ocr(image_path, os.getenv("OCR_IMAGE_TEST_MODE", "parallel"))

    medications = ((payload.get("data") or {}).get("medications") or [])
    comparison = compare_medications(ground_truth.get("medications") or [], medications)

    assert payload.get("success") is True, payload
    assert len(medications) >= max(1, ground_truth.get("medications_count", 1) - 1), payload
    assert comparison["name_matches"] >= max(1, ground_truth.get("medications_count", 1) // 2), {
        "ocr_url": OCR_URL,
        "image": image_name,
        "comparison": comparison,
        "payload": payload,
    }
