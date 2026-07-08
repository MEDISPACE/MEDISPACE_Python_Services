"""Quality scoring and merge helpers for prescription OCR candidates."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import re
from typing import Any, Dict, List, Optional, Tuple


CRITICAL_FLAGS = {
    "invalid_json",
    "empty_medications",
    "date_impossible",
    "unsafe_hallucination",
}

KEY_FIELDS = ["patientName", "doctorName", "hospitalName", "prescriptionDate"]
MEDICATION_NAME_HIGH_WATERMARK = 0.8
MEDICATION_NAME_REVIEW_WATERMARK = 0.7
MEDICATION_DETAIL_HIGH_WATERMARK = 0.7


def empty_prescription_data(error: Optional[str] = None) -> Dict[str, Any]:
    data: Dict[str, Any] = {
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
    }
    if error:
        data["error"] = error
    return data


def normalize_prescription_data(data: Optional[Dict[str, Any]], source: str = "unknown") -> Dict[str, Any]:
    if not isinstance(data, dict):
        normalized = empty_prescription_data("Candidate is not a JSON object")
        normalized["_source"] = source
        return normalized

    normalized = empty_prescription_data()
    for key in normalized:
        if key == "medications":
            continue
        if key in data:
            normalized[key] = _clean_scalar(data.get(key))

    normalized["medications"] = _normalize_medications(data.get("medications"), source)

    if data.get("_extraction_method"):
        normalized["_extraction_method"] = data.get("_extraction_method")
    if data.get("error"):
        normalized["error"] = str(data.get("error"))
    normalized["_source"] = source
    return normalized


def score_candidate(data: Optional[Dict[str, Any]], source: str = "unknown", has_raw_text: bool = False, raw_text: str = "") -> Dict[str, Any]:
    candidate = normalize_prescription_data(data, source)
    flags: List[str] = []
    score = 0

    if not isinstance(data, dict):
        flags.append("invalid_json")
    else:
        score += 10

    meds = candidate.get("medications", [])
    if meds:
        score += 20
    else:
        flags.append("empty_medications")

    if meds:
        with_names = sum(1 for med in meds if _truthy(med.get("productName")))
        with_qty_or_unit = sum(1 for med in meds if med.get("quantity") is not None or _truthy(med.get("unit")))
        with_qty = sum(1 for med in meds if med.get("quantity") is not None)
        with_unit = sum(1 for med in meds if _truthy(med.get("unit")))
        score += round(25 * with_names / len(meds))
        score += round(12 * with_qty / len(meds))
        score += round(8 * with_unit / len(meds))
    else:
        with_names = 0
        with_qty_or_unit = 0
        with_qty = 0
        with_unit = 0

    medication_count = len(meds)

    present_key_fields = sum(1 for field in KEY_FIELDS if _truthy(candidate.get(field)))
    score += round(10 * present_key_fields / len(KEY_FIELDS))

    if has_raw_text or source == "vision":
        score += 5

    mapped_count = sum(1 for med in meds if _truthy(med.get("productId")))
    if meds:
        score += round(10 * mapped_count / len(meds))

    if candidate.get("error"):
        flags.append("candidate_error")
        score -= 10

    invalid_quantity_count = sum(
        1 for med in meds
        if med.get("quantity") is not None and med.get("quantity") <= 0
    )
    if invalid_quantity_count:
        flags.append("invalid_quantity")
        score -= min(20, invalid_quantity_count * 10)

    suspicious_large_quantity_count = sum(
        1 for med in meds
        if med.get("quantity") is not None and med.get("quantity") >= 120
    )
    if source == "traditional" and medication_count <= 1 and suspicious_large_quantity_count:
        flags.append("suspicious_large_quantity")
        score -= 20

    if _date_impossible(candidate.get("prescriptionDate")):
        flags.append("date_impossible")
        score -= 15

    duplicate_count = _duplicate_medication_count(meds)
    if duplicate_count:
        flags.append("duplicate_medications")
        score -= min(duplicate_count * 5, 15)

    name_ratio = (with_names / medication_count) if medication_count else 0
    qty_unit_ratio = (with_qty_or_unit / medication_count) if medication_count else 0
    quantity_ratio = (with_qty / medication_count) if medication_count else 0
    unit_ratio = (with_unit / medication_count) if medication_count else 0

    if medication_count and name_ratio < MEDICATION_NAME_REVIEW_WATERMARK:
        flags.append("weak_medication_names")
        score -= 10
    if medication_count and qty_unit_ratio < 0.5:
        flags.append("weak_medication_details")
        score -= 5

    suspicious_med_count = sum(1 for med in meds if _suspicious_medication(med))
    if suspicious_med_count:
        flags.append("suspicious_medication_text")
        score -= min(25, suspicious_med_count * 12)

    if source == "traditional" and medication_count and _traditional_candidate_looks_unsafe(
        candidate,
        raw_text,
        present_key_fields,
        suspicious_med_count,
        suspicious_large_quantity_count,
    ):
        flags.append("unsafe_hallucination")
        score -= 35

    # Administrative fields are useful, but prescription safety is anchored on
    # medication identity and quantity/unit. Cap quality when those are weak.
    if medication_count and name_ratio < MEDICATION_NAME_REVIEW_WATERMARK:
        score = min(score, 64)
    elif medication_count and qty_unit_ratio < 0.5:
        score = min(score, 74)
    if "unsafe_hallucination" in flags:
        score = min(score, 44)

    score = max(0, min(100, score))

    critical = sorted(set(flags) & CRITICAL_FLAGS)
    vision_name_only_candidate = (
        source == "vision"
        and medication_count >= 2
        and name_ratio >= MEDICATION_NAME_HIGH_WATERMARK
        and score >= 55
        and not critical
    )
    usable_medication_candidate = is_usable_medication_candidate(score, medication_count, name_ratio, qty_unit_ratio, critical)

    return {
        "score": score,
        "level": _score_level(score),
        "flags": sorted(set(flags)),
        "criticalFlags": critical,
        "medicationCount": medication_count,
        "medicationNameRatio": round(name_ratio, 3),
        "medicationQuantityUnitRatio": round(qty_unit_ratio, 3),
        "medicationQuantityRatio": round(quantity_ratio, 3),
        "medicationUnitRatio": round(unit_ratio, 3),
        "usableMedicationCandidate": usable_medication_candidate or vision_name_only_candidate,
        "usableByVisionNamesOnly": vision_name_only_candidate,
        "canEarlyReturn": can_early_return(score, medication_count, name_ratio, qty_unit_ratio, critical),
    }

def is_usable_medication_candidate(
    score: int,
    medication_count: int,
    name_ratio: float,
    qty_unit_ratio: float,
    critical_flags: List[str],
) -> bool:
    return medication_count >= 1 and name_ratio >= 0.5 and qty_unit_ratio >= 0.4 and score >= 55 and not critical_flags


def compare_candidates(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    conflicts: List[Dict[str, Any]] = []
    for field in ["patientName", "doctorName", "hospitalName", "prescriptionDate", "diagnosis"]:
        a = _clean_scalar(left.get(field))
        b = _clean_scalar(right.get(field))
        if a and b and _similarity(str(a), str(b)) < 0.65:
            conflicts.append({"field": field, "left": a, "right": b, "severity": "medium"})

    left_meds = left.get("medications", []) or []
    right_meds = right.get("medications", []) or []
    if left_meds and right_meds:
        matched = 0
        qty_conflicts = 0
        for med in left_meds:
            match = _best_medication_match(med, right_meds)
            if match:
                matched += 1
                if med.get("quantity") is not None and match.get("quantity") is not None and med.get("quantity") != match.get("quantity"):
                    qty_conflicts += 1
        if matched < min(len(left_meds), len(right_meds)) * 0.6:
            conflicts.append({"field": "medications", "left": len(left_meds), "right": len(right_meds), "severity": "high"})
        if qty_conflicts:
            conflicts.append({"field": "medication.quantity", "count": qty_conflicts, "severity": "medium"})

    return {
        "conflicts": conflicts,
        "conflictRate": round(len(conflicts) / 6, 3),
        "hasHighConflict": any(item.get("severity") == "high" for item in conflicts),
    }


def merge_candidates(
    traditional: Optional[Dict[str, Any]],
    vision: Optional[Dict[str, Any]],
    traditional_quality: Optional[Dict[str, Any]] = None,
    vision_quality: Optional[Dict[str, Any]] = None,
    traditional_raw_text: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    traditional_data = normalize_prescription_data(traditional, "traditional") if traditional else None
    vision_data = normalize_prescription_data(vision, "vision") if vision else None
    traditional_quality = traditional_quality or score_candidate(traditional_data, "traditional", True, traditional_raw_text)
    vision_quality = vision_quality or score_candidate(vision_data, "vision", False)

    if not traditional_data and not vision_data:
        merged = empty_prescription_data("No OCR candidate available")
        return merged, {"selectedSource": "none", "conflicts": [], "score": 0, "flags": ["empty_medications"]}
    if traditional_data and not vision_data:
        merged = _mark_medications_source(traditional_data, "traditional")
        quality = score_candidate(merged, "traditional", True)
        if not quality.get("usableMedicationCandidate"):
            empty = empty_prescription_data("No usable traditional OCR candidate available")
            return empty, {**score_candidate(empty, "traditional", True), "selectedSource": "none", "conflicts": []}
        return merged, {**quality, "selectedSource": "traditional", "conflicts": []}
    if vision_data and not traditional_data:
        merged = _mark_medications_source(vision_data, "vision")
        quality = score_candidate(merged, "vision", False)
        if not quality.get("usableMedicationCandidate"):
            empty = empty_prescription_data("No usable vision OCR candidate available")
            return empty, {**score_candidate(empty, "vision", False), "selectedSource": "none", "conflicts": []}
        return merged, {**quality, "selectedSource": "vision", "conflicts": []}

    assert traditional_data is not None and vision_data is not None
    comparison = compare_candidates(traditional_data, vision_data)
    primary, primary_source, selection_reason = _select_primary_candidate(
        traditional_data,
        vision_data,
        traditional_quality,
        vision_quality,
    )
    if primary_source == "none":
        merged = empty_prescription_data("No usable OCR candidate available")
        quality = score_candidate(merged, "merged", bool(traditional_data))
        return merged, {
            **quality,
            "selectedSource": "none",
            "selectionReason": selection_reason,
            "candidateScores": {
                "traditional": traditional_quality,
                "vision": vision_quality,
            },
            "conflicts": comparison["conflicts"],
            "conflictRate": comparison["conflictRate"],
        }
    secondary = traditional_data if primary is vision_data else vision_data

    merged = empty_prescription_data()
    for field in ["patientName", "patientAge", "patientGender", "phoneNumber", "doctorName", "hospitalName", "prescriptionDate", "diagnosis", "specialNotes"]:
        merged[field] = primary.get(field) or secondary.get(field)

    traditional_low_trust = bool(
        primary_source == "vision"
        and (
            not traditional_quality.get("usableMedicationCandidate")
            or bool(set(traditional_quality.get("flags") or []) & {"unsafe_hallucination", "suspicious_large_quantity"})
        )
    )
    if traditional_low_trust:
        merged["medications"] = [
            _mark_single_medication_source(med, "vision", needs_review=True, reason="traditional_low_trust_ignored")
            for med in vision_data.get("medications", [])
        ]
    else:
        merged["medications"] = _merge_medications(traditional_data.get("medications", []), vision_data.get("medications", []))
    merged["confidence"] = _merged_confidence(traditional_quality.get("score", 0), vision_quality.get("score", 0), comparison)
    merged["_extraction_method"] = "parallel_merged"

    merged_quality = score_candidate(merged, "merged", True)
    if comparison["hasHighConflict"]:
        merged_quality["flags"] = sorted(set(merged_quality["flags"] + ["high_conflict"]))
        merged_quality["score"] = max(0, merged_quality["score"] - 10)
        merged_quality["level"] = _score_level(merged_quality["score"])
        merged_quality["canEarlyReturn"] = False

    return merged, {
        **merged_quality,
        "selectedSource": primary_source,
        "selectionReason": selection_reason,
        "candidateScores": {
            "traditional": traditional_quality,
            "vision": vision_quality,
        },
        "conflicts": comparison["conflicts"],
        "conflictRate": comparison["conflictRate"],
    }


def can_early_return(score: int, medication_count: int, name_ratio: float, qty_unit_ratio: float, critical_flags: List[str]) -> bool:
    return (
        score >= 85
        and medication_count >= 1
        and name_ratio >= MEDICATION_NAME_HIGH_WATERMARK
        and qty_unit_ratio >= MEDICATION_DETAIL_HIGH_WATERMARK
        and not critical_flags
    )


def _normalize_medications(medications: Any, source: str) -> List[Dict[str, Any]]:
    if not isinstance(medications, list):
        return []
    normalized = []
    for med in medications:
        if not isinstance(med, dict):
            continue
        product_name = _clean_scalar(med.get("productName") or med.get("name") or med.get("drugName"))
        if _is_medication_header_noise(product_name):
            continue
        item = {
            "productName": product_name or "",
            "dosage": _clean_scalar(med.get("dosage")) or "",
            "quantity": _normalize_quantity(med.get("quantity")),
            "unit": _clean_scalar(med.get("unit")),
            "instructions": _clean_scalar(med.get("instructions")) or "",
            "source": med.get("source") or source,
        }
        for optional in ["productId", "matchedName", "image", "activeIngredient", "confidence", "needsReview", "reviewReason", "sources"]:
            if med.get(optional) is not None:
                item[optional] = med.get(optional)
        normalized.append(item)
    return normalized


def _merge_medications(traditional_meds: List[Dict[str, Any]], vision_meds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    used_vision = set()

    for trad in traditional_meds:
        vision_match = _best_medication_match(trad, vision_meds, used_vision)
        if vision_match:
            used_vision.add(id(vision_match))
            prefer_vision = _medication_completeness(vision_match) >= _medication_completeness(trad)
            item = dict(vision_match if prefer_vision else trad)
            item["productName"] = (vision_match if prefer_vision else trad).get("productName") or trad.get("productName") or vision_match.get("productName")
            item["activeIngredient"] = vision_match.get("activeIngredient") or trad.get("activeIngredient")
            item["source"] = "merged"
            item["sources"] = sorted(set([trad.get("source", "traditional"), vision_match.get("source", "vision")]))
            if _quantity_conflict(trad, vision_match):
                item["needsReview"] = True
                item["reviewReason"] = "quantity_conflict"
            merged.append(item)
        else:
            weak_vision_match = _weak_traditional_vision_name_match(trad, vision_meds, used_vision)
            if weak_vision_match:
                used_vision.add(id(weak_vision_match))
                item = dict(weak_vision_match)
                item["dosage"] = item.get("dosage") or trad.get("dosage") or ""
                item["quantity"] = item.get("quantity") if item.get("quantity") is not None else trad.get("quantity")
                item["unit"] = item.get("unit") or trad.get("unit")
                item["instructions"] = item.get("instructions") or trad.get("instructions") or ""
                item["source"] = "merged"
                item["sources"] = sorted(set([trad.get("source", "traditional"), weak_vision_match.get("source", "vision")]))
                item["needsReview"] = True
                item["reviewReason"] = "weak_traditional_name_replaced"
                merged.append(item)
                continue
            if _weak_noise_medication(trad):
                continue
            item = dict(trad)
            item["needsReview"] = True
            item["reviewReason"] = "only_traditional"
            merged.append(item)

    for vision in vision_meds:
        if id(vision) not in used_vision and not _weak_noise_medication(vision) and not _duplicate_medication_name(vision, merged):
            item = dict(vision)
            item["needsReview"] = True
            item["reviewReason"] = "only_vision"
            merged.append(item)

    return _expand_embedded_numbered_medications(merged)


def _best_medication_match(med: Dict[str, Any], candidates: List[Dict[str, Any]], used_ids: Optional[set] = None) -> Optional[Dict[str, Any]]:
    used_ids = used_ids or set()
    name = str(med.get("productName") or "")
    best = None
    best_score = 0.0
    for candidate in candidates:
        if id(candidate) in used_ids:
            continue
        score = _similarity(name, str(candidate.get("productName") or ""))
        if score > best_score:
            best = candidate
            best_score = score
    if not best:
        return None
    if best_score >= 0.72:
        return best
    if best_score >= 0.55 and _has_medication_identity_overlap(name, str(best.get("productName") or "")):
        return best
    return None

def _has_medication_identity_overlap(left: str, right: str) -> bool:
    left_tokens = _identity_name_tokens(_normalize_text_key(left))
    right_tokens = _identity_name_tokens(_normalize_text_key(right))
    return bool(left_tokens & right_tokens)

def _identity_name_tokens(value: str) -> set[str]:
    tokens = set()
    for token in value.split():
        if len(token) < 4:
            continue
        if any(ch.isdigit() for ch in token):
            continue
        tokens.add(token)
    return tokens

def _weak_traditional_vision_name_match(
    traditional_med: Dict[str, Any],
    vision_meds: List[Dict[str, Any]],
    used_ids: set,
) -> Optional[Dict[str, Any]]:
    if not _weak_noise_medication(traditional_med):
        return None
    if not (_truthy(traditional_med.get("dosage")) or _truthy(traditional_med.get("instructions"))):
        return None

    for vision in vision_meds:
        if id(vision) in used_ids or _weak_noise_medication(vision):
            continue
        if vision.get("quantity") is not None or _truthy(vision.get("unit")):
            continue
        name = str(vision.get("productName") or "").strip()
        if len(name) >= 5:
            return vision
    return None


def _expand_embedded_numbered_medications(meds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    existing_names = {_normalize_text_key(str(med.get("productName") or "")) for med in meds}

    for med in meds:
        item = dict(med)
        extra_items: List[Dict[str, Any]] = []
        for field in ["dosage", "instructions"]:
            text = str(item.get(field) or "")
            parsed = _parse_embedded_numbered_medication(text)
            if not parsed:
                continue

            cleaned_text, extra_name, extra_instruction = parsed
            item[field] = cleaned_text
            if field == "dosage" and str(item.get("instructions") or "") == text:
                item["instructions"] = cleaned_text

            name_key = _normalize_text_key(extra_name)
            if name_key and name_key not in existing_names:
                existing_names.add(name_key)
                extra_items.append({
                    "productName": extra_name,
                    "dosage": extra_instruction,
                    "quantity": None,
                    "unit": None,
                    "instructions": extra_instruction,
                    "source": item.get("source", "merged"),
                    "needsReview": True,
                    "reviewReason": "split_from_embedded_numbered_instruction",
                })

        expanded.append(item)
        expanded.extend(extra_items)

    return expanded

def _parse_embedded_numbered_medication(text: str) -> Optional[Tuple[str, str, str]]:
    if not text or not re.search(r"\b\d+\.\s*", text):
        return None

    match = re.search(r"^(?P<prefix>.*?)[,;]?\s+\d+\.\s*(?P<name>[^,;]+?)(?:[,;]\s*(?P<instruction>.+))?$", text)
    if not match:
        return None

    prefix = (match.group("prefix") or "").strip(" ,;")
    name = (match.group("name") or "").strip(" ,;")
    instruction = (match.group("instruction") or "").strip(" ,;")
    if not prefix or not name or _is_medication_header_noise(name):
        return None
    if len(_normalize_text_key(name)) < 5:
        return None

    return prefix, name, instruction

def _mark_medications_source(data: Dict[str, Any], source: str) -> Dict[str, Any]:
    result = dict(data)
    result["medications"] = [
        _mark_single_medication_source(med, source, needs_review=(source == "vision"), reason="vision_only_candidate")
        for med in data.get("medications", [])
    ]
    return result

def _select_primary_candidate(
    traditional_data: Dict[str, Any],
    vision_data: Dict[str, Any],
    traditional_quality: Dict[str, Any],
    vision_quality: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, str]:
    traditional_usable = bool(traditional_quality.get("usableMedicationCandidate"))
    vision_usable = bool(vision_quality.get("usableMedicationCandidate"))
    traditional_score = traditional_quality.get("score", 0)
    vision_score = vision_quality.get("score", 0)

    if vision_usable and not traditional_usable:
        return vision_data, "vision", "traditional_unusable_vision_has_medications"
    if traditional_usable and not vision_usable:
        return traditional_data, "traditional", "vision_unusable_traditional_has_medications"
    if vision_usable and traditional_usable and vision_score >= traditional_score + 8:
        return vision_data, "vision", "vision_quality_advantage"
    if traditional_usable:
        return traditional_data, "traditional", "traditional_baseline_or_tie"
    if vision_score > traditional_score:
        return vision_data, "vision", "vision_higher_low_confidence"
    return empty_prescription_data("No usable OCR candidate available"), "none", "no_usable_candidate"

def _mark_single_medication_source(
    med: Dict[str, Any],
    source: str,
    needs_review: bool = False,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    item = dict(med)
    item["source"] = item.get("source") or source
    if needs_review:
        item["needsReview"] = True
        item["reviewReason"] = item.get("reviewReason") or reason
    return item


def _medication_completeness(med: Dict[str, Any]) -> int:
    return sum(1 for key in ["productName", "dosage", "quantity", "unit", "instructions", "activeIngredient"] if _truthy(med.get(key)))


def _quantity_conflict(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return left.get("quantity") is not None and right.get("quantity") is not None and left.get("quantity") != right.get("quantity")


def _duplicate_medication_name(candidate: Dict[str, Any], existing_meds: List[Dict[str, Any]]) -> bool:
    candidate_key = _normalize_text_key(str(candidate.get("productName") or ""))
    if not candidate_key:
        return True

    candidate_tokens = _significant_name_tokens(candidate_key)
    for existing in existing_meds:
        existing_key = _normalize_text_key(str(existing.get("productName") or ""))
        if not existing_key:
            continue
        if candidate_key == existing_key or candidate_key in existing_key or existing_key in candidate_key:
            return True
        existing_tokens = _significant_name_tokens(existing_key)
        if candidate_tokens and candidate_tokens.issubset(existing_tokens):
            return True
        if existing_tokens and existing_tokens.issubset(candidate_tokens) and len(existing_tokens) >= 2:
            return True
        if _similarity(candidate_key, existing_key) >= 0.72:
            return True
    return False

def _significant_name_tokens(value: str) -> set[str]:
    return {token for token in value.split() if len(token) >= 3 or any(ch.isdigit() for ch in token)}

def _weak_noise_medication(med: Dict[str, Any]) -> bool:
    name = str(med.get("productName") or "").strip()
    if _is_medication_header_noise(name):
        return True
    normalized = _normalize_text_key(name)
    if _is_date_like_medication_name(name):
        return True
    has_quantity_or_unit = med.get("quantity") is not None or _truthy(med.get("unit"))
    if has_quantity_or_unit:
        return False
    if len(normalized) <= 4 and (_truthy(med.get("dosage")) or _truthy(med.get("instructions"))):
        return True
    return False

def _is_medication_header_noise(name: Any) -> bool:
    normalized = _normalize_text_key(str(name or ""))
    return normalized in {
        "ham luong",
        "hàm lượng",
        "so luong",
        "số lượng",
        "ten thuoc",
        "tên thuốc",
        "ten thuoc ham luong",
        "tên thuốc hàm lượng",
        "danh sach thuoc",
        "danh sách thuốc",
        "thuoc dung",
        "thuốc dùng",
        "thuoc dieu tri",
        "thuốc điều trị",
        "thong tin benh nhan",
        "thông tin bệnh nhân",
        "benh nhan",
        "bệnh nhân",
        "ho ten",
        "họ tên",
        "ho ten nguoi benh",
        "họ tên người bệnh",
        "ten",
        "tên",
        "tuoi",
        "tuổi",
        "dia chi",
        "địa chỉ",
        "chan doan",
        "chẩn đoán",
        "diagnosis",
        "luu y",
        "lưu ý",
        "ghi chu",
        "ghi chú",
        "ngay kham",
        "ngày khám",
        "ngay ke don",
        "ngày kê đơn",
        "ngay tai kham",
        "ngày tái khám",
        "tai kham",
        "tái khám",
        "phan ghi chu cuoi don",
        "phần ghi chú cuối đơn",
        "nuoc sx",
        "nước sx",
        "nuoc san xuat",
        "nước sản xuất",
        "nha san xuat",
        "nhà sản xuất",
        "xuat xu",
        "xuất xứ",
        "viet nam",
        "việt nam",
        "dvt",
        "don vi",
        "đơn vị",
        "stt",
    }

def _is_date_like_medication_name(name: str) -> bool:
    text = str(name or "").strip()
    normalized = _normalize_text_key(text)
    if re.fullmatch(r"\d{1,2}[/.-]\d{1,2}(?:[/.-]\d{2,4})?", text):
        return True
    if re.fullmatch(r"\d{1,2}\s*(?:thang|tháng)\s*\d{1,2}(?:\s*(?:nam|năm)\s*\d{2,4})?", normalized):
        return True
    return False

def _clean_scalar(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        return str(value).strip() or None
    value = value.strip()
    return value or None

def _suspicious_medication(med: Dict[str, Any]) -> bool:
    name = str(med.get("productName") or "")
    dosage = str(med.get("dosage") or med.get("instructions") or "")
    normalized_name = _normalize_text_key(name)
    normalized_dosage = _normalize_text_key(dosage)
    tokens = [token for token in normalized_name.split() if len(token) >= 2]
    repeated_tokens = any(count >= 3 for count in Counter(tokens).values())
    too_many_separators = len(re.findall(r"[-_/.,]", name)) >= 4
    mostly_short_tokens = len(tokens) >= 3 and sum(1 for token in tokens if len(token) <= 2) / len(tokens) >= 0.6
    dosage_fragment = bool(dosage) and len(normalized_dosage.split()) <= 4 and any(token in normalized_dosage.split() for token in {"lan", "tran", "m"})
    return repeated_tokens or too_many_separators or mostly_short_tokens or dosage_fragment

def _traditional_candidate_looks_unsafe(
    candidate: Dict[str, Any],
    raw_text: str,
    present_key_fields: int,
    suspicious_med_count: int,
    suspicious_large_quantity_count: int = 0,
) -> bool:
    meds = candidate.get("medications") or []
    if not meds:
        return False
    raw_quality_bad = _raw_text_noise_ratio(raw_text) >= 0.45 or _numeric_line_ratio(raw_text) >= 0.35
    single_med_no_context = len(meds) == 1 and present_key_fields == 0
    single_med_large_quantity = len(meds) == 1 and suspicious_large_quantity_count > 0
    has_zero_quantity = any(med.get("quantity") is not None and med.get("quantity") <= 0 for med in meds)
    all_from_weak_text = single_med_no_context and raw_quality_bad
    return bool(
        single_med_large_quantity
        or (all_from_weak_text and (suspicious_med_count or has_zero_quantity))
        or (single_med_no_context and suspicious_med_count and has_zero_quantity)
    )

def _raw_text_noise_ratio(raw_text: str) -> float:
    text = raw_text or ""
    if not text.strip():
        return 1.0
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 1.0
    noisy = sum(1 for ch in chars if ch.isdigit() or ch in ".:_-*/\\|[]{}()")
    return noisy / len(chars)

def _numeric_line_ratio(raw_text: str) -> float:
    lines = [line.strip() for line in (raw_text or "").splitlines() if line.strip()]
    if not lines:
        return 1.0
    numeric_like = sum(1 for line in lines if re.fullmatch(r"[0-9\W_]{3,}", line))
    return numeric_like / len(lines)

def _normalize_text_key(value: str) -> str:
    value = (value or "").lower()
    value = re.sub(r"[^a-z0-9à-ỹđ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _normalize_quantity(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def _truthy(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _similarity(left: str, right: str) -> float:
    left = " ".join(left.lower().split())
    right = " ".join(right.lower().split())
    if not left or not right:
        return 0.0
    if left in right or right in left:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _date_impossible(value: Any) -> bool:
    if not value:
        return False
    text = str(value)
    if len(text) >= 4 and text[:4].isdigit():
        year = int(text[:4])
        return year < 1900 or year > 2100
    return False


def _duplicate_medication_count(meds: List[Dict[str, Any]]) -> int:
    names = [str(med.get("productName") or "").strip().lower() for med in meds if med.get("productName")]
    return len(names) - len(set(names))


def _score_level(score: int) -> str:
    if score >= 85:
        return "high"
    if score >= 65:
        return "medium"
    return "low"


def _merged_confidence(traditional_score: int, vision_score: int, comparison: Dict[str, Any]) -> str:
    score = max(traditional_score, vision_score)
    if comparison.get("hasHighConflict"):
        score -= 15
    return _score_level(max(0, score))
