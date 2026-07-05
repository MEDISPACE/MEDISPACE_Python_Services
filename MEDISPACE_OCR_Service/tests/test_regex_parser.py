from src.services.regex_parser import parse_with_regex
from src.services.quality import merge_candidates, score_candidate


def medication_names(raw_text: str) -> list[str]:
    return [med["productName"] for med in parse_with_regex(raw_text)["medications"]]


def test_split_line_prescription_keeps_quantity_units_and_dosage() -> None:
    raw_text = """
1
Ducray Kelual DS
x
1
lọ
Tắm gội 3 lần/tuần, cách ngày
2
Silkron
3
tube
Bôi tại chỗ 2 lần/ ngày
3
EMOLEUM LIPID BALM
x
1
tube
Xoa tại chỗ 3 lần/ ngày
4
Halcort 6
x
30
viên
Uống 3 viên/lần/ ngày
5
Destidin 5mg
x
30
viên
Uống 2 viên/lần/ ngày
6
Mardekin
x
1
hộp
Uống 2 viên/ lần x 2 lần/ ngày
7
HB DIGIC
x
60
viên
Uống 2 viên/ lần x 2 lần/ ngày
Lời dặn:
"""

    medications = parse_with_regex(raw_text)["medications"]

    assert medication_names(raw_text) == [
        "Ducray Kelual DS",
        "Silkron",
        "EMOLEUM LIPID BALM",
        "Halcort 6",
        "Destidin 5mg",
        "Mardekin",
        "HB DIGIC",
    ]
    assert medications[5]["quantity"] == 1
    assert medications[5]["unit"] == "hộp"
    assert medications[6]["quantity"] == 60
    assert medications[6]["unit"] == "viên"


def test_inline_numbered_prescription_starts_new_medication() -> None:
    raw_text = """
1. YESOM 40 40mg 63 viên
Uống ngày 1 viên
2. Biseptol 480mg 20 viên
Uống ngày 2 lần
"""

    medications = parse_with_regex(raw_text)["medications"]

    assert [med["productName"] for med in medications] == ["YESOM 40 40mg", "Biseptol 480mg"]
    assert [med["quantity"] for med in medications] == [63, 20]
    assert [med["unit"] for med in medications] == ["viên", "viên"]


def test_sl_quantity_line_is_not_treated_as_medication_name() -> None:
    raw_text = """
1 Citicolin 0,2g
SL: 30 viên
Uống ngày 2 lần
2 Magne B6
SL 20 viên
Uống sáng chiều
"""

    medications = parse_with_regex(raw_text)["medications"]

    assert [med["productName"] for med in medications] == ["Citicolin 0,2g", "Magne B6"]
    assert [med["quantity"] for med in medications] == [30, 20]
    assert all(med["unit"] == "viên" for med in medications)


def test_noisy_single_traditional_candidate_is_not_usable() -> None:
    raw_text = """
03010000099
0301000000099
1
0310000199
SO HS
030100100101
MICH ĐẾ - THỊ THỊ THỊ
031000100125
0
Tran, lần, m
"""
    traditional = {
        "patientName": None,
        "doctorName": None,
        "hospitalName": None,
        "prescriptionDate": None,
        "medications": [
            {
                "productName": "MICH ĐẾ - THỊ THỊ THỊ",
                "quantity": 0,
                "unit": None,
                "dosage": "Tran, lần, m",
                "instructions": "Tran, lần, m",
            }
        ],
        "confidence": "medium",
    }

    quality = score_candidate(traditional, "traditional", True, raw_text)
    merged, merged_quality = merge_candidates(traditional, None, quality, None, raw_text)

    assert "unsafe_hallucination" in quality["criticalFlags"]
    assert quality["usableMedicationCandidate"] is False
    assert merged["medications"] == []
    assert merged_quality["selectedSource"] == "none"
