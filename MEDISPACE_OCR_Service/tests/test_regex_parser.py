from src.services.regex_parser import parse_with_regex
from src.services.quality import merge_candidates, score_candidate
from src.services.vision_extractor import _extract_medications_from_freeform


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


def test_patient_age_label_is_not_treated_as_birth_year() -> None:
    raw_text = """
Họ tên: Nguyễn Văn A
Phái: NamTuổi: 49
"""

    result = parse_with_regex(raw_text)

    assert result["patientAge"] == "49"


def test_parenthesized_number_with_space_starts_medication() -> None:
    raw_text = """
(2 SUCRATE GEL (Sucralfate 1g) 30 gói
Uống trước ăn ngày 2 lần
"""

    medications = parse_with_regex(raw_text)["medications"]

    assert [med["productName"] for med in medications] == ["SUCRATE GEL (Sucralfate 1g)"]
    assert medications[0]["quantity"] == 30
    assert medications[0]["unit"] == "gói"


def test_bhyt_table_layout_without_item_numbers_extracts_medications() -> None:
    raw_text = """
STT
Tên thuốc/hàm lượng
ĐVT
Số lượng
Penicilin(dưới dạng Phenoxymethylpenicilin Kali)
Viên
30
Paracetamol 500mg
Viên
20
Lời dặn:
"""

    medications = parse_with_regex(raw_text)["medications"]

    assert [med["productName"] for med in medications] == [
        "Penicilin(dưới dạng Phenoxymethylpenicilin Kali)",
        "Paracetamol 500mg",
    ]
    assert [med["quantity"] for med in medications] == [30, 20]
    assert [med["unit"] for med in medications] == ["viên", "viên"]


def test_interleaved_unit_quantity_layout_extracts_structured_medications() -> None:
    raw_text = """
Chan doan: Viem da
Thuoc dung:
Tube
1
Lalise Cleanser SRM Da Dau 100g
Rua mat sang, toi.
2
Genfluid
lo
1
Lau sang, toi truoc khi boi thuoc
OSAINE SPF50
tuyp
1
Boi sang va trua ca mat
UPHAXIME cefixim 200MG
VIEN
20
Uong 1 vien sang, 1 vien toi sau an
Ngay 24 thang 7 nam 2017
Bac si dieu tri
"""

    medications = parse_with_regex(raw_text)["medications"]

    assert [med["productName"] for med in medications] == [
        "Lalise Cleanser SRM Da Dau 100g",
        "Genfluid",
        "OSAINE SPF50",
        "UPHAXIME cefixim 200MG",
    ]
    assert [med["quantity"] for med in medications] == [1, 1, 1, 20]
    assert [med["unit"] for med in medications] == ["tube", "l\u1ecd", "tu\u00fdp", "vi\u00ean"]


def test_date_and_order_note_lines_are_not_medications() -> None:
    raw_text = """
DON THUOC
Chan doan
Roi loan TK thuc vat
Sang uong 1 vien, chieu uong 1 vien, sau an
7/4/21: don W
29/10/2020
Bac si chuyen khoa II
"""

    medications = parse_with_regex(raw_text)["medications"]

    assert medications == []


def test_weak_handwriting_noise_lines_are_not_medications() -> None:
    raw_text = """
DON THUOC
Benh nhan:
Chan doan
R8i duan TK Thik van
Cideolu ogur
Nemy
Xo vien ? thi
moss
Sang uong 1 vien, chieu uong 1 vien, sau an
Nemva
2
Sang uong 1 vien, chieu uong 1 vien, sau an
xau vien
Sang uong 1 vien, chieu uong 1 vien, sau an
H/4/21: don W
Ngay 29, thang 4 Nam 21
Bac si chuyen khoa II
"""

    medications = parse_with_regex(raw_text)["medications"]

    assert medications == []


def test_prescription_date_day_is_not_patient_age() -> None:
    raw_text = """
Chan doan: Viem da
Ngay 24 thang 7 nam 2017
Thuoc dung:
1 Paracetamol 500mg 20 vien
"""

    result = parse_with_regex(raw_text)

    assert result["patientAge"] is None


def test_note_containing_prescription_word_does_not_reset_medication_section() -> None:
    raw_text = """
Chan doan: Viem da
Thuoc dung:
Tube
1
Halox
Boi mong dung mun trung ca
Loi dan:
Quy khach vui long mang don thuoc khi kham lai
"""

    medications = parse_with_regex(raw_text)["medications"]

    assert [med["productName"] for med in medications] == ["Halox"]

def test_merge_filters_vision_header_noise_and_pairs_weak_traditional_name() -> None:
    traditional = {
        "medications": [
            {
                "productName": "Hôn",
                "dosage": "3vx3 lần/ ngày (sau ăn)",
                "quantity": None,
                "unit": None,
                "instructions": "3vx3 lần/ ngày (sau ăn)",
            },
            {
                "productName": "JEX (H/30V)",
                "dosage": "Ivx2 lần/ngày (sau ăn), 7. Cồn xoa bóp Viện, xoa lưng đau",
                "quantity": 2,
                "unit": "viên",
                "instructions": "Ivx2 lần/ngày (sau ăn), 7. Cồn xoa bóp Viện, xoa lưng đau",
            },
        ]
    }
    vision = {
        "medications": [
            {"productName": "Danh sách thuốc", "confidence": "low"},
            {"productName": "Seotolac", "confidence": "low"},
            {"productName": "Hàm lượng", "confidence": "low"},
            {"productName": "Số lượng", "confidence": "low"},
            {"productName": "Độc hoạt TKS viên", "confidence": "low"},
        ]
    }

    merged, _quality = merge_candidates(traditional, vision)

    names = [med["productName"] for med in merged["medications"]]
    assert "Hôn" not in names
    assert "Danh sách thuốc" not in names
    assert "Hàm lượng" not in names
    assert "Số lượng" not in names
    assert names == ["Seotolac", "JEX (H/30V)", "Cồn xoa bóp Viện", "Độc hoạt TKS viên"]
    assert merged["medications"][0]["dosage"] == "3vx3 lần/ ngày (sau ăn)"
    assert merged["medications"][0]["reviewReason"] == "weak_traditional_name_replaced"
    assert merged["medications"][1]["dosage"] == "Ivx2 lần/ngày (sau ăn)"
    assert merged["medications"][2]["instructions"] == "xoa lưng đau"
    assert merged["medications"][2]["reviewReason"] == "split_from_embedded_numbered_instruction"

def test_merge_filters_administrative_vision_noise_and_duplicate_names() -> None:
    traditional = {
        "medications": [
            {"productName": "YESOM 40 40mg (Esomeprazol 40mg)", "quantity": 63, "unit": "viên"},
            {"productName": "BIOCID MH 3.542g", "quantity": 21, "unit": "chai"},
        ]
    }
    vision = {
        "medications": [
            {"productName": "Thông tin bệnh nhân", "confidence": "low"},
            {"productName": "Tên", "confidence": "low"},
            {"productName": "Địa chỉ", "confidence": "low"},
            {"productName": "Chẩn đoán", "confidence": "low"},
            {"productName": "YESOM 40 mg", "confidence": "low"},
            {"productName": "BIOCID MH 3.5+2g", "confidence": "low"},
        ]
    }

    merged, _quality = merge_candidates(traditional, vision)

    assert [med["productName"] for med in merged["medications"]] == [
        "YESOM 40 40mg (Esomeprazol 40mg)",
        "BIOCID MH 3.542g",
    ]

def test_merge_filters_metadata_and_date_like_medication_noise() -> None:
    traditional = {
        "medications": [
            {"productName": "Nước Sx", "quantity": 3, "unit": "lọ"},
            {"productName": "Scilin M30 40U/ml", "quantity": 3, "unit": "lọ"},
            {"productName": "Việt Nam", "quantity": 1, "unit": None},
            {"productName": "Metformin(INDFORM)", "quantity": 90, "unit": "viên"},
        ]
    }
    vision = {
        "medications": [
            {"productName": "Ngày khám", "confidence": "low"},
            {"productName": "7/1/21", "confidence": "low"},
        ]
    }

    merged, _quality = merge_candidates(traditional, vision)

    assert [med["productName"] for med in merged["medications"]] == [
        "Scilin M30 40U/ml",
        "Metformin(INDFORM)",
    ]


def test_donthuoc_jpg_raw_ocr_keeps_all_medications_and_age() -> None:
    raw_text = """
SỐ TIBINH
BỆNH THỊ THỊ MINH
Số nò sơn
Phòng Khám: Pich V.107
701bd.1407A7744
0:55
9
đơn thuốc
Ngày kham
Ngày 25 tháng 09 năm 2014
Đối tượng: Thu phí-KTC
HỌ TÊN: TRẦN TRUNG
Phái: NamTuổi: 49
Địa chì: Xã Bảo Bình, Huyện Cẩm Mỹ, Đồng Nai
Chản đoán: Viêm dạ dày cấp khác
1.YESOM 40 40mg (Esomeprazol 40mg)
63
Viên
(2 SUCRATE GEL (Sucralfate 19 (goi)
Ngày uống 3 lần, mỗi lần 1 Viên
Ngày uống 3 lần, mỗi lần 1Gỗi
63
GÓI
3. ARTHUR (Trimebutine 200)
Ngày uống 3 lần, mỗi lần 1 Viên
63
Viên
4,PAZE (magnesi aluminometasilicate ,na
bicarbonate,cao scopolia, alpha amylase,beta
63
Viên
Ceet
amuylase, protease)
Ngày uống 3 lần, mỗi lần 1 Viên
5. BIOCID MH 3.542g (Nhôm Hydroxide 3.542g.
Chai
Magnesi Hydroxide 100ml
21
Ngày uống 3 lần, mỗi lần 1/3 Chai
6, DIGLUMISAN (L-Arginine Hydrocloride 1000mg)
ỐNG
Ngày uống 3 lần, mỗi lần 1 ONG
63
Cộng khoản: 6
25/09/2014
Bác sĩ điều trị
Lời dăn
Tái khám: 16/10/2014
Nguyễn Văn Hùng
"""

    result = parse_with_regex(raw_text)
    medications = result["medications"]

    assert result["patientAge"] == "49"
    assert [med["productName"] for med in medications] == [
        "YESOM 40 40mg (Esomeprazol 40mg)",
        "SUCRATE GEL (Sucralfate 19 (goi)",
        "ARTHUR (Trimebutine 200)",
        "PAZE (magnesi aluminometasilicate ,na bicarbonate,cao scopolia, alpha amylase,beta amuylase, protease)",
        "BIOCID MH 3.542g (Nhôm Hydroxide 3.542g. Magnesi Hydroxide 100ml",
        "DIGLUMISAN (L-Arginine Hydrocloride 1000mg)",
    ]
    assert [med["quantity"] for med in medications] == [63, 63, 63, 63, 21, 63]
    assert [med["unit"] for med in medications] == ["viên", "gói", "viên", "viên", "chai", "ống"]

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

def test_vision_names_are_used_when_traditional_handwriting_ocr_is_weak() -> None:
    raw_text = """
Ms: 17D/BV-01
BỆNH VIỆN BỆNH NHIỆT ĐỚI
ĐƠN THUỐC
Btx benks Enux
Keranoun
lebeloles
"""
    traditional = {
        "patientName": None,
        "doctorName": None,
        "hospitalName": None,
        "prescriptionDate": None,
        "medications": [
            {
                "productName": "Btx benks Enux",
                "quantity": None,
                "unit": None,
                "dosage": "mỹ ở ngày",
                "instructions": "mỹ ở ngày",
            }
        ],
        "confidence": "low",
    }
    vision = {
        "patientName": None,
        "doctorName": None,
        "hospitalName": None,
        "prescriptionDate": None,
        "medications": [
            {"productName": "Etex bene", "quantity": None, "unit": None, "dosage": "", "instructions": ""},
            {"productName": "Kerarian", "quantity": None, "unit": None, "dosage": "", "instructions": ""},
            {"productName": "Tebacol", "quantity": None, "unit": None, "dosage": "", "instructions": ""},
        ],
        "confidence": "medium",
    }

    traditional_quality = score_candidate(traditional, "traditional", True, raw_text)
    vision_quality = score_candidate(vision, "vision", False)
    merged, merged_quality = merge_candidates(traditional, vision, traditional_quality, vision_quality, raw_text)

    assert traditional_quality["usableMedicationCandidate"] is False
    assert vision_quality["usableMedicationCandidate"] is True
    assert vision_quality["usableByVisionNamesOnly"] is True
    assert merged_quality["selectedSource"] == "vision"
    assert [med["productName"] for med in merged["medications"]] == ["Etex bene", "Kerarian", "Tebacol"]
    assert all(med["needsReview"] is True for med in merged["medications"])

def test_vision_freeform_fallback_extracts_medication_names() -> None:
    reading = """
2. Các thuốc được kê đơn:

* Etex bene: hỗ trợ tiêu hóa.
* Kerarian: dùng theo hướng dẫn bác sĩ.
* Tebacol: số lượng có thể là 15ml.

3. Liều lượng và Cách dùng:
Các loại thuốc này được kê với liều lượng cụ thể.
"""

    result = _extract_medications_from_freeform(reading)

    assert [med["productName"] for med in result["medications"]] == ["Etex bene", "Kerarian", "Tebacol"]
    assert all(med["needsReview"] is True for med in result["medications"])

def test_vision_freeform_fallback_ignores_notes_and_trims_usage_tail() -> None:
    reading = """
* Etex beneules 80mg x nổ/4 ngày
* Kerian 60mg x nổ/2 ngày
* Tebecerol 60mg x nổ/2 ngày
* (Lưu ý: uống sau ăn)
* 7/1/21
* 22/2
"""

    result = _extract_medications_from_freeform(reading)

    assert [med["productName"] for med in result["medications"]] == [
        "Etex beneules 80mg",
        "Kerian 60mg",
        "Tebecerol 60mg",
    ]
