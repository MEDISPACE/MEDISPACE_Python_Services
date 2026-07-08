import re
import unicodedata
from datetime import datetime
from typing import Dict, Any, List, Optional


def parse_with_regex(raw_text: str) -> Dict[str, Any]:
    """Trích xuất thông tin đơn thuốc bằng regex (v3 - fixed)."""
    text = _normalize(raw_text)

    result = {
        "patientName":      _extract_patient_name(text),
        "patientAge":       _extract_patient_age(text),
        "patientGender":    _extract_patient_gender(text),
        "phoneNumber":      _extract_phone_number(text),
        "doctorName":       _extract_doctor_name(text),
        "hospitalName":     _extract_hospital_name(text),
        "prescriptionDate": _extract_date(text),
        "diagnosis":        _extract_diagnosis(text),
        "medications":      _extract_medications(text),
        "specialNotes":     _extract_special_notes(text),
        "confidence":       "medium",
    }

    score = sum(
        1 for k, v in result.items()
        if k not in ("confidence", "_regex_score") and v and v != []
    )
    result["_regex_score"] = score
    return result


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    # Chuẩn hóa Unicode về NFC trước — tránh lỗi NFD (a + combining hook ≠ ả NFC)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\r\n?', '\n', text)
    return text.strip()

def _clean(match: Optional[re.Match], group: int = 1) -> Optional[str]:
    if not match:
        return None
    val = match.group(group).strip()
    val = re.split(r'[|\n]', val)[0].strip()
    return val if len(val) > 0 else None

def _ascii_lower(value: str) -> str:
    """Return lowercase text without Vietnamese accents for OCR-tolerant matching."""
    text = unicodedata.normalize("NFD", value or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    text = text.replace("đ", "d").replace("Đ", "D")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def _canonical_unit(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = _ascii_lower(value)
    compact = key.replace(" ", "")
    if re.fullmatch(r"vi\s*n|vien|vi.?n|v\s*n", key) or compact in {"vin", "vn"}:
        return "viên"
    if re.fullmatch(r"go\s*i|goi|go.?i|g\s*i", key) or compact == "gi":
        return "gói"
    if key in {"lo", "l"} or compact in {"lo", "l"}:
        return "lọ"
    if re.fullmatch(r"o\s*ng|ong|o.?ng", key) or compact == "ng":
        return "ống"
    if re.fullmatch(r"ho\s*p|hop|ho.?p|h\s*p", key) or compact == "hp":
        return "hộp"
    if re.fullmatch(r"tu\s*p|tuyp|tu.?p|t\s*p", key) or key == "tube":
        return "tuýp" if key != "tube" else "tube"
    if key in {"chai"}:
        return "chai"
    if key in {"cai"}:
        return "cái"
    return None

def _is_unit_text(value: str) -> bool:
    return _canonical_unit(value) is not None

def _is_dose_text(value: str) -> bool:
    key = _ascii_lower(value)
    dose_patterns = [
        r"\bu\s*o?\s*ng\b", r"\bb\s*o?\s*i\b", r"\bxoa\b", r"\bt\s*a?\s*m\b", r"\bd\s*u?\s*ng\b", r"\bng\s*a?\s*y\b",
        r"\bs\s*a?\s*ng\b", r"\btr\s*u?\s*a\b", r"\bch\s*i?\s*eu\b", r"\bt\s*o?\s*i\b", r"\bl\s*a?\s*n\b", r"\btu\s*a?\s*n\b",
        r"\bsau\s*an\b", r"\btruoc\s*an\b", r"\bcach\s*ng\s*ay\b", r"\bnho\b", r"\bxit\b", r"\bngam\b", r"\bpha\b", r"\bthoa\b",
    ]
    dose_tokens = {
        "uong", "ung", "boi", "xoa", "tam", "dung", "dng",
        "ngay", "ngy", "sang", "sng", "trua", "tra", "chieu", "chiu", "toi",
        "lan", "tuan", "tun", "thoa",
    }
    words = set(key.split())
    return any(re.search(pattern, key) for pattern in dose_patterns) or bool(words & dose_tokens)

def _prescription_medication_lines(lines: List[str]) -> List[str]:
    """Return the part of OCR text where medication rows normally start."""
    diagnosis_anchors = ("chan doan", "chuan doan", "diagnosis")
    medication_anchors = ("thuoc dieu tri", "thuoc dung", "toa thuoc", "don thuoc")
    header_noise = {"don thuoc", "toa thuoc", "thuoc"}
    start_index = 0
    for idx, line in enumerate(lines):
        key = _ascii_lower(line)
        if key == "cd" or any(anchor in key for anchor in diagnosis_anchors):
            start_index = idx + 1
            continue
        if any(key == anchor or key.startswith(anchor) for anchor in medication_anchors):
            start_index = idx + 1

    trimmed = lines[start_index:]
    while trimmed and _ascii_lower(trimmed[0]) in header_noise:
        trimmed = trimmed[1:]
    return trimmed

def _is_stop_text(value: str) -> bool:
    key = _ascii_lower(value)
    return key.startswith((
        "loi dan", "ghi chu", "dan do", "chu y", "luu y", "tai kham", "kham lai",
        "ngay hen", "bac sy", "bac si", "bs", "cong khoan", "loi khuyen",
        "ky ten", "nguoi ke don",
    ))

def _is_item_number_text(value: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}[\.)]?", value.strip()))

def _is_noise_line(value: str) -> bool:
    key = _ascii_lower(value)
    return key in {"x", "xx", "sl", "so luong", "don vi", "dvt", "thuoc", "ten thuoc", "lieu", "lieu dung", "*"}

def _looks_like_admin_or_date_line(value: str) -> bool:
    key = _ascii_lower(value)
    if re.search(r"\b\d{1,2}\s*/\s*\d{1,2}(?:\s*/\s*\d{2,4})?\b", value):
        return True
    if re.search(r"\b[A-Za-z]\s*/\s*\d{1,2}\s*/\s*\d{2,4}\b", value):
        return True
    if re.search(r"\b\d{1,2}\s*(?:thang|month)\s*\d{1,2}\b", key):
        return True
    return key.startswith((
        "ngay", "hen", "tai kham", "kham lai", "don ", "bac si", "bac sy",
        "bs ", "nguoi ke", "benh nhan", "chan doan", "chuan doan", "dia chi", "dien thoai",
    ))

def _is_probable_med_name(value: str) -> bool:
    stripped = value.strip(" :-")
    key = _ascii_lower(stripped)
    if len(stripped) < 3 or not re.search(r"[A-Za-zÀ-ỹ]", stripped):
        return False
    if _looks_like_admin_or_date_line(stripped):
        return False
    if _is_item_number_text(stripped) or _is_noise_line(stripped) or _is_unit_text(stripped):
        return False
    if key.startswith(("sl ", "so luong", "don vi")):
        return False
    if _is_dose_text(stripped) or _is_stop_text(stripped):
        return False
    if key.startswith(("benh vien", "phong kham", "chan doan", "ho ten", "dia chi", "dien thoai", "ngay")):
        return False
    return True

def _has_strong_med_name_signal(value: str) -> bool:
    stripped = value.strip(" :-")
    key = _ascii_lower(stripped)
    if re.search(r"\b\d+(?:[\.,]\d+)?\s*(?:mg|mcg|g|ml|iu)\b", key):
        return True
    if re.search(r"[A-Z]{2,}", stripped):
        return True
    if len(key.split()) >= 2 and not key.endswith(" vien"):
        return True
    return len(stripped) >= 6 and bool(re.search(r"[A-Za-zÀ-ỹ]", stripped))

def _is_person_name(s: str) -> bool:
    """Kiểm tra chuỗi có phải họ tên người không."""
    s = s.strip()
    if len(s) < 4 or len(s) > 60:
        return False
    if re.search(r'\d', s):          # có số → không phải tên
        return False
    if re.search(r'[:\(\)\[\]]', s): # có ký tự đặc biệt → không phải tên
        return False
    # Phải có ít nhất 2 từ + bắt đầu bằng chữ hoa
    words = s.split()
    if len(words) < 2:
        return False
    return True

def _format_medical_titles(text: str) -> str:
    """Đảm bảo các danh xưng y khoa như BS.CKI, ThS.BS, CKI được viết hoa đúng chuẩn."""
    if not text:
        return text
    
    res = text
    # Các chuyên khoa
    res = re.sub(r'\b(?:Bs\.?\s*Cki|Bác\s*Sĩ\s*Cki)\b', 'BS.CKI', res, flags=re.IGNORECASE)
    res = re.sub(r'\b(?:Bs\.?\s*Ckii|Bác\s*Sĩ\s*Ckii)\b', 'BS.CKII', res, flags=re.IGNORECASE)
    res = re.sub(r'\bCki\b', 'CKI', res, flags=re.IGNORECASE)
    res = re.sub(r'\bCkii\b', 'CKII', res, flags=re.IGNORECASE)
    
    # Học hàm, học vị
    res = re.sub(r'\bThs\.?\s*Bs\.?\b', 'ThS.BS.', res, flags=re.IGNORECASE)
    res = re.sub(r'\bThs\.?\b', 'ThS.', res, flags=re.IGNORECASE)
    res = re.sub(r'\bTs\.?\s*Bs\.?\b', 'TS.BS.', res, flags=re.IGNORECASE)
    res = re.sub(r'\bPgs\.?\s*Ts\.?\b', 'PGS.TS.', res, flags=re.IGNORECASE)
    res = re.sub(r'\bGs\.?\s*Ts\.?\b', 'GS.TS.', res, flags=re.IGNORECASE)
    
    # Đơn lẻ
    res = re.sub(r'\bBs\.?(?!\w)', 'BS.', res, flags=re.IGNORECASE)
    res = re.sub(r'\bTs\.?(?!\w)', 'TS.', res, flags=re.IGNORECASE)
    
    # Chuẩn hoá khoảng trắng / dấu chấm
    res = res.replace("BS. CKI", "BS.CKI").replace("BS. CKII", "BS.CKII")
    res = res.replace("..", ".")
    
    return res

# ─── Tên bệnh nhân ────────────────────────────────────────────────────────────

def _extract_patient_name(text: str) -> Optional[str]:
    patterns = [
        # "Họ tên bệnh nhân: Nguyễn Văn A"
        r"Họ\s*(?:và\s*)?tên\s*(?:bệnh\s*nhân|BN)?\s*[:\-]?\s*([^\n\d\|]{3,50})",
        # "HỌ TÊN: TRẦN TRUNG"
        r"HỌ\s*TÊN\s*[:\-]?\s*([^\n\d\|]{3,50})",
        # "Tên BN: ..."
        r"Tên\s*(?:BN|bệnh\s*nhân)\s*[:\-]?\s*([^\n\d\|]{3,50})",
    ]
    bad_suffixes = re.compile(
        r'\s*(?:Và\s*)?[Tt]uổi.*|'
        r'\s+[Pp]hái.*|'
        r'\s+[Gg]iới.*|'
        r'\s*,.*$',
        re.IGNORECASE
    )
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        val = _clean(m)
        if val:
            val = bad_suffixes.sub('', val).strip()
            if len(val) >= 3:
                return val.title()
    return None


# ─── Tuổi ─────────────────────────────────────────────────────────────────────

def _extract_patient_age(text: str) -> Optional[str]:
    patterns = [
        r"(?:Và\s*)?[Tt]uổi\s*[:\-]?\s*0*([1-9]\d{0,2})(?!\d)", # Tránh SĐT dạng 090...
        r"(?<!\d)(\d{1,3})\s*(?:tuổi|tháng)(?!\s*[:\-])", # Exp: "3 tuổi" hoặc "5 tháng"
        r"[Nn]ăm\s*sinh\s*[:\-]\s*(19\d{2}|20\d{2})",
        r"\bSN\s*[:\-]\s*(19\d{2}|20\d{2})",
    ]
    for i, pat in enumerate(patterns):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            if i == 1 and _ascii_lower(text[max(0, m.start() - 12):m.start()]).endswith("ngay"):
                continue
            val = m.group(1)
            if i >= 2:
                try:
                    return str(datetime.now().year - int(val))
                except Exception:
                    pass
            return val
    return None


# ─── Số điện thoại ────────────────────────────────────────────────────────────

def _extract_phone_number(text: str) -> Optional[str]:
    """Tìm số điện thoại liên hệ."""
    PAT = r"(0[3|5|7|8|9][0-9\s\.\-]{8,12})"
    
    # Ưu tiên tìm sau các nhãn
    for pat in [
        r"(?:SĐ?T|Điện\s*thoại|Tel|Phone)\s*[:\-]?\s*" + PAT,
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            clean_num = re.sub(r'\D', '', m.group(1))
            if len(clean_num) == 10: return clean_num

    # Nếu không có nhãn thì kiếm đại SĐT thỏa mã mạng VN
    m = re.search(PAT, text)
    if m:
        clean_num = re.sub(r'\D', '', m.group(1))
        if len(clean_num) == 10: return clean_num
        
    return None


# ─── Giới tính ────────────────────────────────────────────────────────────────

def _extract_patient_gender(text: str) -> Optional[str]:
    """
    Trả về 'male'/'female' để khớp với FE Select values.
    """
    patterns = [
        # "Giới tính: Nam" / "Phái: Nữ"
        r"(?:Giới\s*tính|Phái|GT)\s*[:\-]\s*(Nam|Nữ|Nu|Male|Female|BD)",
        # "NamTuổi" (dính liền — OCR không có khoảng trắng)
        r"(Nam|Nữ|Nu)(?=Tuổi|\d)",
        # Standalone cạnh dấu phẩy hoặc cuối dòng
        r"(?:^|[\s,\|])(Nam|Nữ)(?:[,\s]|$)",
    ]
    to_standard = {
        "nam": "male",
        "male": "male",
        "nữ": "female",
        "nu": "female",
        "female": "female",
        "bd": "female",
    }
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            # Lấy group có giá trị cuối
            raw = m.group(m.lastindex).strip().lower()
            std = to_standard.get(raw)
            if std:
                return std
    return None


# ─── Bác sĩ ───────────────────────────────────────────────────────────────────

def _extract_doctor_name(text: str) -> Optional[str]:
    """
    Tìm tên bác sĩ: thường nằm SAU nhãn 'Bác sĩ điều trị'
    nhưng có thể cách vài dòng (tên nằm dưới chữ ký).
    Strategy: tìm nhãn → quét 5 dòng tiếp theo → chọn dòng có dạng họ tên.
    """
    lines = text.split('\n')

    # Nhãn gạch đầu dòng (thế này sẽ k bẫy được "PHÒNG KHÁM NỘI BS.CKI ĐÌNH CHI")
    label_pat = re.compile(
        r"^(?:Bác\s*sĩ|BS\.?)\s*(?:khám(?:\s*b[eêệ]nh)?|điều\s*trị|kê\s*đơn)\s*[:\-]?\s*$",
        re.IGNORECASE
    )
    skip_pats = [
        re.compile(r"(?:Lời|Ghi|Dặn|Tái|Chú|Lưu)", re.IGNORECASE),
        re.compile(r"[:\-]"),          # dòng nhãn khác
        re.compile(r"^\d"),            # bắt đầu số
        re.compile(r"Nguyễn|Trần|Lê|Phạm|Võ|Đặng|Bùi|Đỗ|Hồ|Ngô|Dương|Phan",
                   re.IGNORECASE),     # danh sách họ VN phổ biến
    ]

    for i, line in enumerate(lines):
        if label_pat.search(line.strip()):
            # Quét 6 dòng tiếp theo
            for j in range(i + 1, min(i + 7, len(lines))):
                candidate = lines[j].strip()
                if not candidate:
                    continue
                # Bỏ qua dòng nhãn / số
                if any(p.search(candidate) for p in skip_pats[:3]):
                    continue
                # Ưu tiên dòng bắt đầu bằng họ VN phổ biến
                if _is_person_name(candidate) and candidate.lower() not in ["đơn thuốc", "phòng khám"]:
                    return _format_medical_titles(candidate.title())
            break

    # Fallback: pattern inline
    inline_pats = [
        r"(?:Bác\s*sĩ|BS\.?)\s*[:\-]\s*([^\n\d\|]{3,50})",
        r"(?:Người\s*kê\s*đơn|BS\s*kê\s*đơn)\s*[:\-]\s*([^\n\d\|]{3,50})",
        # standalone line that starts with Bs "Bs Chi Đinh"
        r"^(?:BS\.?|Bác\s*sĩ)\s+([A-ZÀ-Ỹ][a-zà-ỹA-ZÀ-Ỹ\s]{2,40})$",
        # inline inside clinic name like "BS.CKI ĐÌNH CHI"
        r"(?:BS\.?(?:CKI|CKII)?|Bác\s*sĩ\.)\s*([A-ZÀ-Ỹ][A-ZÀ-Ỹ\s]{2,40})"
    ]
    bad = {"điều trị", "khám", "kê đơn", ""}
    for pat in inline_pats:
        m = re.search(pat, text, re.IGNORECASE)
        val = _clean(m)
        if val and val.lower() not in bad and len(val) >= 3:
            return _format_medical_titles(val.title())

    return None


# ─── Bệnh viện / Phòng khám ───────────────────────────────────────────────────

def _extract_hospital_name(text: str) -> Optional[str]:
    """
    Tên BV/PK phải chứa từ khoá 'bệnh viện'/'phòng khám'... trong kết quả.
    - Nếu có label inline (e.g. "Phòng Khám: Dịch Vụ 109"), ghép lại để giữ keyword.
    - Nếu cả dòng đã chứa keyword (e.g. "BỆNH VIỆN BÌNH DÂN"), trả về cả dòng.
    """
    KEYWORD_PAT = re.compile(
        r"(?:bệnh\s*viện|ph[oò]ng\s*khám|trung\s*tâm\s*y\s*tế|cơ\s*sở\s*y\s*tế|"
        r"clinic|hospital|pkđk|medic)",
        re.IGNORECASE
    )

    # 1. Inline label: "Phòng Khám: Dịch Vụ 102"
    m = re.search(
        r"((?:Bệnh\s*viện|Ph[oò]ng\s*khám|Cơ\s*sở\s*y\s*tế|Trung\s*tâm\s*y\s*tế)"
        r"\s*[:\-]\s*[^\n\|]{2,60})",
        text, re.IGNORECASE
    )
    if m:
        candidate = re.sub(r'\s*[:\-]\s*', ' ', m.group(1)).strip()
        candidate = re.sub(r'\s+', ' ', candidate)
        if len(candidate) >= 5 and not re.fullmatch(r'[\d\s\-\/]+', candidate):
            return _format_medical_titles(candidate.title())

    # 2. Quét 8 dòng đầu — chỉ lấy dòng CÓ CHỨA keyword trong nội dung
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:8]:
        if KEYWORD_PAT.search(line):
            # Cần skip những dòng social media hoặc địa chỉ
            if re.search(r"Fb:|Zalo|Facebook|Web|ĐC:|Địa\s*chỉ|SĐT", line, re.IGNORECASE):
                continue
            if len(line) >= 5 and not re.fullmatch(r'[\d\s\-\/]+', line):
                return _format_medical_titles(line.title())

    return None


# ─── Ngày kê đơn ──────────────────────────────────────────────────────────────

def _extract_date(text: str) -> Optional[str]:
    # Ưu tiên pattern "Ngày dd tháng mm năm yyyy"
    m = re.search(
        r"[Nn]gày\s*(\d{1,2})\s*[Tt]háng\s*(\d{1,2})\s*[Nn]ăm\s*(\d{4})",
        text
    )
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        mo_int = int(mo)
        if 1 <= mo_int <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"

    # "dd/mm/yyyy" hoặc "dd-mm-yyyy"
    # Tìm tất cả & chọn hợp lệ
    for m in re.finditer(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", text):
        d, mo, y = m.group(1), m.group(2), m.group(3)
        mo_int = int(mo)
        if 1 <= mo_int <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"

    return None


# ─── Chẩn đoán ────────────────────────────────────────────────────────────────

def _extract_diagnosis(text: str) -> Optional[str]:
    """
    Chấp nhận mọi biến thể OCR của 'Chẩn đoán':
    - 'Chẩn đoán' (chuẩn)
    - 'Chản đoán' (OCR lỗi ả→ẩ)
    - 'Chần đoán' (OCR lỗi khác)
    - 'Chan doan' (mất dấu hoàn toàn)
    NFD/NFC: đã xử lý bởi _normalize() ở đầu pipeline.
    """
    # Dùng ký tự rộng nhất: sau 'Ch' có thể là bất kỳ ký tự nào trước 'n đoán'
    patterns = [
        # Chuẩn: Chẩn đoán / Chản đoán / Chần đoán (tất cả biến thể có dấu)
        r"Ch[\wÀ-ỹ]n\s*[đĐ]oán\s*[:\-]\s*(.+)",
        # Không dấu hoặc OCR sai hoàn toàn
        r"[Cc]han\s*[Dd]oan\s*[:\-]\s*(.+)",
        # Viết tắt
        r"(?:CĐ|CD)\s*[:\-]\s*(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        val = _clean(m)
        if val and len(val) >= 4:
            return val
    return None


# ─── Ghi chú ──────────────────────────────────────────────────────────────────

def _extract_special_notes(text: str) -> Optional[str]:
    m = re.search(
        r"(?:Lời\s*dặn|Lời\s*dăn|Ghi\s*chú|Dặn\s*dò|Chú\s*ý|Lưu\s*ý)\s*[:\-]?\s*(.+)",
        text, re.IGNORECASE | re.DOTALL
    )
    if not m:
        return None
    val = m.group(1).strip()
    # Chỉ lấy tới dòng tiếp theo có nội dung thực
    val = re.split(r'\n(?=\S)', val)[0].strip()
    return val[:300] if val else None


# ─── Danh sách thuốc ──────────────────────────────────────────────────────────

def _extract_medications(text: str) -> List[Dict[str, Any]]:
    """
    Phân tích danh sách thuốc.  Logic cốt lõi:
    - pending_qty: first-wins  → số ĐẦUTIÊN trên 1 dòng độc lập có giá trị, không ghi đè
    - unit: last-wins          → đơn vị CUỐI CÙNG gặp được mới là đúng (xử lý OCR interleave)
    - Case 3 (qty_inline): CHỈ áp dụng trên dòng KHÔNG phải liều dùng
      (tránh bắt "1 Viên" trong "mỗi lần 1 Viên" thành số lượng)
    """
    medications: List[Dict[str, Any]] = []
    lines = _prescription_medication_lines([l.strip() for l in text.split('\n') if l.strip()])
    table_items = _extract_table_layout_medications(lines)
    numbered_items = _extract_numbered_line_medications_v2(lines) or _extract_numbered_line_medications(lines)
    interleaved_items = _extract_interleaved_layout_medications(lines)
    separated_items = table_items or numbered_items
    numbered_has_runaway_dosage = any(len(str(item.get("dosage") or "")) > 180 for item in numbered_items)
    if interleaved_items and (
        not numbered_items
        or (len(numbered_items) < 3 and len(interleaved_items) >= len(numbered_items) + 2)
        or (numbered_has_runaway_dosage and len(interleaved_items) >= len(numbered_items) + 3)
    ):
        separated_items = interleaved_items
    if separated_items:
        return separated_items

    # ── Patterns ──────────────────────────────────────────────────────────────

    med_header_pat = re.compile(
        r"^[\s\(\[\{]*(\d{1,2})[\.\)\,\-\/]\s*"
        r"([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚẮẶẤẦẾỆỈỊỌỘỢỰỨ]"
        r"[A-Za-zÀ-ỹ0-9\s\.\,\-\+\(\)\/]{2,})",
    )

    UNITS   = r"[Vv]iên|[Gg][óoÓO]i|GÓI|[Cc]hai|CHAI|[Ốốoo]ng|ÔNG|[Ll]ọ|[Tt]uýp|[Hh]ộp"
    qty_inline_pat = re.compile(rf"(\d+)\s*({UNITS})", re.IGNORECASE)
    unit_only_pat  = re.compile(rf"^({UNITS})$",       re.IGNORECASE)
    multi_num_pat  = re.compile(r"^(\d+)(?:\s+(\d+))?$")  # "63" hoặc "21 6"
    dose_pat       = re.compile(
        r"[Nn]gày|[Ss]áng|[Tt]rưa|[Cc]hiều|[Tt]ối|[Uu]ống|[Dd]ùng|lần/ngày",
        re.IGNORECASE
    )
    stop_labels    = re.compile(
        r"^(?:Cộng\s*khoản|Bác\s*sĩ|Lời\s*d[ặă]n|Tái\s*khám|Ghi\s*chú|Khám\s*lại)",
        re.IGNORECASE
    )

    # ── State ─────────────────────────────────────────────────────────────────

    current_med: Optional[Dict] = None
    pending_qty: Optional[int]  = None   # số chờ đơn vị; first-wins
    in_med_section: bool        = False

    def flush(med: Optional[Dict]) -> None:
        if med and med["productName"] and len(med["productName"]) >= 4:
            if not re.fullmatch(r'[\d\s\.\-]+', med["productName"]):
                medications.append(med)

    # ── Main loop ─────────────────────────────────────────────────────────────

    for line in lines:
        if in_med_section and stop_labels.match(line):
            break

        if med_header_pat.match(line):
            flush(current_med)
            in_med_section = True
            pending_qty    = None
            hdr = med_header_pat.match(line)
            raw_name = hdr.group(2).strip()

            # Qty+unit embedded in name?
            q = qty_inline_pat.search(raw_name)
            qty, unit = None, None
            if q:
                try:    qty = int(q.group(1))
                except: pass
                unit     = q.group(2)
                raw_name = raw_name[:q.start()].strip()

            current_med = {
                "productName":  raw_name,
                "dosage":       None,
                "quantity":     qty,
                "unit":         unit,
                "instructions": None
            }
            continue

        if current_med is None:
            continue

        is_dose = bool(dose_pat.search(line))

        # ── Case 1: standalone unit ────────────────────────────────────────
        u_match = unit_only_pat.match(line)
        if u_match:
            unit_val = u_match.group(1)
            if pending_qty is not None and current_med["quantity"] is None:
                # Lần đầu: gán pending_qty + unit này
                current_med["quantity"] = pending_qty
                current_med["unit"]     = unit_val
            elif current_med["quantity"] is not None:
                # Qty đã có → chỉ cập nhật unit (last-wins: đơn vị sau đúng hơn)
                current_med["unit"] = unit_val
            pending_qty = None
            continue

        # ── Case 2: standalone number (first-wins → không ghi đè pending) ──
        n_match = multi_num_pat.match(line)
        if n_match:
            if pending_qty is None:           # ← first-wins
                n1 = int(n_match.group(1))
                n2 = int(n_match.group(2)) if n_match.group(2) else None
                pending_qty = max(n1, n2) if n2 else n1
            continue

        # Reset pending CHỈ khi gặp dòng nội dung bình thường (không phải dose)
        if not is_dose:
            pending_qty = None

        # ── Case 3: qty+unit cùng dòng — CHỈ cho dòng KHÔNG phải liều dùng ──
        # (tránh bắt "mỗi lần 1 Viên" → qty=1 thay vì qty thực 63)
        if not is_dose and current_med["quantity"] is None:
            q = qty_inline_pat.search(line)
            if q:
                try:    current_med["quantity"] = int(q.group(1))
                except: pass
                current_med["unit"] = q.group(2)

            # ── Case 4: liều dùng — chỉ lấy dòng ĐẦU TIÊN ────────────────────
        if is_dose and current_med["dosage"] is None:
            # Sửa các lỗi chính tả OCR thường gặp trong liều dùng
            dose_str = line.replace("Gỗi", "Gói").replace("Gỗ", "Gói").replace("1Gói", "1 Gói").replace("ONG", "ÔNG")
            current_med["dosage"]       = dose_str
            current_med["instructions"] = dose_str

    flush(current_med)

    # ── Dedup & filter ────────────────────────────────────────────────────────
    seen, result = set(), []
    for m in medications:
        key = m["productName"][:20].lower()
        if key not in seen:
            seen.add(key)
            result.append(m)

    return result

def _extract_numbered_line_medications(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse common OCR layout where item number, drug name, quantity, unit are split across lines."""
    units_pattern = re.compile(r"^(vi.?n|go.?i|chai|o.?ng|lo|l.?|tu.?p|tube|ho.?p)$", re.IGNORECASE)
    stop_pattern = re.compile(
        r"^(?:lời\s*dặn|loi\s*dan|ghi\s*chú|ghi\s*chu|khám\s*lại|kham\s*lai|ngày\s*hẹn|ngay\s*hen|bác\s*sỹ|bac\s*sy|bác\s*sĩ|bac\s*si)",
        re.IGNORECASE,
    )
    dose_pattern = re.compile(
        r"(uống|uong|bôi|boi|xoa|tắm|tam|dùng|dung|ngày|ngay|sáng|sang|trưa|trua|chiều|chieu|tối|toi|lần|lan|tuần|tuan|sau\s*ăn|sau\s*an)",
        re.IGNORECASE,
    )
    ignore_name_pattern = re.compile(
        r"^(?:x|sl|số\s*lượng|so\s*luong|đơn\s*vị|don\s*vi|toa\s*thuốc|don\s*thuốc|đơn\s*thuốc)$",
        re.IGNORECASE,
    )

    def is_item_number(value: str) -> bool:
        return bool(re.fullmatch(r"\d{1,2}[\.)]?", value.strip()))

    def clean_number(value: str) -> Optional[int]:
        if re.fullmatch(r"\d{1,4}", value.strip()):
            return int(value.strip())
        return None

    def is_probable_name(value: str) -> bool:
        stripped = value.strip()
        if len(stripped) < 3:
            return False
        if _looks_like_admin_or_date_line(stripped):
            return False
        if is_item_number(stripped) or ignore_name_pattern.match(stripped) or units_pattern.match(stripped):
            return False
        if dose_pattern.search(stripped):
            return False
        return bool(re.search(r"[A-Za-zÀ-ỹ]", stripped))

    medications: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        if stop_pattern.match(lines[i]):
            break
        if not is_item_number(lines[i]):
            i += 1
            continue

        j = i + 1
        while j < len(lines) and lines[j].lower() in {"x", "xx", "*"}:
            j += 1
        if j >= len(lines) or not is_probable_name(lines[j]):
            i += 1
            continue

        name = lines[j].strip(" :-")
        med = {
            "productName": name,
            "dosage": None,
            "quantity": None,
            "unit": None,
            "instructions": None,
        }

        k = j + 1
        while k < len(lines):
            current = lines[k].strip()
            next_line = lines[k + 1].strip() if k + 1 < len(lines) else ""
            if stop_pattern.match(current) or (is_item_number(current) and is_probable_name(next_line)):
                break
            if current.lower() in {"x", "xx", "*"}:
                k += 1
                continue
            quantity = clean_number(current)
            if quantity is not None and med["quantity"] is None:
                if quantity <= 20 and _is_dose_text(next_line) and not _is_unit_text(next_line):
                    k += 1
                    continue
                med["quantity"] = quantity
                k += 1
                continue
            if units_pattern.match(current):
                med["unit"] = current
                k += 1
                continue
            if dose_pattern.search(current):
                existing = med.get("dosage")
                med["dosage"] = f"{existing}, {current}" if existing else current
                med["instructions"] = med["dosage"]
            k += 1

        has_detail = med.get("quantity") is not None or med.get("unit")
        if has_detail or (med.get("dosage") and _has_strong_med_name_signal(str(med.get("productName") or ""))):
            medications.append(med)
        i = k

    return medications if len(medications) >= 1 else []

def _extract_numbered_line_medications_v2(lines: List[str]) -> List[Dict[str, Any]]:
    """OCR-tolerant parser for numbered prescriptions split across multiple lines.

    Handles layouts like:
      1 / Drug name / x / 30 / vien / Uong...
      1. Drug name 30 vien / Uong...
      1 Drug name / SL: 30 vien / Uong...
    """
    medications: List[Dict[str, Any]] = []
    i = 0

    def clean_number(value: str) -> Optional[int]:
        match = re.search(r"(?<!\d)(\d{1,4})(?!\d)", value.strip())
        return int(match.group(1)) if match else None

    def parse_inline_qty_unit(value: str) -> tuple[Optional[int], Optional[str]]:
        normalized = value.strip()
        for match in re.finditer(r"(?<!\d)(\d{1,4})\s*([A-Za-zÀ-ỹ?\.]{1,12})(?!\w)", normalized):
            unit = _canonical_unit(match.group(2))
            if unit:
                return int(match.group(1)), unit
        return None, None

    def parse_inline_header(value: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
        match = re.match(r"^\s*[\(\[\{]?\s*\d{1,2}(?:[\.)\,\-\/]\s*|\s+)(.+?)\s*$", value)
        if not match:
            return None, None, None
        remainder = match.group(1).strip(" :-")
        qty, unit = parse_inline_qty_unit(remainder)
        if qty is not None and unit:
            remainder = re.sub(r"\s*\d{1,4}\s*[A-Za-zÀ-ỹ?\.]{1,12}\s*$", "", remainder).strip(" :-")
        return (remainder if _is_probable_med_name(remainder) else None), qty, unit

    while i < len(lines):
        line = lines[i].strip()
        if _is_stop_text(line):
            break

        name: Optional[str] = None
        quantity: Optional[int] = None
        unit: Optional[str] = None

        inline_name, inline_qty, inline_unit = parse_inline_header(line)
        if inline_name:
            name, quantity, unit = inline_name, inline_qty, inline_unit
            j = i + 1
        elif _is_item_number_text(line):
            j = i + 1
            while j < len(lines) and _is_noise_line(lines[j]):
                j += 1
            if j >= len(lines) or not _is_probable_med_name(lines[j]):
                i += 1
                continue
            name = lines[j].strip(" :-")
            j += 1
        else:
            i += 1
            continue

        med = {
            "productName": name,
            "dosage": None,
            "quantity": quantity,
            "unit": unit,
            "instructions": None,
        }

        while j < len(lines):
            current = lines[j].strip()
            next_line = lines[j + 1].strip() if j + 1 < len(lines) else ""
            if _is_stop_text(current):
                break
            next_inline_name, _, _ = parse_inline_header(current)
            if next_inline_name:
                break
            if _is_item_number_text(current) and _is_probable_med_name(next_line) and not _is_unit_text(next_line):
                break
            if _is_noise_line(current):
                j += 1
                continue

            is_dose = _is_dose_text(current)

            if is_dose:
                existing = med.get("dosage")
                med["dosage"] = f"{existing}, {current}" if existing else current
                med["instructions"] = med["dosage"]
                j += 1
                continue

            inline_qty, inline_unit = parse_inline_qty_unit(current)
            if inline_qty is not None and inline_unit and med["quantity"] is None:
                med["quantity"] = inline_qty
                med["unit"] = inline_unit
                j += 1
                continue

            if med["quantity"] is None:
                standalone_qty = clean_number(current)
                if standalone_qty is not None and _ascii_lower(current) == str(standalone_qty):
                    if standalone_qty <= 20 and _is_dose_text(next_line) and not _is_unit_text(next_line):
                        j += 1
                        continue
                    med["quantity"] = standalone_qty
                    j += 1
                    continue

            parsed_unit = _canonical_unit(current)
            if parsed_unit:
                med["unit"] = parsed_unit
                j += 1
                continue

            j += 1

        if med["productName"] and (med.get("quantity") is not None or med.get("unit") or med.get("dosage")):
            medications.append(med)
        i = j

    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for med in medications:
        key = _ascii_lower(str(med.get("productName") or ""))[:40]
        if key and key not in seen:
            seen.add(key)
            deduped.append(med)
    return deduped

def _extract_interleaved_layout_medications(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse OCR output where item number, unit, quantity, name and directions are interleaved.

    This covers photographed/handwritten templates where OCR reads columns in a
    visually mixed order, for example: unit / quantity / drug name / directions.
    It deliberately uses structural signals only, not any fixed drug names.
    """
    medications: List[Dict[str, Any]] = []

    def parse_qty(value: str) -> Optional[int]:
        stripped = value.strip()
        return int(stripped) if re.fullmatch(r"\d{1,4}", stripped) else None

    def inline_qty_unit(value: str) -> tuple[Optional[int], Optional[str]]:
        for match in re.finditer(r"(?<!\d)(\d{1,4})\s*([A-Za-zÀ-ỹ?\.]{1,12})(?!\w)", value):
            unit = _canonical_unit(match.group(2))
            if unit:
                return int(match.group(1)), unit
        return None, None

    def is_candidate_name(idx: int) -> bool:
        value = lines[idx].strip()
        if not _is_probable_med_name(value):
            return False
        if _looks_like_admin_or_date_line(value):
            return False
        if inline_qty_unit(value)[1]:
            return True
        window = [item for item in lines[max(0, idx - 4): min(len(lines), idx + 5)] if not _is_dose_text(item)]
        has_unit = any(_is_unit_text(item) or inline_qty_unit(item)[1] for item in window)
        has_qty = any(parse_qty(item) is not None for item in window)
        has_med_strength = bool(re.search(r"\b\d+(?:[\.,]\d+)?\s*(?:mg|mcg|g|ml|iu)\b", _ascii_lower(value)))
        previous_is_item = idx > 0 and _is_item_number_text(lines[idx - 1])
        has_following_dose = any(_is_dose_text(item) for item in lines[idx + 1: min(len(lines), idx + 4)])
        has_shape = len(value.split()) >= 2 or bool(re.search(r"[A-Z]{2,}", value)) or has_med_strength or previous_is_item or has_following_dose
        return has_unit and (has_qty or has_med_strength or has_following_dose) and has_shape

    def nearby_detail(idx: int) -> tuple[Optional[int], Optional[str]]:
        quantity: Optional[int] = None
        unit: Optional[str] = None

        if idx + 2 < len(lines):
            next_unit = _canonical_unit(lines[idx + 1].strip())
            next_qty = parse_qty(lines[idx + 2].strip())
            if next_unit and next_qty is not None:
                return next_qty, next_unit
        if idx - 2 >= 0:
            prev_unit = _canonical_unit(lines[idx - 1].strip())
            prev_qty = parse_qty(lines[idx - 2].strip())
            if prev_unit and prev_qty is not None:
                return prev_qty, prev_unit

        next_name_pos: Optional[int] = None
        for pos in range(idx + 1, min(len(lines), idx + 8)):
            value = lines[pos].strip()
            if _looks_like_admin_or_date_line(value) or _is_stop_text(value):
                next_name_pos = pos
                break
            if _is_probable_med_name(value) and not _is_dose_text(value):
                next_name_pos = pos
                break

        candidates: List[tuple[int, str]] = []
        for pos in range(max(0, idx - 4), min(len(lines), idx + 6)):
            if pos == idx:
                continue
            if next_name_pos is not None and pos >= next_name_pos:
                continue
            value = lines[pos].strip()
            if _is_stop_text(value) or _is_dose_text(value) or _looks_like_admin_or_date_line(value):
                continue
            candidates.append((pos, value))

        for pos, value in sorted(candidates, key=lambda item: abs(item[0] - idx)):
            qty, parsed_unit = inline_qty_unit(value)
            if parsed_unit and unit is None:
                unit = parsed_unit
            if qty is not None and quantity is None:
                quantity = qty
            parsed_unit = _canonical_unit(value)
            if parsed_unit and unit is None:
                unit = parsed_unit

        for pos, value in sorted(candidates, key=lambda item: abs(item[0] - idx)):
            qty = parse_qty(value)
            if qty is None:
                continue
            # A bare number immediately before a name is often the item index.
            if pos == idx - 1 and 1 <= qty <= 20:
                previous = lines[pos - 1].strip() if pos - 1 >= 0 else ""
                if not _is_unit_text(previous):
                    continue
            quantity = qty
            break
        return quantity, unit

    def collect_dosage(idx: int) -> Optional[str]:
        doses: List[str] = []
        for pos in range(idx + 1, min(len(lines), idx + 5)):
            value = lines[pos].strip()
            if _is_stop_text(value) or _looks_like_admin_or_date_line(value):
                break
            if pos != idx and is_candidate_name(pos):
                break
            if _is_dose_text(value):
                doses.append(value)
        return ", ".join(doses) if doses else None

    for idx, line in enumerate(lines):
        if _is_stop_text(line):
            break
        if not is_candidate_name(idx):
            continue
        qty, unit = inline_qty_unit(line)
        clean_name = re.sub(r"^\s*[\(\[]?\d{1,2}[\.)\,\-\/\s]+", "", line).strip(" :-")
        if qty is not None and unit:
            clean_name = re.sub(r"\s*\d{1,4}\s*[A-Za-zÀ-ỹ?\.]{1,12}\s*$", "", clean_name).strip(" :-")
        else:
            qty, unit = nearby_detail(idx)
        dosage = collect_dosage(idx)
        if not clean_name or _looks_like_admin_or_date_line(clean_name):
            continue
        medications.append({
            "productName": clean_name,
            "dosage": dosage,
            "quantity": qty,
            "unit": unit,
            "instructions": dosage,
        })

    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for med in medications:
        key = _ascii_lower(str(med.get("productName") or ""))[:40]
        if key and key not in seen and (med.get("quantity") is not None or med.get("unit") or med.get("dosage")):
            seen.add(key)
            deduped.append(med)
    return deduped if len(deduped) >= 2 else []

def _extract_table_layout_medications(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse BHYT/table OCR where drug rows may not have leading item numbers."""
    medications: List[Dict[str, Any]] = []
    in_table = False
    i = 0
    header_tokens = ("ten thuoc", "ham luong", "dvt", "don vi", "so luong", "stt")

    def is_table_med_name(value: str) -> bool:
        key = _ascii_lower(value)
        if _is_probable_med_name(value):
            return True
        if _is_noise_line(value) or _is_unit_text(value) or any(token in key for token in header_tokens):
            return False
        return bool(re.search(r"[A-Za-zÀ-ỹ]", value) and re.search(r"\b\d+(?:[\.,]\d+)?\s*(?:mg|g|mcg|ml|iu)\b", key))

    while i < len(lines):
        line = lines[i].strip()
        key = _ascii_lower(line)

        if _is_stop_text(line):
            break
        if any(token in key for token in header_tokens):
            in_table = True
            i += 1
            continue
        if not in_table or not is_table_med_name(line):
            i += 1
            continue

        med = {
            "productName": line.strip(" :-"),
            "dosage": None,
            "quantity": None,
            "unit": None,
            "instructions": None,
        }

        j = i + 1
        while j < len(lines):
            current = lines[j].strip()
            current_key = _ascii_lower(current)
            if _is_stop_text(current):
                break
            if any(token in current_key for token in header_tokens) or _is_noise_line(current):
                j += 1
                continue
            if is_table_med_name(current) and (med["quantity"] is not None or med["unit"] or med["dosage"]):
                break

            qty_unit = re.search(r"(?<!\d)(\d{1,4})\s*([A-Za-zÀ-ỹ?\.]{1,12})(?!\w)", current)
            if qty_unit:
                parsed_unit = _canonical_unit(qty_unit.group(2))
                if parsed_unit:
                    med["quantity"] = med["quantity"] or int(qty_unit.group(1))
                    med["unit"] = med["unit"] or parsed_unit
                    j += 1
                    continue

            parsed_unit = _canonical_unit(current)
            if parsed_unit:
                med["unit"] = parsed_unit
                j += 1
                continue
            if re.fullmatch(r"\d{1,4}", current) and med["quantity"] is None:
                med["quantity"] = int(current)
                j += 1
                continue
            if _is_dose_text(current):
                existing = med.get("dosage")
                med["dosage"] = f"{existing}, {current}" if existing else current
                med["instructions"] = med["dosage"]
                j += 1
                continue
            if is_table_med_name(current):
                break
            j += 1

        has_detail = med.get("quantity") is not None or med.get("unit")
        if med["productName"] and (has_detail or (med.get("dosage") and _has_strong_med_name_signal(str(med.get("productName") or "")))):
            medications.append(med)
        i = max(j, i + 1)

    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for med in medications:
        key = _ascii_lower(str(med.get("productName") or ""))[:40]
        if key and key not in seen:
            seen.add(key)
            deduped.append(med)
    return deduped
