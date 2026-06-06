import re

# ──────────────────────────────────────────────────────────────────────────────
# DANGEROUS PATTERNS — Phát hiện output nguy hiểm từ AI
#
# NGUYÊN TẮC: Chỉ block khi AI đưa ra liều lượng CỤ THỂ cho người dùng,
# KHÔNG block khi mô tả thông tin thuốc chung hoặc hướng dẫn sản phẩm.
# ──────────────────────────────────────────────────────────────────────────────

DANGEROUS_PATTERNS = [
    # Liều dùng cá nhân hóa cụ thể: "uống 500mg/ngày", "dùng 2 viên/lần"
    # Phải có cả đơn vị VÀ tần suất mới block (tránh false positive)
    r'\b\d+\s*(mg|ml|g)\s*/\s*(ngày|lần|sáng|tối|chiều|tuần)\b',
    r'\b(uống|tiêm|dùng|bôi)\s+\d+\s*(viên|ống|gói|lọ|ml|mg)\s+(mỗi|mỗi\s+)(ngày|lần|buổi)\b',
    r'\b\d+\s*viên\s+x\s*\d+\s*(lần|ngày)\b',

    # Hướng dẫn tự kê đơn (AI đóng vai bác sĩ kê đơn)
    # Block khi AI CHỦ ĐỘNG kê tên thuốc Rx cụ thể cho user
    r'\b(tôi kê|tôi chỉ định)\s+(?:cho bạn\s+)?(?:thuốc\s+)?[A-ZÀ-Ỹa-zà-ỹ]+\b',
    r'\b(tôi kê|tôi chỉ định)\b.{0,30}\b(để điều trị|để chữa|cho bệnh)\b',
]

# ──────────────────────────────────────────────────────────────────────────────
# BRAND NAME AUTO-CORRECTION
# LLM đôi khi cắt ngắn "Medispace" → "Medis" hoặc "Medi"
# ──────────────────────────────────────────────────────────────────────────────

BRAND_NAME_CORRECTIONS = [
    # "Medis." / "Medis," / "Medis " / "Medis\n" / "Medis'" → "Medispace..."
    # Không match: "Medisafe", "Medison" (chữ cái thường sau)
    (
        r'\bMedis(?=[^a-zA-Zàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ])',
        'Medispace'
    ),
    # "Medis" ở cuối chuỗi
    (r'\bMedis$', 'Medispace'),
    # "Medi " (có khoảng trắng sau) — LLM đôi khi cắt thêm
    (r'\bMedi(?=\s+(?:nhà thuốc|dược|hỗ trợ|sẽ|có thể|luôn|của))', 'Medispace'),
]

# ──────────────────────────────────────────────────────────────────────────────
# MARKDOWN CLEANUP — Gemma đôi khi trả về markdown dù đã dặn plain text
# ──────────────────────────────────────────────────────────────────────────────

MARKDOWN_PATTERNS = [
    (r'```(?:json|python|text|markdown)?', ''),   # code block markers
    (r'```', ''),
    (r'^\s*#{1,3}\s+', '', re.MULTILINE),          # heading markers (## Title)
    (r'\*{2}([^*]+)\*{2}', r'\1'),                 # **bold** → text
    (r'\*([^*]+)\*', r'\1'),                        # *italic* → text
]


def _clean_markdown(text: str) -> str:
    """Xoá markdown formatting từ output của LLM."""
    # Code blocks
    text = re.sub(r'```(?:json|python|text|markdown)?', '', text)
    text = re.sub(r'```', '', text)
    # Headings
    text = re.sub(r'(?m)^\s*#{1,3}\s+', '', text)
    # Bold **text**
    text = re.sub(r'\*{2}([^*\n]+)\*{2}', r'\1', text)
    # Italic *text*
    text = re.sub(r'\*([^*\n]+)\*', r'\1', text)
    # Bullet points "- " or "• " at start of line → giữ lại vì plain text OK
    return text.strip()


def sanitize_response(ai_response: str) -> tuple[str, bool]:
    """
    Kiểm tra và làm sạch phản hồi từ LLM.

    Args:
        ai_response: Phản hồi thô từ LLM

    Returns:
        (cleaned_response, was_blocked)
        - was_blocked=True: Phát hiện nội dung nguy hiểm, đã thay bằng fallback
        - was_blocked=False: Phản hồi hợp lệ (có thể đã clean markdown/brand)
    """
    # 1. Kiểm tra nội dung nguy hiểm
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, ai_response, re.IGNORECASE):
            fallback = (
                "Tôi không thể tư vấn chi tiết về liều lượng sử dụng cụ thể cho từng trường hợp. "
                "Vui lòng đọc kỹ tờ hướng dẫn sử dụng đi kèm sản phẩm hoặc kết nối với Dược sĩ "
                "của Medispace để được hướng dẫn an toàn và chính xác nhất."
            )
            return fallback, True

    # 2. Làm sạch markdown
    clean_text = _clean_markdown(ai_response)

    # 3. Tự động sửa tên thương hiệu bị viết tắt
    for pattern, replacement in BRAND_NAME_CORRECTIONS:
        clean_text = re.sub(pattern, replacement, clean_text)

    # 4. Chuẩn hóa khoảng trắng thừa
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)  # Tối đa 2 dòng trống
    clean_text = clean_text.strip()

    return clean_text, False
