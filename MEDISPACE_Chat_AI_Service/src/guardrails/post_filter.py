import re

# Pattern detect output nguy hiểm từ AI
DANGEROUS_PATTERNS = [
    r'\d+\s*(mg|ml|g|viên|gói|ống)\s*/?\s*(ngày|lần|sáng|tối|chiều)',  # liều cụ thể
    r'(uống|tiêm|dùng)\s+\d+\s*(viên|ống|gói|lọ)',         # "uống 2 viên"
    r'(kê|bán|cần mua|hãy mua)\s+(?:thuốc\s+)?(?!đơn\b|toa\b)\w+',         # kê đơn trá hình
]

# Sửa tên thương hiệu bị viết tắt (LLM đôi khi cắt ngắn "Medispace" → "Medis")
# Pattern: "Medis" đứng cuối câu, trước dấu câu hoặc trước ký tự không phải chữ thường
BRAND_NAME_CORRECTIONS = [
    # "Medis." / "Medis," / "Medis " / "Medis\n" / "Medis'" → "Medispace..."
    (r'\bMedis(?=[^a-zA-Zàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ])', 'Medispace'),
    # "của Medis" ở cuối chuỗi (không có ký tự tiếp theo)
    (r'\bMedis$', 'Medispace'),
]

SAFE_SUFFIX = ""

def sanitize_response(ai_response: str) -> tuple[str, bool]:
    """
    Returns: (sanitized_response, was_modified)
    Nếu phát hiện nội dung nguy hiểm -> replace toàn bộ bằng fallback.
    """
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, ai_response, re.IGNORECASE):
            fallback = (
                "Tôi không thể tư vấn chi tiết về liều lượng sử dụng cụ thể. "
                "Vui lòng kết nối với Dược sĩ của Medispace để được hướng dẫn an toàn và chính xác nhất."
            )
            return fallback, True

    # Xoá markdown code block markers nếu mô hình trả về (hay gặp ở Gemma)
    clean_text = re.sub(r'```(?:json)?|```', '', ai_response).strip()

    # Tự động sửa tên thương hiệu bị viết tắt
    for pattern, replacement in BRAND_NAME_CORRECTIONS:
        clean_text = re.sub(pattern, replacement, clean_text)

    return clean_text, False
