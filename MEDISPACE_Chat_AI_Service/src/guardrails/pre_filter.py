import re

# ---- PHÂN LOẠI CÂU HỎI ----

# Giới hạn độ dài tin nhắn (ký tự)
MAX_MESSAGE_LENGTH = 800

EMERGENCY_KEYWORDS = [
    'đau ngực', 'khó thở', 'ngất xỉu', 'co giật', 'xuất huyết',
    'ngộ độc', 'dị ứng nặng', 'sưng họng', 'mất ý thức',
    'cấp cứu', '115', 'đột quỵ', 'nhồi máu'
]

# Mental health crisis keywords — phải check trước prescription để ưu tiên cao hơn
MENTAL_HEALTH_KEYWORDS = [
    'tự tử', 'muốn chết', 'tự làm hại', 'tự làm đau', 'không muốn sống',
    'kết thúc cuộc đời', 'kết thúc tất cả', 'không muốn tồn tại',
    'muốn biến mất', 'chán sống', 'sống không có ý nghĩa'
]

# Dùng regex để giảm false positive — chỉ block khi có ngữ cảnh rõ ràng muốn mua/dùng kê đơn
PRESCRIPTION_PATTERNS = [
    r'\bkê đơn\b',
    r'\bcho tôi mua\b',
    r'\bcần đơn thuốc\b',
    r'\bliều bao nhiêu\b',
    r'\buống mấy viên\b',
    r'\btiêm bao nhiêu\b',
    r'\bđơn thuốc\b',
    # "kháng sinh" chỉ block khi đi kèm ý định mua/dùng, không block câu hỏi thông tin
    r'\b(mua|bán|kê|cho tôi|cần|tôi dùng|dùng được)\s+(thuốc\s+)?(kháng sinh|antibiotic)\b',
]


def classify_message(content: str) -> str:
    """
    Returns: 'too_long' | 'emergency' | 'mental_health_crisis' |
             'prescription_request' | 'general'
    """
    # 1. Kiểm tra độ dài trước tiên
    if len(content) > MAX_MESSAGE_LENGTH:
        return 'too_long'

    lower = content.lower().strip()

    # 2. Khẩn cấp y tế thể chất
    if any(kw in lower for kw in EMERGENCY_KEYWORDS):
        return 'emergency'

    # 3. Khủng hoảng tâm lý — ưu tiên cao, cần xử lý nhạy cảm
    if any(kw in lower for kw in MENTAL_HEALTH_KEYWORDS):
        return 'mental_health_crisis'

    # 4. Yêu cầu kê đơn / thuốc kê đơn (regex-based)
    for pattern in PRESCRIPTION_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return 'prescription_request'

    return 'general'


EMERGENCY_RESPONSE = (
    "⚠️ Đây có vẻ là tình huống khẩn cấp về y tế. "
    "Vui lòng gọi ngay số cấp cứu 115 hoặc đến cơ sở y tế gần nhất.\n\n"
    "Tôi đang tự động chuyển cuộc hội thoại này cho Dược sĩ để hỗ trợ bạn."
)

PRESCRIPTION_RESPONSE = (
    "Câu hỏi này liên quan đến kê đơn hoặc tư vấn y khoa chuyên sâu. "
    "Để đảm bảo an toàn, tôi sẽ chuyển yêu cầu này cho Dược sĩ của Medispace để tư vấn trực tiếp cho bạn nhé."
)

MENTAL_HEALTH_RESPONSE = (
    "Tôi cảm nhận được bạn đang trải qua giai đoạn rất khó khăn. "
    "Cảm ơn bạn đã tin tưởng chia sẻ — điều đó cần rất nhiều dũng cảm.\n\n"
    "Bạn không đơn độc. Vui lòng gọi ngay đường dây hỗ trợ sức khỏe tâm thần miễn phí "
    "1800 599 920 (24/7, miễn phí) để được chuyên gia lắng nghe và hỗ trợ.\n\n"
    "Tôi đang kết nối bạn với Dược sĩ của Medispace để đồng hành cùng bạn ngay lúc này."
)

TOO_LONG_RESPONSE = (
    "Tin nhắn của bạn quá dài để tôi có thể xử lý chính xác. "
    "Bạn vui lòng chia nhỏ câu hỏi hoặc tóm tắt lại trong khoảng 800 ký tự nhé. "
    "Tôi sẵn sàng hỗ trợ bạn!"
)

