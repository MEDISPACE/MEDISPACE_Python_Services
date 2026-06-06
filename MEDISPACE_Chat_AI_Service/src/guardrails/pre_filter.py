import re

# ---- CẤU HÌNH ----

# Giới hạn độ dài tin nhắn (ký tự)
MAX_MESSAGE_LENGTH = 800

# ──────────────────────────────────────────────────────────────────────────────
# EMERGENCY & SAFETY KEYWORDS
# ──────────────────────────────────────────────────────────────────────────────

EMERGENCY_KEYWORDS = [
    'đau ngực', 'khó thở', 'ngất xỉu', 'co giật', 'xuất huyết',
    'ngộ độc', 'dị ứng nặng', 'sưng họng', 'mất ý thức',
    'cấp cứu', '115', 'đột quỵ', 'nhồi máu', 'bất tỉnh',
    'không thở được', 'mạch không đập', 'xuất huyết não'
]

# Mental health crisis — ưu tiên cao hơn prescription
MENTAL_HEALTH_KEYWORDS = [
    'tự tử', 'muốn chết', 'tự làm hại', 'tự làm đau', 'không muốn sống',
    'kết thúc cuộc đời', 'kết thúc tất cả', 'không muốn tồn tại',
    'muốn biến mất', 'chán sống', 'sống không có ý nghĩa'
]

# ──────────────────────────────────────────────────────────────────────────────
# PRESCRIPTION PATTERNS — chỉ block khi có ý định MUA/KÊ ĐƠN rõ ràng
# Câu hỏi thông tin thuốc Rx → LLM xử lý qua system prompt
# ──────────────────────────────────────────────────────────────────────────────

PRESCRIPTION_PATTERNS = [
    # Yêu cầu kê đơn trực tiếp
    r'\bkê đơn\b',
    r'\bcần đơn thuốc\b',
    # Yêu cầu mua thuốc kê đơn cụ thể (tên thuốc + ý định mua)
    r'\b(mua|bán|cho tôi mua|tôi cần mua)\s+(thuốc\s+)?(kháng sinh|antibiotic|amoxicillin|augmentin|cephalexin|metronidazole|ciprofloxacin|azithromycin|clarithromycin)\b',
    # Tiêm thuốc (luôn cần chỉ định bác sĩ)
    r'\btiêm\s+(thuốc|vắc\s*xin|insulin|morphine)\b',
]

# ──────────────────────────────────────────────────────────────────────────────
# NEW INTENTS — E-Commerce Feature Coverage
# ──────────────────────────────────────────────────────────────────────────────

# Theo dõi đơn hàng
ORDER_TRACKING_PATTERNS = [
    r'\b\u0111\u01a1n h\u00e0ng\b.{0,30}\b\u0111\u00e2u\b',
    r'\b\u0111\u01a1n h\u00e0ng\b.{0,30}\btr\u1ea1ng th\u00e1i\b',
    r'\bORD[-\s]?\d+\b',
    r'\btheo d\u00f5i\b.{0,20}\b\u0111\u01a1n\b',
    r'\bgiao h\u00e0ng\b.{0,20}\bbao (gi\u1edd|l\u00e2u|nhi\u00eau ng\u00e0y)\b',
    r'\b\u0111\u01a1n\b.{0,20}\bgiao (ch\u01b0a|r\u1ed3i|\u0111\u1ebfn \u0111\u00e2u)\b',
    r'\bv\u1eadn \u0111\u01a1n\b',
    r'\btracking\b',
    r'\bship(ping)?\b.{0,20}\b\u0111\u01a1n\b',
    r'\bki\u1ec3m tra\b.{0,20}\b\u0111\u01a1n h\u00e0ng\b',
    r'\b\u0111\u01a1n (\u0111\u00e3|b\u1ecb)\b.{0,20}\b(hu\u1ef7|h\u1ee7|cancel)\b',
    # Thêm: "X ngày rồi chưa thấy giao"
    r'\bch\u01b0a th\u1ea5y giao\b',
    r'\bch\u01b0a nh\u1eadn\b.{0,20}\bh\u00e0ng\b',
    r'\b\u0111\u1eb7t h\u00e0ng\b.{0,20}\bch\u01b0a (giao|nh\u1eadn)\b',
]

# Loyalty points & hạng thành viên
LOYALTY_PATTERNS = [
    r'\b(bao nhiêu|còn|xem)\b.{0,20}\bđiểm\b',
    r'\bđiểm.{0,20}(tích|lũy|còn|hết|hạn|đổi)\b',
    r'\bhạng\b.{0,20}\b(thành viên|silver|gold|bạc|vàng|kim cương|platinum)\b',
    r'\blên hạng\b',
    r'\btích điểm\b',
    r'\bđổi điểm\b',
    r'\bđiểm thưởng\b',
    r'\b(loyalty|điểm tích lũy)\b',
    r'\bcòn bao nhiêu.{0,15}điểm\b',
    r'\bđiểm.{0,20}sắp hết hạn\b',
]

# Coupon & khuyến mãi
COUPON_PATTERNS = [
    r'\bmã giảm giá\b',
    r'\bkhuyến mãi\b',
    r'\bgiảm giá\b.{0,20}\bmã\b',
    r'\bcoupon\b',
    r'\bvoucher\b',
    r'\bmã\b.{0,20}\bgiảm\b',
    r'\bflash sale\b',
    r'\bưu đãi\b.{0,20}\b(nào|gì|không)\b',
    r'\báp dụng\b.{0,20}\bgiảm giá\b',
    r'\bmiễn phí vận chuyển\b',
    r'\bfreeship\b',
]

# Đổi trả hàng & hoàn tiền
RETURN_PATTERNS = [
    r'\btrả hàng\b',
    r'\bđổi hàng\b',
    r'\bhoàn tiền\b',
    r'\bsản phẩm (bị lỗi|hư|vỡ|hỏng)\b',
    r'\bhàng (bị lỗi|hư|vỡ|hỏng|sai)\b',
    r'\bRET[-\s]?\d+\b',
    r'\byêu cầu\b.{0,20}\b(đổi|trả|hoàn)\b',
    r'\bgiao sai\b',
    r'\bkhông đúng\b.{0,20}\b(hàng|sản phẩm|thuốc)\b',
    r'\bphản ứng dị ứng\b.{0,30}\b(trả|đổi|hoàn)\b',
    r'\bchính sách\b.{0,20}\btrả hàng\b',
    # Thêm: "bị vỡ, muốn đổi" — pattern rõ ràng hơn
    r'\b(bị vỡ|bị hỏng|bị lỗi|bị hư).{0,20}\bmuốn (đổi|trả)\b',
    r'\bnhận được\b.{0,20}\b(bị vỡ|bị hỏng|bị lỗi|sai)\b',
]

# Kiểm tra trạng thái đơn thuốc kê đơn (không phải mua)
PRESCRIPTION_STATUS_PATTERNS = [
    r'\bđơn thuốc\b.{0,30}\b(duyệt|xét duyệt|kiểm tra|trạng thái|đã gửi)\b',
    r'\bPRE[-\s]?\d+\b',
    r'\bkiểm tra\b.{0,20}\bđơn thuốc\b',
    r'\bđơn thuốc\b.{0,20}\b(còn|hết)\b.{0,10}\bhiệu lực\b',
    r'\bđơn thuốc\b.{0,20}\b(được|bị)\b.{0,10}\b(duyệt|từ chối|reject)\b',
    r'\bkết quả\b.{0,20}\bxét duyệt\b',
]

# Tìm kiếm sản phẩm (product search intent — để RAG xử lý tốt hơn)
PRODUCT_SEARCH_PATTERNS = [
    r'\btìm\b.{0,30}\b(thuốc|sản phẩm|vitamin|thực phẩm chức năng)\b',
    r'\bcó bán\b.{0,30}\b(thuốc|sản phẩm)\b',
    r'\bgiá\b.{0,30}\b(thuốc|sản phẩm)\b',
    r'\b(thuốc|sản phẩm|vitamin)\b.{0,30}\bgiá bao nhiêu\b',
    r'\bgợi ý\b.{0,30}\b(thuốc|sản phẩm)\b',
    r'\bnên dùng\b.{0,30}\bloại (nào|gì)\b',
    # Thêm: "so sánh X và Y" — product comparison (tổng quát hơn)
    r'\bso sánh\b.{0,80}\b(và|vs\.?|hoặc)\b',   # "so sánh A và B"
    r'\bso sánh\b.{0,50}\b(thuốc|sản phẩm|vitamin|blackmores|centrum)\b',
    r'\bso sánh\b.{0,30}(và|vs\.?|hoặc).{0,30}\b(tốt hơn|khác gì|khác nhau)\b',
    r'\b(loại nào|cái nào)\b.{0,30}\b(tốt hơn|tốt nhất)\b',
]


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFY FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def classify_message(content: str) -> str:
    """
    Phân loại tin nhắn của user theo intent.

    Returns:
        'too_long'            — Tin nhắn quá dài
        'emergency'           — Tình huống khẩn cấp y tế
        'mental_health_crisis'— Khủng hoảng tâm lý
        'prescription_request'— Yêu cầu mua/kê đơn thuốc Rx
        'order_tracking'      — Hỏi về đơn hàng / vận chuyển
        'loyalty_inquiry'     — Hỏi về điểm thưởng / hạng thành viên
        'coupon_inquiry'      — Hỏi về mã giảm giá / khuyến mãi
        'return_request'      — Yêu cầu đổi trả hàng
        'prescription_status' — Kiểm tra trạng thái đơn thuốc đã gửi
        'product_search'      — Tìm kiếm / hỏi giá sản phẩm
        'general'             — Tư vấn sức khỏe / thuốc thông thường
    """
    # 1. Kiểm tra độ dài
    if len(content) > MAX_MESSAGE_LENGTH:
        return 'too_long'

    lower = content.lower().strip()

    # 2. Khẩn cấp y tế — ưu tiên cao nhất
    if any(kw in lower for kw in EMERGENCY_KEYWORDS):
        return 'emergency'

    # 3. Khủng hoảng tâm lý
    if any(kw in lower for kw in MENTAL_HEALTH_KEYWORDS):
        return 'mental_health_crisis'

    # 4. Đơn hàng (trước prescription để "đơn thuốc + trạng thái" không bị nhầm)
    for pattern in ORDER_TRACKING_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return 'order_tracking'

    # 5. Trạng thái đơn thuốc đã gửi (trước prescription_request)
    for pattern in PRESCRIPTION_STATUS_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return 'prescription_status'

    # 6. Mua / kê đơn thuốc Rx — block, escalate
    for pattern in PRESCRIPTION_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return 'prescription_request'

    # 7. Loyalty points & hạng thành viên
    for pattern in LOYALTY_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return 'loyalty_inquiry'

    # 8. Coupon & khuyến mãi
    for pattern in COUPON_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return 'coupon_inquiry'

    # 9. Đổi trả hàng
    for pattern in RETURN_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return 'return_request'

    # 10. Tìm kiếm sản phẩm
    for pattern in PRODUCT_SEARCH_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return 'product_search'

    return 'general'


# ──────────────────────────────────────────────────────────────────────────────
# PREFILTER RESPONSE MESSAGES
# (dùng khi không cần gọi LLM — trả thẳng từ prefilter)
# ──────────────────────────────────────────────────────────────────────────────

EMERGENCY_RESPONSE = (
    "⚠️ Đây có vẻ là tình huống khẩn cấp về y tế. "
    "Vui lòng gọi ngay số cấp cứu 115 hoặc đến cơ sở y tế gần nhất.\n\n"
    "Tôi đang tự động chuyển cuộc hội thoại này cho Dược sĩ để hỗ trợ bạn."
)

PRESCRIPTION_RESPONSE = (
    "Yêu cầu này liên quan đến thuốc kê đơn cần có chỉ định của bác sĩ. "
    "Để đảm bảo an toàn, tôi sẽ kết nối bạn với Dược sĩ của Medispace để hỗ trợ trực tiếp nhé."
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

# Các intent mới — được xử lý bởi LLM với prompt chuyên biệt (không prefilter cứng)
# Nhưng vẫn có fallback message nếu cần
ORDER_TRACKING_FALLBACK = (
    "Để kiểm tra trạng thái đơn hàng, bạn có thể vào mục 'Đơn hàng của tôi' "
    "trong ứng dụng Medispace. Hoặc cung cấp mã đơn hàng (ORD-xxx) để tôi hỗ trợ tra cứu nhé!"
)

LOYALTY_FALLBACK = (
    "Để xem điểm thưởng và hạng thành viên, bạn vào mục 'Tài khoản' → 'Điểm thưởng'. "
    "Bạn cũng có thể hỏi tôi và cung cấp thông tin tài khoản để được hỗ trợ!"
)

COUPON_FALLBACK = (
    "Các mã giảm giá hiện có thể xem tại mục 'Khuyến mãi' trên ứng dụng Medispace. "
    "Bạn có thể cho tôi biết giá trị đơn hàng dự kiến để tôi gợi ý mã phù hợp nhất nhé!"
)

RETURN_FALLBACK = (
    "Để yêu cầu đổi/trả hàng, vui lòng cung cấp:\n"
    "1. Mã đơn hàng (ORD-xxx)\n"
    "2. Sản phẩm muốn đổi/trả\n"
    "3. Lý do đổi/trả và ảnh minh chứng (nếu có)\n\n"
    "Tôi sẽ hướng dẫn bạn tạo yêu cầu hoặc kết nối với Dược sĩ để hỗ trợ trực tiếp."
)

PRESCRIPTION_STATUS_FALLBACK = (
    "Để kiểm tra trạng thái đơn thuốc đã gửi, bạn vào mục 'Đơn thuốc của tôi'. "
    "Hoặc cung cấp mã đơn thuốc (PRE-xxx) để tôi tra cứu giúp bạn nhé!"
)
