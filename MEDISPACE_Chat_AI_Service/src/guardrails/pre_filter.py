import re
import unicodedata

MAX_MESSAGE_LENGTH = 800


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize('NFD', value or '')
    value = ''.join(ch for ch in value if unicodedata.category(ch) != 'Mn')
    value = value.replace('đ', 'd').replace('Đ', 'D')
    return value.lower().strip()


EMERGENCY_KEYWORDS = [
    'đau ngực', 'tức ngực', 'đau thắt ngực', 'khó thở', 'không thở được',
    'mạch không đập', 'tim ngừng đập', 'đột quỵ', 'nhồi máu', 'nhồi máu cơ tim',
    'mặt méo', 'miệng méo', 'nói ngọng đột ngột', 'nói không ra lời',
    'liệt tay', 'liệt chân', 'tê liệt đột ngột', 'yếu tay đột ngột',
    'mắt mờ đột ngột', 'nhìn đôi đột ngột', 'đầu đau dữ dội đột ngột',
    'đau đầu sét đánh', 'ngất xỉu', 'bất tỉnh', 'mất ý thức', 'hôn mê',
    'ngã đập đầu', 'chấn thương đầu', 'chảy máu không cầm', 'xuất huyết',
    'xuất huyết não', 'chảy máu nhiều', 'dị ứng nặng', 'sốc phản vệ',
    'sưng họng', 'sưng môi đột ngột', 'co giật', 'ngộ độc', 'cấp cứu', '115',
]

MENTAL_HEALTH_KEYWORDS = [
    'tự tử', 'muốn chết', 'tự làm hại', 'tự làm đau', 'không muốn sống',
    'kết thúc cuộc đời', 'kết thúc tất cả', 'không muốn tồn tại',
    'muốn biến mất', 'chán sống', 'sống không có ý nghĩa',
]

GREETING_PATTERNS = [
    r'^\s*(chao|chao ban|xin chao|hello|hi|hey|alo|aloo|medispace oi)[!?.\s]*$',
]

PRESCRIPTION_PATTERNS = [
    r'\bke don\b',
    r'\bcan don thuoc\b',
    r'\b(mua|ban|cho toi mua|toi can mua)\s+(thuoc\s+)?(khang sinh|antibiotic|amoxicillin|augmentin|cephalexin|metronidazole|ciprofloxacin|azithromycin|clarithromycin)\b',
    r'\btiem\s+(thuoc|vac\s*xin|insulin|morphine)\b',
]

PERSONALIZED_DOSAGE_PATTERNS = [
    r'\b(uong|dung|xai|boi|nho|tiem)\b.{0,50}\b(may|bao nhieu)\s*(vien|lan|ml|mg|goi|giot|ngay|lieu)\b',
    r'\b(may|bao nhieu)\s*(vien|lan|ml|mg|goi|giot|ngay|lieu)\b.{0,50}\b(uong|dung|xai|boi|nho|tiem)\b',
    r'\b(lieu dung|lieu luong|cach dung)\b.{0,40}\b(cho toi|toi|em|be|tre|nguoi gia|mang thai|benh nay)\b',
]

ORDER_TRACKING_PATTERNS = [
    r'\bkiem tra\b.{0,30}\bdon hang\b',
    r'\bdon hang\b.{0,30}\b(dau|trang thai|giao chua|giao roi|den dau)\b',
    r'\bORD[-\s]?\d+\b',
    r'\btheo doi\b.{0,20}\bdon\b',
    r'\bgiao hang\b.{0,20}\b(bao gio|bao lau|nhieu ngay)\b',
    r'\bvan don\b', r'\btracking\b', r'\bchua thay giao\b',
    r'\bchua nhan\b.{0,20}\bhang\b',
]

LOYALTY_PATTERNS = [
    r'\b(bao nhieu|con|xem)\b.{0,20}\bdiem\b',
    r'\bdiem.{0,20}(tich|luy|con|het|han|doi)\b',
    r'\bhang\b.{0,20}\b(thanh vien|silver|gold|bac|vang|kim cuong|platinum)\b',
    r'\blen hang\b', r'\btich diem\b', r'\bdoi diem\b', r'\bdiem thuong\b',
    r'\b(loyalty|diem tich luy)\b',
]

COUPON_PATTERNS = [
    r'\bma giam gia\b', r'\bkhuyen mai\b', r'\bgiam gia\b.{0,20}\bma\b',
    r'\bcoupon\b', r'\bvoucher\b', r'\bma\b.{0,20}\bgiam\b', r'\bflash sale\b',
    r'\buu dai\b.{0,20}\b(nao|gi|khong)\b', r'\bmien phi van chuyen\b', r'\bfreeship\b',
]

RETURN_PATTERNS = [
    r'\btra hang\b', r'\bdoi hang\b', r'\bhoan tien\b', r'\bhoan tra\b',
    r'\bdon hang\b.{0,40}\b(da|duoc)?\s*(hoan tra|tra lai|hoan tien)\b',
    r'\bsan pham (bi loi|hu|vo|hong)\b', r'\bhang (bi loi|hu|vo|hong|sai)\b',
    r'\bsan pham\b.{0,40}\b(bi loi|hu|vo|hong|sai)\b.{0,40}\b(doi|tra|hoan)\b',
    r'\bRET[-\s]?\d+\b', r'\bye u cau\b.{0,20}\b(doi|tra|hoan)\b',
    r'\bgiao sai\b', r'\bkhong dung\b.{0,20}\b(hang|san pham|thuoc)\b',
    r'\bchinh sach\b.{0,20}\btra hang\b',
]

PRESCRIPTION_STATUS_PATTERNS = [
    r'\bdon thuoc\b.{0,30}\b(duyet|xet duyet|kiem tra|trang thai|da gui)\b',
    r'\bPRE[-\s]?\d+\b', r'\bkiem tra\b.{0,20}\bdon thuoc\b',
    r'\bdon thuoc\b.{0,20}\b(con|het)\b.{0,10}\bhieu luc\b',
    r'\bket qua\b.{0,20}\bxet duyet\b',
]

PRODUCT_SEARCH_PATTERNS = [
    r'\btim\b.{0,30}\b(thuoc|san pham|vitamin|thuc pham chuc nang)\b',
    r'\bco ban\b.{0,30}\b(thuoc|san pham)\b',
    r'\bgia\b.{0,30}\b(thuoc|san pham)\b',
    r'\b(thuoc|san pham|vitamin)\b.{0,30}\bgia bao nhieu\b',
    r'\bgoi y\b.{0,30}\b(thuoc|san pham)\b',
    r'\bnen dung\b.{0,30}\bloai (nao|gi)\b',
    r'\bso sanh\b.{0,80}\b(va|vs\.?|hoac)\b',
]

DRUG_INFO_GENERAL_PATTERNS = [
    r'\b(thuoc|hoat chat|vien|siro|gel)\b.{0,50}\b(cong dung|tac dung|dung de lam gi|dieu tri benh gi|tri benh gi|chi dinh|la thuoc gi)\b',
    r'\b(cong dung|tac dung|dung de lam gi|dieu tri benh gi|tri benh gi|chi dinh)\b.{0,50}\b(thuoc|hoat chat|vien|siro|gel)\b',
    r'\b[a-zA-Z][a-zA-Z0-9-]{2,}\b.{0,30}\b(cong dung|tac dung|dung de lam gi|dieu tri benh gi|tri benh gi|la thuoc gi)\b',
]


def classify_message(content: str) -> str:
    #gửi ảnh không có text
    if not content or not content.strip():
        return 'image_only'

    if len(content) > MAX_MESSAGE_LENGTH:
        return 'too_long'

    lower = content.lower().strip()
    normalized = _normalize_text(lower)
    searchable = f"{lower} {normalized}"

    # chào hỏi
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return 'greeting'

    #khẩn cấp
    if any(_normalize_text(kw) in searchable for kw in EMERGENCY_KEYWORDS):
        return 'emergency'

    #khủng hoảng tinh thần
    if any(_normalize_text(kw) in searchable for kw in MENTAL_HEALTH_KEYWORDS):
        return 'mental_health_crisis'

    for pattern in ORDER_TRACKING_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            return 'order_tracking'

    for pattern in PRESCRIPTION_STATUS_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            return 'prescription_status'

    for pattern in PRESCRIPTION_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            return 'prescription_request'

    for pattern in PERSONALIZED_DOSAGE_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            return 'personalized_dosage'

    for pattern in LOYALTY_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            return 'loyalty_inquiry'

    for pattern in COUPON_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            return 'coupon_inquiry'

    for pattern in RETURN_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            return 'return_request'

    for pattern in DRUG_INFO_GENERAL_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            return 'drug_info_general'

    for pattern in PRODUCT_SEARCH_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            return 'product_search'

    return 'general'


GREETING_RESPONSE = (
    "Chào bạn, mình là Trợ lý Sức khỏe AI của Medispace. "
    "Mình có thể hỗ trợ bạn tra cứu thông tin thuốc, sản phẩm, đơn hàng hoặc hướng dẫn kết nối Dược sĩ khi cần. "
    "Bạn cần mình hỗ trợ gì hôm nay?"
)

EMERGENCY_RESPONSE = (
    "Đây có vẻ là tình huống khẩn cấp về y tế. "
    "Vui lòng gọi ngay số cấp cứu 115 hoặc đến cơ sở y tế gần nhất.\n\n"
    "Tôi đang tự động chuyển cuộc hội thoại này cho Dược sĩ để hỗ trợ bạn."
)

PRESCRIPTION_RESPONSE = (
    "Yêu cầu này liên quan đến thuốc kê đơn cần có chỉ định của bác sĩ. "
    "Để đảm bảo an toàn, tôi sẽ kết nối bạn với Dược sĩ của Medispace để hỗ trợ trực tiếp nhé."
)

PERSONALIZED_DOSAGE_RESPONSE = (
    "Mình không thể tự đưa liều dùng cá nhân hóa như uống mấy viên, mỗi ngày bao nhiêu lần, "
    "hoặc dùng trong bao lâu nếu chưa có đánh giá của Bác sĩ/Dược sĩ. Liều dùng phụ thuộc vào tuổi, cân nặng, "
    "bệnh nền, chức năng gan thận, tình trạng hiện tại và các thuốc đang dùng. Bạn vui lòng gửi đơn thuốc/ảnh hộp thuốc "
    "hoặc kết nối Dược sĩ Medispace để được kiểm tra an toàn nhé."
)

MENTAL_HEALTH_RESPONSE = (
    "Tôi cảm nhận được bạn đang trải qua giai đoạn rất khó khăn. "
    "Cảm ơn bạn đã tin tưởng chia sẻ.\n\n"
    "Bạn không đơn độc. Vui lòng gọi ngay đường dây hỗ trợ sức khỏe tâm thần miễn phí "
    "1800 599 920 (24/7, miễn phí) để được chuyên gia lắng nghe và hỗ trợ.\n\n"
    "Tôi đang kết nối bạn với Dược sĩ của Medispace để đồng hành cùng bạn ngay lúc này."
)

TOO_LONG_RESPONSE = (
    "Tin nhắn của bạn quá dài để tôi có thể xử lý chính xác. "
    "Bạn vui lòng chia nhỏ câu hỏi hoặc tóm tắt lại trong khoảng 800 ký tự nhé."
)

ORDER_TRACKING_FALLBACK = (
    "Để kiểm tra trạng thái đơn hàng, bạn có thể vào mục 'Đơn hàng của tôi' trong ứng dụng Medispace "
    "hoặc cung cấp mã đơn hàng (ORD-xxx) để tôi hỗ trợ tra cứu nhé."
)

LOYALTY_FALLBACK = (
    "Để xem điểm thưởng và hạng thành viên, bạn vào mục 'Tài khoản' -> 'Điểm thưởng'."
)

COUPON_FALLBACK = (
    "Các mã giảm giá hiện có có thể xem tại mục 'Khuyến mãi' trên ứng dụng Medispace."
)

RETURN_FALLBACK = (
    "Để yêu cầu đổi/trả hàng, vui lòng cung cấp mã đơn hàng, sản phẩm muốn đổi/trả và lý do kèm ảnh minh chứng nếu có."
)

PRESCRIPTION_STATUS_FALLBACK = (
    "Để kiểm tra trạng thái đơn thuốc đã gửi, bạn vào mục 'Đơn thuốc của tôi' hoặc cung cấp mã đơn thuốc (PRE-xxx)."
)
