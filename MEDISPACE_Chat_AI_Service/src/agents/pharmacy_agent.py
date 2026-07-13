"""
pharmacy_agent.py — PharmacyAgent core

Phase 1 Improvements:
- [A1] Context window management: giới hạn history theo turns và chars
- [A2] Post-filter false positive giảm (handled in post_filter.py)
- [B2] Intent-aware routing: 11 intents thay vì 5
- [B3] Dynamic temperature theo intent
- Few-shot examples mở rộng: 3 → 8+ examples
- Prompt templates chuyên biệt cho từng intent nhóm

Fix Log:
- [FIX-1] is_escalated: dùng intent-based logic thay vì was_sanitized
- [FIX-4] LLM retry: thêm exponential backoff 2 lần khi timeout
- [FIX-5] [GỢI Ý] parser: mở rộng pattern match nhiều dạng
- [FIX-6] System prompt: thêm ngày hiện tại
- [FIX-7] History: MAX_HISTORY_TURNS=6 đồng bộ với BE (6 messages)
- [FIX-8] RAG articles: mở rộng cho return_request và coupon_inquiry
"""
import os
import json
import base64
import httpx
import asyncio
import logging
import re
import unicodedata
from datetime import datetime
from src.guardrails.pre_filter import (
    classify_message,
    GREETING_RESPONSE,
    EMERGENCY_RESPONSE,
    PRESCRIPTION_RESPONSE,
    PERSONALIZED_DOSAGE_RESPONSE,
    MENTAL_HEALTH_RESPONSE,
    TOO_LONG_RESPONSE,
)
from src.guardrails.post_filter import sanitize_response
from src.rag.typesense_client import search_products_for_rag, search_articles_for_rag

logger = logging.getLogger("chat_ai.agent")

LLM_BASE  = os.getenv("CUSTOM_LLM_BASE_URL", "https://llm.datateam.space").rstrip("/")
LLM_MODEL = os.getenv("CUSTOM_LLM_MODEL", "gemma-4-e4b-it.gguf")
LLM_MAX_TOKENS = int(os.getenv("CUSTOM_LLM_MAX_TOKENS", "1536"))
LLM_IMAGE_MAX_TOKENS = int(os.getenv("CUSTOM_LLM_IMAGE_MAX_TOKENS", str(max(4096, LLM_MAX_TOKENS))))
LLM_MAX_RETRIES = int(os.getenv("CUSTOM_LLM_MAX_RETRIES", "2"))  # [FIX-4]

# ── Context window limits ─────────────────────────────────────────────────────
# [FIX-7] Đồng bộ với BE: BE giới hạn limit(6) messages khi gửi history
MAX_HISTORY_TURNS = 6      # Tối đa 6 lượt hội thoại (user+assistant mỗi lượt)
MAX_HISTORY_CHARS = 3000    # Tối đa 3000 ký tự trong history

# ── RAG config ────────────────────────────────────────────────────────
# Nếu FE gửi ít hơn ngưỡng này, tự động query Typesense bổ sung
RAG_MIN_PRODUCTS = 2        # Dưới 2 sản phẩm → trigger Typesense auto-fetch
RAG_MAX_PRODUCTS = 6        # Tối đa sản phẩm đưa vào RAG context

MAX_IMAGE_BYTES = int(os.getenv("CHAT_AI_MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
IMAGE_FETCH_TIMEOUT = float(os.getenv("CHAT_AI_IMAGE_FETCH_TIMEOUT", "12"))
SUPPORTED_IMAGE_MIME = {"image/jpeg", "image/png", "image/webp"}


def _detect_image_mime(image_bytes: bytes) -> str | None:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return None


class ImageFetchError(Exception):
    pass


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFD", value or "")
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return value.lower()


async def normalize_image_for_llm(image_url: str) -> str:
    """Download an image URL and return a data URL so the LLM receives bytes directly."""
    if image_url and re.match(r"^data:image/(jpeg|jpg|png|webp);base64,", image_url, re.IGNORECASE):
        return image_url
    if not image_url or not re.match(r"^https?://", image_url, re.IGNORECASE):
        raise ImageFetchError("image_url must be an http(s) URL")

    try:
        async with httpx.AsyncClient(timeout=IMAGE_FETCH_TIMEOUT, follow_redirects=True) as client:
            async with client.stream("GET", image_url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                chunks = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_IMAGE_BYTES:
                        raise ImageFetchError("Ảnh quá lớn, vui lòng gửi ảnh nhỏ hơn 8MB")
                    chunks.append(chunk)

        image_bytes = b"".join(chunks)
        if not image_bytes:
            raise ImageFetchError("Không tải được nội dung ảnh")
        detected_type = _detect_image_mime(image_bytes)
        if content_type not in SUPPORTED_IMAGE_MIME:
            if not detected_type:
                logger.warning(
                    "[ImageFetch] Unsupported content-type=%s url=%s",
                    content_type or "missing", image_url[:120]
                )
                raise ImageFetchError("URL khong phai anh hop le")
            logger.info(
                "[ImageFetch] Using detected image type %s instead of content-type=%s",
                detected_type, content_type or "missing"
            )
            content_type = detected_type
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{content_type};base64,{encoded}"
    except ImageFetchError:
        raise
    except Exception as exc:
        raise ImageFetchError("Ảnh không truy cập được, vui lòng gửi lại ảnh rõ hơn") from exc

# ── Intent → Temperature mapping ─────────────────────────────────────────────
INTENT_TEMPERATURE: dict[str, float] = {
    # An toàn y tế → ít sáng tạo, ít hallucination
    "general":              0.30,
    "drug_info_general":    0.25,
    "product_search":       0.35,
    "image_only":           0.30,   # phân tích ảnh
    # Thông tin tra cứu → trả lời chính xác
    "order_tracking":       0.20,
    "prescription_status":  0.20,
    "loyalty_inquiry":      0.20,
    "coupon_inquiry":       0.25,
    "return_request":       0.20,
}

# ── Image analysis system prompt ─────────────────────────────────────────
IMAGE_SYSTEM_PROMPT = """Bạn là Trợ lý Nhà thuốc AI của Medispace — một nền tảng nhà thuốc trực tuyến uy tín tại Việt Nam.
Bạn vừa nhận được một hình ảnh từ khách hàng. Hãy phân tích ảnh và phản hồi theo các tình huống sau:

**Nếu ảnh là ĐƠN THUỐC KÊ ĐƠN (có tên bầc sĩ, tên bệnh viện, danh sách thuốc, liều lượng):**
- Đọc và liệt kê rõ các thuốc có trong đơn (tên thuốc, liều lượng, cách dùng nếu thấy)
- Giải thích ngắn gọn công dụng của từng thuốc bằng ngôn ngữ dễ hiểu
- Cuối cùng HỎi người dùng: "Bạn có muốn gửi đơn thuốc này cho Dược sĩ của Medispace để được tư vấn và xác nhận đơn không?"
- QUAN TRỌNG: Nhắc rõ đây là thuốc kê đơn, cần theo đúng chỉ định của bác sĩ

**Nếu ảnh là HỘP/VỆ THUỐC, thực phẩm chức năng, mỹ phẩm (có nhãn hiệu, thông tin sản phẩm):**
- Mô tả sản phẩm tìm thấy trong ảnh
- Nêu công dụng, thành phần chính nếu thấy
- Giợi ý tìm kiếm sản phẩm tương tự hoặc chính hãng tại Medispace

**Nếu ảnh là TRIỆU CHỨNG cơ thể (vết thương, da liễu, phát ban, sưng tấy...)**:
- Kh?ng chẩn đoán bệnh dựa trên ảnh
- Nhẹ nhàng gi?i thích giới hạn của AI trong việc chẩn đoán
- Khưyến nghị gặp bác sĩ hoặc dược sĩ để được thăm khám trực tiếp

**Nếu ảnh là KẾT QUẢ XÉT NGHIỆM, phiếu chẩn đoán, phiếu xết nghiệm:**
- Kh?ng giải thích kết quả y tế chính xác dựa trên ảnh
- Khưyến nghị gặp bác sĩ được điều trị để giải thích kết quả
- Đề nghị kết nối Dược sĩ Medispace nếu cần tư vấn thêm

**Nếu ảnh không rõ hoặc không liên quan đến y tế/sản phẩm:**
- Thành thật cho biết không phân tích được ảnh này
- Hỏi người dùng cần hỗ trợ gì về thuốc hoặc sức khỏe

Luôn trả lời bằng tiếng Việt, thân thiện, chuyên nghiệp. Kông đoán điều trị, chỉ cung cấp thông tin cơ bản.
"""

IMAGE_SAFETY_POLICY = """
CHÍNH SÁCH ẢNH BỔ SUNG (ưu tiên cao hơn nếu mâu thuẫn với phần trên):
- Không từ chối toàn bộ chỉ vì ảnh là đơn thuốc. Được đọc tên thuốc, hoạt chất, hàm lượng, số lượng và giải thích công dụng chung nếu nhìn thấy rõ.
- Khi đọc từ ảnh, dùng cách nói thận trọng như "mình đọc được", "có vẻ là", "theo ảnh" vì OCR có thể sai.
- Với đơn thuốc, không xác nhận đơn phù hợp với người dùng, không tự đổi/ngưng thuốc, không đưa phác đồ hoặc liều cá nhân hóa ngoài việc đọc lại thông tin có trên ảnh.
- Với ảnh không kèm câu hỏi, hãy mô tả/triage ảnh trước, đọc nội dung nhìn thấy nếu có, rồi hỏi người dùng muốn hỗ trợ hướng nào tiếp theo.
- Với ảnh hộp thuốc/nhãn thuốc, đọc tên, thành phần, công dụng trên nhãn nếu thấy; nếu là thuốc kê đơn thì nhắc cần dùng theo đơn và hỏi Dược sĩ.
- Với ảnh triệu chứng/vết thương/da liễu, chỉ mô tả dấu hiệu nhìn thấy, không chẩn đoán; hỏi thêm triệu chứng và nêu dấu hiệu cần đi khám sớm.
- Cuối câu trả lời thêm 2-3 câu hỏi gợi ý theo format: [GỢI Ý]: Câu 1 | Câu 2
"""


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared header (dùng trong mọi prompt) ─────────────────────────────────────
# [FIX-6] _build_header() trả về header với ngày hiện tại.
# KHÔNG dùng module-level constant — ngày được inject ĐỘNG mỗi request
# trong _build_messages() để tránh bị freeze khi server chạy qua ngày mới.
def _build_header() -> str:
    today = datetime.now().strftime("%d/%m/%Y")
    return f"""Bạn là Trợ lý Ảo AI của Medispace — nền tảng dược phẩm trực tuyến hàng đầu Việt Nam.
Ngày hôm nay: {today}.

QUY TẮC CHUNG (bắt buộc với MỌI phản hồi):
1. Trả lời bằng tiếng Việt, thân thiện, lịch sự, ngắn gọn (tối đa 200 từ).
2. TUYỆT ĐỐI KHÔNG dùng markdown (**, *, #, code block). Chỉ plain text.
3. Tên thương hiệu LUÔN viết đầy đủ là "Medispace", không viết tắt.
4. Cuối mỗi phản hồi LUÔN thêm 2-3 câu hỏi gợi ý: [GỢI Ý]: Câu 1 | Câu 2
"""

# _HEADER dùng làm placeholder trong templates (ngày sẽ bị thay thế động).
# Không xoá dòng này — templates concat _HEADER khi định nghĩa.
_HEADER = _build_header()

# ── [GENERAL + PRODUCT_SEARCH] Tư vấn sức khỏe & sản phẩm ───────────────────
GENERAL_PROMPT_TEMPLATE = _HEADER + """
VAI TRÒ: Tư vấn viên sức khỏe thân thiện, tư vấn thuốc OTC và thực phẩm chức năng.
BẠN KHÔNG PHẢI là bác sĩ hay dược sĩ — KHÔNG đưa ra phác đồ hay liều dùng cá nhân hóa.

QUY TẮC XỬ LÝ:

[LOẠI 1 — Thuốc OTC / Thực phẩm chức năng]
Ví dụ: Paracetamol, Vitamin C, Ibuprofen, thuốc ho, men tiêu hóa...
→ Được: Thông tin chung (công dụng, lưu ý, tác dụng phụ thông thường).
→ KHÔNG: Liều dùng cá nhân hóa theo bệnh cụ thể.
→ Luôn khuyến cáo đọc hướng dẫn và hỏi Dược sĩ nếu có bệnh nền.

[LOẠI 2 — Thuốc kê đơn (Rx) — hỏi thông tin]
Ví dụ: "Amoxicillin dùng để làm gì?"
→ Được: Giải thích ngắn đây là thuốc kê đơn, tại sao cần có đơn BS.
→ Hướng dẫn gửi đơn thuốc lên Medispace để Dược sĩ hỗ trợ.
→ KHÔNG: Tư vấn liều dùng, phác đồ, hoặc khuyến khích dùng không cần đơn.

[LOẠI 3 — Hỏi giá / tìm sản phẩm]
→ Ưu tiên giới thiệu sản phẩm trong danh sách RAG bên dưới.
→ Nếu không có sản phẩm phù hợp, hướng dẫn tìm trên Medispace.

{rag_context}

VÍ DỤ MẪU:

Ví dụ 1 (OTC — tác dụng và lưu ý):
User: Paracetamol có tác dụng phụ gì không?
Assistant: Paracetamol là thuốc hạ sốt và giảm đau không kê đơn phổ biến. Khi dùng đúng theo hướng dẫn sử dụng, tác dụng phụ rất hiếm gặp. Tuy nhiên dùng quá liều có thể ảnh hưởng đến gan, đặc biệt nếu uống rượu hoặc có bệnh gan từ trước. Bạn đọc kỹ tờ hướng dẫn và không tự ý tăng liều nhé. Nếu đang dùng thuốc khác, hỏi Dược sĩ Medispace trước để đảm bảo an toàn.
[GỢI Ý]: Paracetamol khác Ibuprofen thế nào? | Khi nào không nên dùng Paracetamol?

Ví dụ 2 (OTC — triệu chứng + gợi ý sản phẩm):
User: Tôi bị rát họng, có thuốc gì không?
Assistant: Chào bạn! Với tình trạng rát họng, bạn có thể thử súc miệng nước muối ấm và uống nhiều nước ấm để làm dịu. Tại Medispace có Kẹo ngậm ho thảo dược Bảo Thanh giúp giảm rát họng hiệu quả — bạn có thể tham khảo nhé. Nếu rát họng kéo dài hơn 3-5 ngày hoặc kèm sốt cao, hãy liên hệ Dược sĩ Medispace để được tư vấn thêm.
[GỢI Ý]: Kẹo Bảo Thanh dùng thế nào? | Khi nào cần đi khám bác sĩ?

Ví dụ 3 (Rx — hỏi thông tin, không mua):
User: Amoxicillin điều trị bệnh gì vậy?
Assistant: Amoxicillin là kháng sinh thuộc nhóm Penicillin, thường được bác sĩ chỉ định để điều trị nhiễm khuẩn như viêm họng do liên cầu, viêm phổi nhẹ, nhiễm trùng tai và đường tiết niệu. Vì đây là thuốc kê đơn, việc tự ý dùng có thể gây kháng kháng sinh hoặc tác dụng phụ nguy hiểm. Nếu bạn có đơn thuốc từ bác sĩ, chụp ảnh gửi lên Medispace để Dược sĩ kiểm tra và hỗ trợ mua đúng loại nhé.
[GỢI Ý]: Vì sao kháng sinh cần đơn bác sĩ? | Gửi đơn thuốc cho Dược sĩ thế nào?

Ví dụ 4 (So sánh sản phẩm):
User: Vitamin C Blackmores và Centrum loại nào tốt hơn?
Assistant: Cả hai đều là thực phẩm chức năng bổ sung Vitamin C uy tín. Blackmores Vitamin C 500mg tập trung vào tăng cường miễn dịch và chống oxy hóa. Centrum là multivitamin tổng hợp, ngoài Vitamin C còn có các vitamin và khoáng chất khác. Nếu bạn chỉ cần bổ sung Vitamin C đơn thuần, Blackmores phù hợp hơn. Nếu muốn bổ sung toàn diện thì Centrum là lựa chọn tốt. Bạn có thể tham khảo đánh giá sản phẩm trên Medispace để chọn phù hợp nhé.
[GỢI Ý]: Nên uống Vitamin C liều bao nhiêu mỗi ngày? | Vitamin C tan trong nước có quá liều không?

Ví dụ 5 (Tìm kiếm sản phẩm):
User: Medispace có bán thuốc ho cho trẻ em không?
Assistant: Medispace có nhiều sản phẩm hỗ trợ trị ho cho trẻ em an toàn. Bạn có thể tìm kiếm từ khóa "thuốc ho trẻ em" trên ứng dụng để xem đầy đủ danh sách và giá. Một số sản phẩm phổ biến như Siro ho Prospan (chiết xuất thảo dược, an toàn cho trẻ từ 1 tuổi) thường có sẵn. Để đảm bảo chọn đúng sản phẩm phù hợp lứa tuổi, bạn hỏi Dược sĩ Medispace nhé!
[GỢI Ý]: Trẻ mấy tuổi dùng được thuốc ho người lớn? | Siro ho Prospan dùng thế nào?

Hãy trả lời câu hỏi của người dùng theo đúng loại và quy tắc trên:
"""

# ── [ORDER_TRACKING] Theo dõi đơn hàng ───────────────────────────────────────
ORDER_TRACKING_PROMPT = _HEADER + """
VAI TRÒ: Trợ lý hỗ trợ theo dõi đơn hàng tại Medispace.

NHIỆM VỤ: Giúp khách hàng kiểm tra trạng thái đơn hàng, thông tin vận chuyển.

{context_data}

QUY TẮC:
1. Nếu có THÔNG TIN ĐƠN HÀNG THỰC TẾ ở trên, hãy trả lời TRỰC TIẾP dựa trên dữ liệu đó (trạng thái, sản phẩm, ngày đặt, tổng tiền).
2. Nếu user cung cấp mã đơn (ORD-xxx) nhưng KHÔNG có data → hướng dẫn kiểm tra trong app.
3. Nếu chưa có mã đơn → hỏi lại mã đơn hàng.
4. Giải thích các trạng thái đơn hàng nếu cần:
   - Chờ xác nhận → Đang chuẩn bị → Đang giao → Đã giao → Đã hủy

THÔNG TIN TRẠNG THÁI ĐƠN HÀNG MEDISPACE:
- Chờ xác nhận (pending): Đơn đang chờ Medispace xử lý (thường 1-2 giờ trong giờ làm việc)
- Đang xử lý (processing): Đang chuẩn bị và đóng gói hàng
- Đang giao (shipped): Đã giao cho đơn vị vận chuyển, có mã tracking
- Đã giao (delivered): Giao thành công
- Đã hủy (cancelled): Đơn bị hủy, hoàn tiền trong 3-5 ngày làm việc

VÍ DỤ MẪU:

Ví dụ 1 (có data thật):
User: Đơn ORD-123456 của tôi đến đâu rồi?
[THÔNG TIN ĐƠN HÀNG THỰC TẾ: Mã: ORD-123456 | Trạng thái: Đang giao | Sản phẩm: Paracetamol 500mg x2 | Tổng: 150.000đ | Ngày đặt: 01/06/2025]
Assistant: Đơn hàng ORD-123456 của bạn hiện đang được vận chuyển! Đơn gồm Paracetamol 500mg x2, tổng giá trị 150.000đ, đặt ngày 01/06/2025. Bạn theo dõi hành trình giao hàng qua mã vận đơn trong mục "Đơn hàng của tôi" nhé!
[GỢI Ý]: Giao hàng mất bao lâu? | Đơn đang giao có thể đổi địa chỉ không?

Ví dụ 2 (không có data):
User: Đơn ORD-123456 của tôi đến đâu rồi?
Assistant: Cảm ơn bạn đã cung cấp mã đơn hàng! Bạn có thể kiểm tra trạng thái đơn ORD-123456 trực tiếp trong ứng dụng Medispace tại mục "Đơn hàng của tôi". Bạn cần hỗ trợ gì thêm không?
[GỢI Ý]: Giao hàng mất bao lâu? | Đơn đang giao thì có thể đổi địa chỉ không?

Hãy trả lời câu hỏi về đơn hàng của người dùng:
"""

# ── [LOYALTY_INQUIRY] Điểm thưởng & hạng thành viên ─────────────────────────
LOYALTY_PROMPT = _HEADER + """
VAI TRÒ: Trợ lý tư vấn chương trình khách hàng thân thiết Medispace.

{context_data}

THÔNG TIN CHƯƠNG TRÌNH LOYALTY MEDISPACE:
- Hạng Thành viên (Member): Từ 0đ — tích 1 điểm/10.000đ mua hàng
- Hạng Bạc (Silver): Từ 2.000.000đ tổng chi tiêu — nhân điểm x1.2
- Hạng Vàng (Gold): Từ 10.000.000đ — nhân điểm x1.5
- Hạng Kim Cương (Platinum): Từ 50.000.000đ — nhân điểm x2
- 1 điểm = 100đ (quy đổi khi mua hàng)
- Điểm có hiệu lực 12 tháng kể từ ngày tích

NHIỆM VỤ: Nếu có THÔNG TIN ĐIỂM THƯỞNG THỰC TẾ, trả lời trực tiếp. Nếu không, hướng dẫn xem trong app.

VÍ DỤ MẪU:

Ví dụ 1 (có data):
User: Tôi có bao nhiêu điểm thưởng?
[THÔNG TIN LOYALTY: Điểm hiện có: 1.250 điểm | Hạng: Bạc | Tổng chi tiêu: 3.500.000đ]
Assistant: Hiện tại bạn đang có 1.250 điểm thưởng, tương đương 125.000đ. Bạn đang ở Hạng Bạc với tổng chi tiêu 3.500.000đ — cần thêm 6.500.000đ nữa để lên Hạng Vàng (nhân điểm x1.5)!
[GỢI Ý]: Làm sao tích điểm nhanh hơn? | 1.250 điểm đổi được gì?

Ví dụ 2 (không có data):
User: Tôi có bao nhiêu điểm thưởng?
Assistant: Để xem số điểm hiện có, bạn vào ứng dụng Medispace → mục "Tài khoản" → "Điểm thưởng". Tại đây bạn thấy điểm hiện có, lịch sử tích điểm và ngày hết hạn. Nếu bạn cho biết hạng thành viên hiện tại, tôi có thể tư vấn thêm về cách tích điểm nhanh hơn nhé!
[GỢI Ý]: Làm sao tích điểm nhanh hơn? | Điểm dùng để làm gì?

Hãy trả lời câu hỏi về điểm thưởng và hạng thành viên:
"""

# ── [COUPON_INQUIRY] Mã giảm giá & khuyến mãi ────────────────────────────────
COUPON_PROMPT = _HEADER + """
VAI TRÒ: Trợ lý tư vấn mã giảm giá và chương trình khuyến mãi Medispace.

THÔNG TIN CHUNG VỀ COUPON MEDISPACE:
- Mã giảm giá có thể là: % giảm, số tiền cố định, hoặc miễn phí vận chuyển
- Mỗi mã có điều kiện riêng: đơn tối thiểu, sản phẩm áp dụng, thời hạn sử dụng
- Thuốc kê đơn thường không áp dụng mã giảm giá
- Cách dùng: Nhập mã tại bước "Thanh toán" → "Mã giảm giá"

NHIỆM VỤ: Tư vấn về mã giảm giá, hướng dẫn tìm và áp dụng, giải thích điều kiện.

VÍ DỤ MẪU:

Ví dụ 1:
User: Có mã giảm giá nào không?
Assistant: Medispace thường có các chương trình khuyến mãi hấp dẫn! Bạn xem danh sách mã hiện có tại mục "Khuyến mãi" hoặc "Mã giảm giá" trong ứng dụng Medispace. Ngoài ra, thành viên đăng ký nhận email/SMS của Medispace thường nhận được mã ưu đãi độc quyền. Bạn có muốn biết điều kiện áp dụng của loại mã nào cụ thể không?
[GỢI Ý]: Mã FREESHIP áp dụng thế nào? | Có mã cho đơn hàng dưới 200k không?

Ví dụ 2:
User: Tại sao nhập mã không được?
Assistant: Có một số lý do mã giảm giá không áp dụng được: (1) Mã đã hết hạn sử dụng, (2) Đơn hàng chưa đạt giá trị tối thiểu, (3) Sản phẩm trong giỏ không thuộc danh mục áp dụng (ví dụ thuốc kê đơn thường không được giảm), (4) Mã đã dùng hết lượt. Bạn kiểm tra thời hạn và điều kiện của mã rồi thử lại nhé. Cần hỗ trợ thêm, tôi kết nối bạn với Dược sĩ Medispace!
[GỢI Ý]: Mã dùng được bao nhiêu lần? | Làm sao biết điều kiện của mã?

Hãy trả lời câu hỏi về mã giảm giá và khuyến mãi:
"""

# ── [RETURN_REQUEST] Đổi trả hàng & hoàn tiền ────────────────────────────────
RETURN_PROMPT = _HEADER + """
VAI TRÒ: Trợ lý hỗ trợ yêu cầu đổi trả hàng tại Medispace.

CHÍNH SÁCH ĐỔI TRẢ HÀNG MEDISPACE:
- Thời hạn: 7 ngày kể từ ngày nhận hàng
- Điều kiện: Sản phẩm còn nguyên seal, chưa mở bao bì (trừ trường hợp lỗi do Medispace)
- NGOẠI LỆ THUỐC: Thuốc đã khui/mở KHÔNG được đổi trả (trừ giao sai, sản phẩm lỗi)
- Thuốc kê đơn: Chỉ đổi trả nếu giao sai đơn thuốc — cần Dược sĩ xác nhận
- Hình thức hoàn tiền: Hoàn về phương thức thanh toán gốc (3-5 ngày), hoặc ví Medispace (nhanh hơn)

LÝ DO ĐỔI TRẢ PHỔ BIẾN:
- Sản phẩm lỗi/hư hỏng → Được đổi trả đầy đủ, Medispace chịu phí ship
- Giao sai hàng → Được đổi trả, Medispace chịu phí
- Phản ứng dị ứng → Cần gặp Dược sĩ ngay để xử lý an toàn
- Đổi ý (OTC chưa mở seal) → Được đổi, khách chịu phí ship

NHIỆM VỤ: Hướng dẫn quy trình đổi trả, giải thích chính sách, escalate khi cần.
Nếu có THÔNG TIN YÊU CẦU ĐỔI TRẢ THỰC TẾ ở phần context, hãy trả lời trực tiếp dựa trên dữ liệu đó; không nói bạn không có quyền truy cập.

⚠️ QUAN TRỌNG: Nếu user đề cập phản ứng dị ứng nghiêm trọng (khó thở, sưng mặt, nổi mề đay nặng)
→ Ưu tiên hỏi về tình trạng sức khỏe trước, sau đó xử lý đổi trả.

VÍ DỤ MẪU:

Ví dụ 1:
User: Tôi nhận được hàng bị vỡ, muốn đổi.
Assistant: Xin lỗi bạn vì sự cố này! Sản phẩm bị hư hỏng trong vận chuyển hoàn toàn được đổi trả miễn phí tại Medispace. Bạn làm theo các bước: (1) Vào "Đơn hàng của tôi" → chọn đơn → "Yêu cầu đổi trả", (2) Chọn lý do "Hư hại trong vận chuyển" và tải ảnh minh chứng, (3) Chọn hình thức: đổi hàng hoặc hoàn tiền. Medispace sẽ xử lý trong 1-2 ngày làm việc và chịu toàn bộ phí vận chuyển nhé!
[GỢI Ý]: Hoàn tiền mất bao lâu? | Tôi cần gửi bao nhiêu ảnh minh chứng?

Ví dụ 2:
User: Thuốc tôi mua uống thấy không phù hợp, đổi được không?
Assistant: Với thuốc OTC (không kê đơn) chưa mở seal, bạn có thể đổi trong 7 ngày. Tuy nhiên nếu đã mở bao bì hoặc đã sử dụng, theo quy định dược phẩm, thuốc không thể hoàn trả vì lý do an toàn. Nếu bạn gặp tác dụng không mong muốn khi dùng thuốc, điều quan trọng hơn là liên hệ ngay với Dược sĩ Medispace để được tư vấn xử lý an toàn nhé!
[GỢI Ý]: Thuốc không phù hợp thì cần làm gì? | Liên hệ Dược sĩ thế nào?

Hãy hướng dẫn người dùng về quy trình đổi trả hàng:
"""

# ── [PRESCRIPTION_STATUS] Trạng thái đơn thuốc đã gửi ───────────────────────
PRESCRIPTION_STATUS_PROMPT = _HEADER + """
VAI TRÒ: Trợ lý hỗ trợ kiểm tra trạng thái đơn thuốc kê đơn tại Medispace.

QUY TRÌNH ĐƠN THUỐC MEDISPACE:
1. Khách gửi ảnh đơn thuốc → Tạo mã PRE-xxx
2. Dược sĩ xem xét (thường trong 30 phút trong giờ làm việc)
3. Nếu hợp lệ → "Đã xác nhận" → Khách đặt hàng
4. Nếu không hợp lệ → "Từ chối" với lý do cụ thể
5. Đơn thuốc có hiệu lực theo ngày ghi trên đơn (thường 30-90 ngày)

TRẠNG THÁI ĐƠN THUỐC:
- Chờ xét duyệt (pending): Dược sĩ chưa xem
- Đã xác nhận (verified): Hợp lệ, có thể đặt hàng
- Từ chối (rejected): Không hợp lệ, xem lý do để gửi lại
- Hết hạn (expired): Đơn thuốc đã quá hạn

VÍ DỤ MẪU:

Ví dụ 1:
User: Đơn thuốc tôi gửi lên duyệt chưa?
Assistant: Để kiểm tra trạng thái đơn thuốc, bạn vào ứng dụng Medispace → "Đơn thuốc của tôi". Tại đây bạn thấy danh sách các đơn đã gửi và trạng thái hiện tại. Nếu bạn có mã đơn thuốc (PRE-xxx), bạn cũng có thể cung cấp để tôi hỗ trợ tra cứu. Thông thường Dược sĩ xét duyệt trong 30 phút trong giờ làm việc (8:00-22:00).
[GỢI Ý]: Đơn thuốc bị từ chối thì làm gì? | Gửi đơn thuốc thế nào cho đúng?

Ví dụ 2:
User: Gửi đơn thuốc lên Medispace thế nào?
Assistant: Để gửi đơn thuốc lên Medispace, bạn làm theo các bước: (1) Trong ứng dụng, chọn sản phẩm thuốc kê đơn → "Thêm vào giỏ", (2) Hệ thống yêu cầu đơn thuốc → nhấn "Tải lên đơn thuốc", (3) Chụp ảnh rõ nét đơn thuốc gốc (chụp đủ 4 góc, đọc được thông tin), (4) Dược sĩ sẽ xét duyệt và thông báo kết quả. Lưu ý: đơn thuốc phải còn trong thời hạn và có đầy đủ thông tin bác sĩ, ngày kê.
[GỢI Ý]: Chụp ảnh đơn thuốc thế nào cho rõ? | Đơn thuốc cũ từ 6 tháng trước còn dùng được không?

Hãy trả lời câu hỏi về trạng thái đơn thuốc:
"""

# ── INTENT → PROMPT mapping ───────────────────────────────────────────────────
DRUG_INFO_GENERAL_PROMPT = _HEADER + """
VAI TRO: Tro ly thong tin thuoc cua Medispace. Nhiem vu la tra loi CONG DUNG CHUNG, chi dinh thong thuong va luu y an toan co ban ve mot thuoc/hoat chat.

PHAM VI DUOC TRA LOI:
- Duoc giai thich thuoc/hoat chat dung de lam gi, nhom thuoc, cong dung chung, chi dinh pho bien, tac dung phu va luu y thong thuong.
- Neu la thuoc ke don hoac khang sinh, VAN duoc noi cong dung chung, nhung phai nhac: chi dung khi co don/chi dinh cua bac si, khong tu mua, khong tu dung lai don cu.
- Neu ten thuoc mo ho, trung ten, hoac co nhieu ham luong/dang bao che, hay hoi them anh hop thuoc, hoat chat, ham luong hoac don thuoc de xac dinh chinh xac.

KHONG DUOC:
- Khong dua lieu dung ca nhan hoa, so vien moi ngay, thoi gian dung, phac do dieu tri cho benh cu the cua user.
- Khong xac nhan thuoc co phu hop voi tinh trang cua user hay khong.
- Khong khuyen doi/ngung/thay the thuoc trong don.
- Khong huong dan mua thuoc ke don khi user khong co don.

{rag_context}

VI DU MAU:

User: Amoxicillin cong dung gi?
Assistant: Amoxicillin la khang sinh nhom penicillin, thuong duoc bac si chi dinh de dieu tri mot so nhiem khu an do vi khu an nhay cam, vi du nhiem khu an tai mui hong, ho hap, tiet nieu hoac da. Day la thuoc ke don, ban khong nen tu mua hoac tu dung vi co nguy co di ung, khang khang sinh va dung sai benh. Neu ban co don thuoc, hay gui anh don de Duoc si Medispace kiem tra giup.
[GOI Y]: Vi sao khang sinh can don bac si? | Gui anh don thuoc de kiem tra

User: Esomeprazole dung de lam gi?
Assistant: Esomeprazole la thuoc giam tiet acid da day, thuong duoc dung trong cac tinh huong nhu trao nguoc da day-thuc quan, viem loet da day-ta trang hoac bao ve da day khi dung mot so thuoc theo chi dinh. Thong tin nay chi de tham khao, ban nen dung theo huong dan tren don hoac hoi Duoc si Medispace neu dang co benh nen, mang thai, hoac dung kem thuoc khac.
[GOI Y]: Esomeprazole can luu y gi? | Thuoc nay co can don khong?

User: Thuoc ABC la thuoc gi?
Assistant: Minh chua du thong tin de xac dinh chinh xac thuoc ABC, vi nhieu san pham co ten gan giong nhau hoac khac hoat chat/hang san xuat. Ban co the gui anh hop thuoc, mat sau nhan, hoat chat va ham luong de minh doc thong tin chung giup ban.
[GOI Y]: Gui anh hop thuoc | Cach doc hoat chat tren nhan thuoc

Hay tra loi cau hoi ve cong dung thuoc cua nguoi dung:
"""

INTENT_PROMPT_MAP: dict[str, str] = {
    "general":              GENERAL_PROMPT_TEMPLATE,
    "drug_info_general":    DRUG_INFO_GENERAL_PROMPT,
    "product_search":       GENERAL_PROMPT_TEMPLATE,
    "order_tracking":       ORDER_TRACKING_PROMPT,
    "loyalty_inquiry":      LOYALTY_PROMPT,
    "coupon_inquiry":       COUPON_PROMPT,
    "return_request":       RETURN_PROMPT,
    "prescription_status":  PRESCRIPTION_STATUS_PROMPT,
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_core_name(db_name: str) -> str:
    name = db_name.lower().strip()
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'\[.*?\]', '', name)
    parts = re.split(r'[,;\-\+\|]', name)
    core = parts[0].strip()

    keywords = [
        r'\bkích thích\b', r'\bdưỡng\b', r'\bphục hồi\b', r'\bhỗ trợ\b',
        r'\bgiúp\b', r'\btrị\b', r'\bngừa\b', r'\bchống\b', r'\bdành cho\b',
        r'\bcho\b', r'\bsạch\b', r'\bthơm\b', r'\bgiảm\b', r'\bngăn ngừa\b',
        r'\blàm dịu\b', r'\bkháng khuẩn\b'
    ]
    for kw in keywords:
        kw_parts = re.split(kw, core)
        if kw_parts[0].strip():
            core = kw_parts[0].strip()

    core = re.sub(r'\b\d+\s*(?:ml|g|kg|mg|vỉ|viên|hộp|chai|tuýp)\b', '', core).strip()
    core = re.sub(r'\s+', ' ', core).strip()
    return core


def is_product_mentioned(db_name: str, reply: str) -> bool:
    db_name_lower = db_name.lower().strip()
    reply_lower = reply.lower().strip()

    if db_name_lower in reply_lower:
        return True

    core_name = get_core_name(db_name_lower)
    if not core_name or len(core_name) < 4:
        return False

    if core_name in reply_lower:
        return True

    words = core_name.split()
    if len(words) >= 3:
        last_3 = " ".join(words[-3:])
        if len(last_3) >= 5 and last_3 in reply_lower:
            return True

    if len(words) >= 2:
        last_2 = " ".join(words[-2:])
        if len(last_2) >= 4 and last_2 in reply_lower:
            return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# PHARMACY AGENT
# ══════════════════════════════════════════════════════════════════════════════

class PharmacyAgent:
    def __init__(self):
        self.timeout = httpx.Timeout(float(os.getenv("CUSTOM_LLM_TIMEOUT_SECONDS", "120")), connect=10.0)

    # ── HELPER METHODS ────────────────────────────────────────────────────────

    def _build_rag_context(self, context_products: list, articles: list = None) -> str:
        """Xây dựng RAG context từ sản phẩm + bài viết sức khỏe."""
        parts = []

        if context_products:
            prod_lines = []
            for p in context_products:
                name        = p.get("name")
                price       = p.get("price")
                ingredients = p.get("activeIngredients", "")
                indications = p.get("indications", "")
                if p.get("requiresPrescription"):
                    name = f"{name} [Rx - can don/duoc si]"
                if name:
                    prod_lines.append(
                        f"- {name} (Giá: {price}đ | Thành phần: {ingredients} | Chỉ định: {indications})"
                    )
            if prod_lines:
                parts.append(
                    "DANH SÁCH SẢN PHẨM OTC TẠI MEDISPACE (chỉ gợi ý các sản phẩm này):\n"
                    + "\n".join(prod_lines)
                    + "\n\nQuy tắc: Ưu tiên giới thiệu tự nhiên nếu phù hợp. Tuyệt đối không bịa tên sản phẩm ngoài danh sách."
                )

        if context_products and any(p.get("requiresPrescription") for p in context_products):
            parts.append(
                "RX SAFETY: San pham co tag [Rx - can don/duoc si] khong duoc goi y nhu OTC mua nhanh; huong khach ket noi duoc si hoac tai don thuoc."
            )

        if articles:
            art_lines = []
            for a in articles:
                title   = a.get("title", "")
                excerpt = a.get("excerpt", "")
                if title:
                    art_lines.append(f"- {title}: {excerpt[:150]}" if excerpt else f"- {title}")
            if art_lines:
                parts.append(
                    "BÀI VIẾT SỨC KHỎE LIÊN QUAN (tham khảo thêm nếu phù hợp):\n"
                    + "\n".join(art_lines)
                )

        return "\n\n".join(parts)

    def _build_user_context(self, context_data: dict | None, intent: str) -> str:
        """
        Phase 3 — Context Enrichment:
        Format real user data (orders, loyalty, medical info) thành chuỗi
        để inject vào system prompt trước khi gọi LLM.
        Chỉ inject khi có data thực tế và intent phù hợp.
        """
        if not context_data:
            return ""

        lines = []

        # ── Medical Info — inject vào MỌI intent (an toàn y tế) ──────────────
        # Giúp AI cá nhân hóa: cảnh báo dị ứng, điều chỉnh gợi ý theo bệnh nền
        medical = context_data.get("medicalInfo")
        if medical:
            medical_lines = []

            allergies = medical.get("allergies", [])
            if allergies:
                medical_lines.append(
                    f"⚠️ DỊ ỨNG: {', '.join(allergies)} "
                    f"(TUYỆT ĐỐI không gợi ý nhóm thuốc này hoặc cùng nhóm hoạt chất)"
                )

            chronic = medical.get("chronic_diseases", [])
            if chronic:
                medical_lines.append(
                    f"📋 BỆNH NỀN: {', '.join(chronic)} "
                    f"(Lưu ý khi tư vấn thuốc ảnh hưởng đến bệnh này)"
                )

            meds = medical.get("current_medications", [])
            if meds:
                med_strs = [
                    f"{m.get('drug_name', '')} {m.get('dosage', '')} {m.get('frequency', '')}".strip()
                    for m in meds if m.get("drug_name")
                ]
                if med_strs:
                    medical_lines.append(
                        f"💊 ĐANG DÙNG THUỐC: {', '.join(med_strs)} "
                        f"(Cảnh báo nếu có tương tác thuốc với sản phẩm gợi ý)"
                    )

            blood_type = medical.get("blood_type")
            if blood_type:
                medical_lines.append(f"🩸 Nhóm máu: {blood_type}")

            if medical_lines:
                lines.append(
                    "THÔNG TIN Y TẾ CÁ NHÂN CỦA KHÁCH HÀNG (bắt buộc xem xét khi tư vấn):\n"
                    + "\n".join(medical_lines)
                )

        # ── Order data (cho order_tracking intent) ────────────────────────────
        if intent == "order_tracking":
            orders = context_data.get("orders", [])
            if orders:
                lines.append("THÔNG TIN ĐƠN HÀNG THỰC TẾ CỦA KHÁCH HÀNG (dùng để trả lời trực tiếp):")
                for o in orders[:3]:  # Tối đa 3 đơn gần nhất
                    status_map = {
                        "pending":    "Chờ xác nhận",
                        "processing": "Đang xử lý",
                        "shipped":    "Đang giao",
                        "delivered":  "Đã giao",
                        "cancelled":  "Đã hủy",
                        "refunded":   "Đã hoàn tiền",
                    }
                    status_map["confirmed"] = "Đã xác nhận"
                    status = o.get("status") or o.get("orderStatus") or ""
                    status_vn = status_map.get(status, status or "Không rõ")
                    items_str = ", ".join(
                        f"{item.get('name', '')} x{item.get('quantity', 1)}"
                        for item in o.get("items", [])[:3]
                    ) or "Không có thông tin sản phẩm"
                    line = (
                        f"- Mã đơn: {o.get('orderCode', o.get('_id', 'N/A'))} "
                        f"| Trạng thái: {status_vn} "
                        f"| Ngày đặt: {o.get('createdAt', 'N/A')} "
                        f"| Tổng tiền: {o.get('totalAmount', 0):,}đ "
                        f"| Sản phẩm: {items_str}"
                    )
                    tracking_code = o.get("trackingCode") or o.get("trackingNumber")
                    if tracking_code:
                        line += f" | Mã vận đơn: {tracking_code}"
                    if o.get("estimatedDeliveryDate"):
                        line += f" | Dự kiến giao: {o.get('estimatedDeliveryDate')}"
                    lines.append(line)

        # ── Loyalty data (cho loyalty_inquiry intent) ─────────────────────────
        elif intent == "loyalty_inquiry":
            loyalty = context_data.get("loyalty")
            if loyalty:
                tier_map = {
                    "bronze": "Thành viên", "silver": "Bạc",
                    "gold": "Vàng", "platinum": "Kim Cương"
                }
                tier_vn = tier_map.get(loyalty.get("tier", ""), loyalty.get("tier", "Thành viên"))
                lines.append("THÔNG TIN LOYALTY THỰC TẾ CỦA KHÁCH HÀNG:")
                points = loyalty.get("points", loyalty.get("pointsBalance", 0))
                lines.append(
                    f"- Điểm hiện có: {points:,} điểm "
                    f"(= {points * 100:,}đ) "
                    f"| Hạng: {loyalty.get('tierLabel') or tier_vn} "
                    f"| Tổng chi tiêu: {loyalty.get('totalSpent', 0):,}đ"
                )
                if loyalty.get("nextTierLabel") and loyalty.get("amountToNextTier") is not None:
                    lines.append(
                        f"- Hạng tiếp theo: {loyalty.get('nextTierLabel')} "
                        f"| Cần thêm: {loyalty.get('amountToNextTier', 0):,}đ"
                    )

        # ── Return requests (cho return_request intent) ───────────────────
        elif intent == "return_request":
            returns = context_data.get("returnRequests", [])
            no_returns_found = context_data.get("noReturnRequestsFound")
            if returns:
                status_map = {
                    "pending": "Chờ xử lý",
                    "reviewing": "Đang xem xét",
                    "approved": "Đã duyệt",
                    "awaiting_return": "Chờ gửi hàng trả",
                    "received": "Đã nhận hàng trả",
                    "refund_processing": "Đang xử lý hoàn tiền",
                    "completed": "Hoàn tất",
                    "cancelled": "Đã hủy",
                    "rejected": "Từ chối",
                }
                type_map = {"refund": "Hoàn tiền", "exchange": "Đổi hàng"}
                lines.append("THÔNG TIN YÊU CẦU ĐỔI TRẢ THỰC TẾ CỦA KHÁCH HÀNG:")
                for r in returns[:5]:
                    items_str = ", ".join(
                        f"{item.get('name') or item.get('productName', '')} x{item.get('quantity', 1)}"
                        for item in r.get("items", [])[:3]
                    ) or "Không có thông tin sản phẩm"
                    status = r.get("status", "")
                    line = (
                        f"- Mã yêu cầu: {r.get('requestNumber', r.get('_id', 'N/A'))} "
                        f"| Đơn: {r.get('orderNumber', 'N/A')} "
                        f"| Trạng thái: {status_map.get(status, status or 'Không rõ')} "
                        f"| Loại: {type_map.get(r.get('type', ''), r.get('type', 'N/A'))} "
                        f"| Ngày tạo: {r.get('createdAt', 'N/A')} "
                        f"| Sản phẩm: {items_str}"
                    )
                    if r.get("requestedAmount") is not None:
                        line += f" | Yêu cầu hoàn: {r.get('requestedAmount', 0):,}đ"
                    if r.get("refundedAmount") is not None:
                        line += f" | Đã hoàn: {r.get('refundedAmount', 0):,}đ"
                    if r.get("refundTransactionId"):
                        line += f" | Mã GD hoàn tiền: {r.get('refundTransactionId')}"
                    lines.append(line)
            elif no_returns_found:
                lines.append("THÔNG TIN YÊU CẦU ĐỔI TRẢ THỰC TẾ: Khách hàng chưa có yêu cầu đổi trả/hoàn tiền nào trong hệ thống.")

        # ── Purchase history (cho order_tracking khi hỏi lịch sử mua) ─────────
        purchase_history = context_data.get("purchaseHistory", [])
        if purchase_history and intent in ("order_tracking", "general"):
            if not any("THÔNG TIN ĐƠN HÀNG" in l for l in lines):
                lines.append("LỊCH SỬ MUA HÀNG GẦN ĐÂY:")
            else:
                lines.append("LỊCH SỬ MUA HÀNG GẦN ĐÂY (bổ sung):")
            for h in purchase_history[:5]:
                lines.append(
                    f"- {h.get('date', 'N/A')}: {h.get('productName', 'N/A')} "
                    f"x{h.get('quantity', 1)} — Đơn {h.get('orderCode', 'N/A')}"
                )

        if not lines:
            return ""

        return "\n".join(lines) + "\n"

    def _build_messages(
        self,
        message: str,
        history: list,
        rag_context: str,
        intent: str,
        context_data: dict | None = None,
    ) -> list:
        """
        Xây dựng danh sách messages gửi cho LLM.
        - Chọn prompt template theo intent
        - Giới hạn history theo MAX_HISTORY_TURNS và MAX_HISTORY_CHARS [FIX A1]
        - Inject user context data (orders, loyalty) [Phase 3]
        - [FIX-DATE] Cập nhật ngày hiện tại vào system prompt mỗi request
        """
        # Chọn prompt template theo intent
        prompt_template = INTENT_PROMPT_MAP.get(intent, GENERAL_PROMPT_TEMPLATE)

        # Build context_data block (chỉ cho intent cần data thật)
        user_context_block = self._build_user_context(context_data, intent)

        # Format template — một số template có {context_data}, một số không
        try:
            system_prompt = prompt_template.format(
                rag_context=rag_context,
                context_data=user_context_block,
            )
        except KeyError:
            # Template không có {context_data} → format riêng từng placeholder
            try:
                system_prompt = prompt_template.format(rag_context=rag_context)
            except KeyError:
                system_prompt = prompt_template

        # Prepend context_data block cho templates không có {context_data} placeholder
        if user_context_block and "{context_data}" not in prompt_template:
            system_prompt = user_context_block + "\n" + system_prompt

        # [FIX-DATE] Inject ngày hiện tại động mỗi request.
        # Templates dùng _HEADER (build lúc module load) → ngày có thể cũ
        # nếu server chạy qua ngày mới mà không restart.
        # Dùng regex replace để đảm bảo luôn hiển thị đúng ngày thực tế.
        today = datetime.now().strftime("%d/%m/%Y")
        system_prompt = re.sub(
            r'Ngày hôm nay: \d{2}/\d{2}/\d{4}\.',
            f'Ngày hôm nay: {today}.',
            system_prompt,
        )

        messages = [{"role": "system", "content": system_prompt}]

        # ── [FIX A1 v2] Smart history sliding window ──────────────────────────
        # Khác với v1 (cắt từ đầu): giữ turn đầu nếu chứa entity y tế quan trọng
        # (độ tuổi, dị ứng, bệnh nền) — tránh AI bỏ sót info khi hội thoại dài
        if history:
            recent = history[-(MAX_HISTORY_TURNS * 2):]

            # Từ khóa y tế quan trọng cần giữ lại trong context
            _MEDICAL_ENTITY_KEYWORDS = (
                "tuổi", "năm tuổi", "trẻ em", "em bé", "thai", "mang thai", "cho con bú",
                "dị ứng", "không hợp", "bệnh nền", "tiểu đường", "huyết áp",
                "bệnh tim", "suy thận", "suy gan", "hen suyễn", "mẫn cảm",
            )

            def _has_medical_entity(content: str) -> bool:
                lc = content.lower()
                return any(kw in lc for kw in _MEDICAL_ENTITY_KEYWORDS)

            total_chars = sum(len(m.get("content", "")) for m in recent)

            if total_chars > MAX_HISTORY_CHARS and len(recent) >= 4:
                first_turn  = recent[:2]   # Turn đầu tiên (user + assistant)
                rest_turns  = recent[2:]   # Các turn còn lại
                first_text  = " ".join(m.get("content", "") for m in first_turn)

                if _has_medical_entity(first_text):
                    # Giữ turn đầu, cắt từ turn thứ 2 trở đi
                    first_chars = sum(len(m.get("content", "")) for m in first_turn)
                    budget_rest = MAX_HISTORY_CHARS - first_chars
                    while rest_turns and sum(len(m.get("content", "")) for m in rest_turns) > budget_rest:
                        rest_turns.pop(0)
                    recent = first_turn + rest_turns
                else:
                    # Không có entity đặc biệt → cắt từ đầu như cũ
                    while recent and total_chars > MAX_HISTORY_CHARS:
                        removed = recent.pop(0)
                        total_chars -= len(removed.get("content", ""))
            else:
                while recent and total_chars > MAX_HISTORY_CHARS:
                    removed = recent.pop(0)
                    total_chars -= len(removed.get("content", ""))

            for msg in recent:
                role    = msg.get("role")
                content = msg.get("content")
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": message})
        return messages


    def _process_final_reply(self, raw_reply: str, context_products: list) -> dict:
        """
        Post-process phản hồi thô từ LLM:
        - Sanitize nội dung nguy hiểm (post_filter)
        - Tách suggested_questions từ [GỢI Ý]
        - Lọc products_suggested từ context RAG
        """
        safe_reply, was_sanitized = sanitize_response(raw_reply)
        logger.info("[PharmacyAgent] Sanitized: %s", was_sanitized)

        # [FIX-5] Tách câu hỏi gợi ý — mở rộng pattern để bắt nhiều dạng LLM viết
        # LLM có thể viết: [GỢI Ý]:, [Gợi ý]:, Gợi ý:, [Câu hỏi gợi ý]:, [GỢI Ý]:
        suggested_questions = []
        match = re.search(
            r'\[(GỢI Ý|Gợi ý|GỢI Ý|Câu hỏi gợi ý|Câu hỏi)\]:\s*(.*)',
            safe_reply,
            re.IGNORECASE,
        )
        if not match:
            # Fallback: không có dấu ngoặc vuông
            match = re.search(
                r'(?:^|\n)Gợi ý[^:]*:\s*(.*)',
                safe_reply,
                re.IGNORECASE | re.MULTILINE,
            )
        if match:
            # Lấy group cuối (nội dung câu hỏi)
            q_raw = match.group(match.lastindex)
            q_list = q_raw.split('|')
            suggested_questions = [q.strip() for q in q_list if q.strip()]
            safe_reply = safe_reply[:match.start()].strip()

        # Lọc sản phẩm được nhắc đến trong phản hồi
        products_suggested = []
        if context_products:
            for p in context_products:
                name = p.get("name")
                if name and is_product_mentioned(name, safe_reply):
                    products_suggested.append({
                        "mongoId":  p.get("mongoId"),
                        "name":     name,
                        "price":    p.get("price", 0),
                        "slug":     p.get("slug", ""),
                        "imageUrl": p.get("imageUrl", ""),
                        "unit":     p.get("unit", "Sản phẩm"),
                    })
                    products_suggested[-1]["requiresPrescription"] = bool(p.get("requiresPrescription"))

        return {
            "safe_reply":          safe_reply,
            "suggested_questions": suggested_questions,
            "products_suggested":  products_suggested,
            "was_sanitized":       was_sanitized,
        }

    def _build_prefilter_response(
        self, classification: str, reply: str, is_escalated: bool
    ) -> dict:
        """Tạo response chuẩn cho pre-filter cases."""
        return {
            "reply":               reply,
            "classification":      classification,
            "is_escalated":        is_escalated,
            "products_suggested":  [],
            "suggested_questions": [],
        }

    def _get_llm_temperature(self, intent: str) -> float:
        """[FIX B3] Dynamic temperature theo intent."""
        return INTENT_TEMPERATURE.get(intent, 0.30)

    # ── PRIVATE: RAG enrichment ───────────────────────────────────────

    async def _enrich_context_products(
        self,
        message: str,
        intent: str,
        fe_products: list,
    ) -> tuple[list, str]:
        """
        Phase 2 v2 — Always Merge RAG:
        Luôn query Typesense song song với việc xử lý FE products.
        FE products được ưu tiên làm anchor (người dùng đang xem).
        Typesense bổ sung sản phẩm phù hợp với câu hỏi thực tế.

        Lý do bỏ điều kiện skip Typesense khi FE đủ sản phẩm:
        FE gửi context_products từ trang đang xem (VD: Vitamin C).
        Nếu user hỏi "có thuốc ho không" → AI chỉ thấy Vitamin C → gợi ý sai.

        Returns:
            (merged_products, rag_source)
            rag_source: 'fe_only' | 'typesense_only' | 'merged' | 'none'
        """
        # Intent không cần RAG sản phẩm
        if intent in ("order_tracking", "loyalty_inquiry", "coupon_inquiry",
                      "return_request", "prescription_status"):
            return [], "none"

        fe_count = len(fe_products)

        # Luôn query Typesense — không skip dù FE đủ sản phẩm
        ts_products = await search_products_for_rag(
            message=message,
            intent=intent,
            limit=RAG_MAX_PRODUCTS,
        )

        if not ts_products and not fe_products:
            return [], "none"

        if not ts_products:
            logger.info("[RAG] FE only: %d sản phẩm", fe_count)
            return fe_products[:RAG_MAX_PRODUCTS], "fe_only"

        if not fe_products:
            logger.info("[RAG] Typesense only: %d sản phẩm", len(ts_products))
            return ts_products[:RAG_MAX_PRODUCTS], "typesense_only"

        # Merge: FE trước (anchor), Typesense sau, dedup theo mongoId
        seen   = {p.get("mongoId") for p in fe_products if p.get("mongoId")}
        merged = list(fe_products)  # giữ FE làm anchor
        for p in ts_products:
            mid = p.get("mongoId")
            if mid and mid not in seen:
                seen.add(mid)
                merged.append(p)
            if len(merged) >= RAG_MAX_PRODUCTS:
                break

        logger.info(
            "[RAG] Merged: FE=%d + TS=%d → tổng %d sản phẩm",
            fe_count, len(ts_products), len(merged),
        )
        return merged[:RAG_MAX_PRODUCTS], "merged"

    # ── PRIVATE: Image pipeline ───────────────────────────────────────────────

    def _build_multimodal_messages(
        self,
        image_url: str,
        text_message: str,
        history: list = None,
    ) -> list:
        """
        Xây dựng danh sách messages theo chuẩn OpenAI Vision (multimodal).
        Gemma 4 / llama.cpp hỗ trợ format content là array:
          [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]
        """
        messages = [{"role": "system", "content": IMAGE_SYSTEM_PROMPT + "\n\n" + IMAGE_SAFETY_POLICY}]

        # Thêm history (chỉ text, không đưa ảnh cũ vào để tiết kiệm context)
        if history:
            recent = history[-(4 * 2):]  # 4 turns gần nhất
            for msg in recent:
                role    = msg.get("role")
                content = msg.get("content")
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content})

        # User message với ảnh (multimodal content array)
        user_content: list = [
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            }
        ]
        # Nếu có text kèm theo ảnh, thêm vào
        if text_message and text_message.strip():
            user_content.insert(0, {
                "type": "text",
                "text": text_message.strip(),
            })
        else:
            user_content.insert(0, {
                "type": "text",
                "text": "Tôi chỉ gửi ảnh, chưa có câu hỏi cụ thể. Hãy mô tả/triage ảnh, đọc nội dung nhìn thấy nếu có, và hỏi tôi muốn hỗ trợ hướng nào tiếp theo.",
            })

        messages.append({"role": "user", "content": user_content})
        return messages

    def _is_prescription_in_reply(self, reply: str) -> bool:
        """Detect nếu AI đã xác định ảnh là đơn thuốc kê đơn → escalate."""
        prescription_signals = [
            "đơn thuốc kê đơn", "thuốc kê đơn", "chỉ định của bác sĩ",
            "gửi đơn thuốc", "dược sĩ của medispace", "đơn thuốc này",
            "kê đơn", "prescription"
        ]
        reply_lower = reply.lower()
        return any(sig in reply_lower for sig in prescription_signals)

    def _image_classification(self, reply: str, text_message: str = "") -> str:
        normalized = _normalize_text(f"{reply} {text_message}")

        non_medical_or_unclear = any(term in normalized for term in [
            "logo", "khong lien quan den y te", "khong phai la noi dung lien quan",
            "khong phai noi dung lien quan", "khong chua thong tin y te",
            "khong co thong tin y te", "khong phai la don thuoc",
        ])
        if non_medical_or_unclear and not any(term in _normalize_text(text_message) for term in ["don thuoc", "toa thuoc"]):
            return "image_only_triage"

        prescription_negative = re.search(
            r"\b(khong|khong phai|khong co|khong chua|khong thay|khong nhan thay)\b.{0,160}\b(don thuoc|toa thuoc|prescription|bac si|benh vien)\b",
            normalized,
        )
        prescription_positive = any(term in normalized for term in [
            "day la don thuoc", "co ve la don thuoc", "hinh anh la don thuoc",
            "anh la don thuoc", "don thuoc ke don", "toa thuoc",
            "thuoc duoc ke", "bac si ke", "prescription image",
        ])
        if prescription_positive and not prescription_negative:
            return "prescription_image_info"
        if any(term in normalized for term in ["vet thuong", "phat ban", "da lieu", "sung", "mun", "ngua"]):
            return "image_symptom_triage"
        if any(term in normalized for term in ["hop thuoc", "nhan thuoc", "san pham", "thanh phan", "hoat chat"]):
            return "product_image_info"
        return "image_only_triage"

    def _should_escalate_image_reply(self, reply: str) -> bool:
        normalized = _normalize_text(reply)
        urgent_terms = [
            "cap cuu", "di kham ngay", "gap bac si ngay", "kho tho", "sot cao",
            "soc phan ve", "sung mat", "sung moi", "chay mau", "tu tu", "nguy hiem"
        ]
        return any(term in normalized for term in urgent_terms)

    def _looks_truncated_image_reply(self, reply: str) -> bool:
        text = (reply or "").strip()
        if not text:
            return True
        tail = text[-160:].strip()
        if tail in {"*", "**", "-", "•"}:
            return True
        if tail.endswith(("*", "**", "-", ":")):
            return True
        if tail.count("**") % 2 == 1:
            return True
        last_line = tail.splitlines()[-1].strip()
        if re.fullmatch(r"\d+\.?", last_line):
            return True
        normalized_tail = _normalize_text(tail)
        dangling_phrases = [
            "minh chi doc", "luu y quan trong", "thong tin cac loai thuoc",
            "cong dung chung", "lieu dung", "theo anh"
        ]
        return any(normalized_tail.endswith(phrase) for phrase in dangling_phrases)

    def _image_suggested_questions(self, classification: str, is_escalated: bool) -> list[str]:
        if is_escalated:
            return ["Kết nối Dược sĩ Medispace", "Khi nào cần đi khám ngay?"]
        if classification == "prescription_image_info":
            return ["Đọc danh sách thuốc trong đơn", "Giải thích công dụng chung từng thuốc", "Kết nối Dược sĩ kiểm tra đơn"]
        if classification == "product_image_info":
            return ["Sản phẩm này dùng để làm gì?", "Có cần đơn thuốc không?", "Tìm sản phẩm này trên Medispace"]
        if classification == "image_symptom_triage":
            return ["Khi nào cần đi khám?", "Tôi nên theo dõi dấu hiệu nào?", "Kết nối Dược sĩ"]
        return ["Bạn muốn mình mô tả kỹ hơn không?", "Gửi lại ảnh rõ hơn", "Kết nối Dược sĩ Medispace"]

    async def _handle_image_request(
        self,
        image_url: str,
        text_message: str,
        user_id: str,
        conversation_id: str,
        history: list = None,
        context_data: dict = None,
        streaming: bool = False,
    ) -> dict:
        """
        Non-streaming image pipeline:
        Gọi LLM với multimodal messages, trả về dict chuẩn.
        """
        try:
            image_payload_url = await normalize_image_for_llm(image_url)
        except ImageFetchError as exc:
            logger.warning("[PharmacyAgent Vision] Image fetch failed: %s url=%s", exc, image_url[:120])
            return {
                "reply": str(exc),
                "classification": "image_only_triage",
                "is_escalated": False,
                "products_suggested": [],
                "suggested_questions": ["Gửi lại ảnh rõ hơn", "Kết nối Dược sĩ Medispace"],
            }

        messages = self._build_multimodal_messages(image_payload_url, text_message, history)

        endpoint = f"{LLM_BASE}/v1/chat/completions"
        payload  = {
            "model":       LLM_MODEL,
            "messages":    messages,
            "temperature": 0.30,
            "max_tokens":  LLM_IMAGE_MAX_TOKENS,
            "stream":      False,
        }
        logger.info("[PharmacyAgent Vision] Gọi LLM với ảnh: %s", image_url[:80])

        raw_reply = ""
        finish_reason = None
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(endpoint, json=payload)
                    resp.raise_for_status()
                    data      = resp.json()
                    choice = data["choices"][0]
                    finish_reason = choice.get("finish_reason")
                    raw_reply = choice["message"]["content"]
                    break
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt < LLM_MAX_RETRIES:
                    await asyncio.sleep(1.5 * (attempt + 1))
                else:
                    logger.error("[PharmacyAgent Vision] Timeout sau %d lần: %s", LLM_MAX_RETRIES + 1, e)
                    raise e
            except Exception as e:
                logger.error("[PharmacyAgent Vision] LLM Error: %s", e)
                raise e

        if finish_reason == "length":
            logger.warning(
                "[PharmacyAgent Vision] LLM stopped by max_tokens; raw_len=%d max_tokens=%d",
                len(raw_reply), LLM_IMAGE_MAX_TOKENS,
            )
            return {
                "reply": "Phan hoi phan tich anh bi cat do qua dai. Vui long thu lai hoac hoi theo tung phan cua don thuoc.",
                "classification": "image_only_triage",
                "is_escalated": False,
                "products_suggested": [],
                "suggested_questions": ["Doc danh sach thuoc trong don", "Giai thich cong dung tung thuoc", "Ket noi Duoc si Medispace"],
            }

        # Sanitize reply
        safe_reply, _ = sanitize_response(raw_reply)
        if self._looks_truncated_image_reply(safe_reply):
            logger.warning("[PharmacyAgent Vision] Incomplete-looking reply; raw_len=%d", len(raw_reply))
            return {
                "reply": "Phan hoi phan tich anh chua hoan tat. Vui long thu lai hoac hoi theo tung phan cua don thuoc.",
                "classification": "image_only_triage",
                "is_escalated": False,
                "products_suggested": [],
                "suggested_questions": ["Doc danh sach thuoc trong don", "Giai thich cong dung tung thuoc", "Ket noi Duoc si Medispace"],
            }

        # Đơn thuốc kê đơn → escalate sang dược sĩ thật
        classification = self._image_classification(safe_reply, text_message)
        is_escalated = self._should_escalate_image_reply(safe_reply)

        logger.info("[PharmacyAgent Vision] Done. escalated=%s", is_escalated)
        return {
            "reply":               safe_reply,
            "classification":      classification,
            "is_escalated":        is_escalated,
            "products_suggested":  [],
            "suggested_questions": self._image_suggested_questions(classification, is_escalated),
        }

    async def _stream_image_request(
        self,
        image_url: str,
        text_message: str,
        user_id: str,
        conversation_id: str,
        history: list = None,
        context_data: dict = None,
    ):
        """
        Streaming image pipeline: yield chunk + done.
        """
        try:
            image_payload_url = await normalize_image_for_llm(image_url)
        except ImageFetchError as exc:
            logger.warning("[PharmacyAgent Vision Stream] Image fetch failed: %s url=%s", exc, image_url[:120])
            yield json.dumps({
                "type": "error",
                "message": str(exc),
            }, ensure_ascii=False) + "\n"
            return

        messages = self._build_multimodal_messages(image_payload_url, text_message, history)

        endpoint      = f"{LLM_BASE}/v1/chat/completions"
        payload       = {
            "model":       LLM_MODEL,
            "messages":    messages,
            "temperature": 0.30,
            "max_tokens":  LLM_IMAGE_MAX_TOKENS,
            "stream":      True,
        }
        logger.info("[PharmacyAgent Vision Stream] Gọi LLM stream với ảnh: %s", image_url[:80])

        full_raw_reply = ""
        finish_reason = None
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", endpoint, json=payload) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_lines():
                        if not chunk.strip():
                            continue
                        if chunk.startswith("data: "):
                            data_str = chunk[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                choices   = data_json.get("choices", [])
                                if choices:
                                    finish_reason = choices[0].get("finish_reason") or finish_reason
                                    delta   = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        full_raw_reply += content
                                        yield json.dumps({
                                            "type":    "chunk",
                                            "content": content,
                                        }, ensure_ascii=False) + "\n"
                            except Exception:
                                pass
        except Exception as e:
            logger.error("[PharmacyAgent Vision Stream] Error: %s", e)
            yield json.dumps({
                "type":    "error",
                "message": f"Lỗi hệ thống khi phân tích ảnh: {str(e)}",
            }, ensure_ascii=False) + "\n"
            return

        if finish_reason == "length":
            logger.warning(
                "[PharmacyAgent Vision Stream] LLM stopped by max_tokens; raw_len=%d max_tokens=%d",
                len(full_raw_reply), LLM_IMAGE_MAX_TOKENS,
            )
            yield json.dumps({
                "type": "error",
                "message": "Phan hoi phan tich anh bi cat do qua dai. Vui long thu lai hoac hoi theo tung phan cua don thuoc.",
            }, ensure_ascii=False) + "\n"
            return

        # Post-process
        safe_reply, _ = sanitize_response(full_raw_reply)
        if self._looks_truncated_image_reply(safe_reply):
            logger.warning("[PharmacyAgent Vision Stream] Incomplete-looking reply; raw_len=%d", len(full_raw_reply))
            yield json.dumps({
                "type": "error",
                "message": "Phan hoi phan tich anh chua hoan tat. Vui long thu lai hoac hoi theo tung phan cua don thuoc.",
            }, ensure_ascii=False) + "\n"
            return
        classification = self._image_classification(safe_reply, text_message)
        is_escalated  = self._should_escalate_image_reply(safe_reply)

        suggested = self._image_suggested_questions(classification, is_escalated)
        yield json.dumps({
            "type":                "done",
            "reply":               safe_reply,
            "classification":      classification,
            "is_escalated":        is_escalated,
            "products_suggested":  [],
            "suggested_questions": suggested,
        }, ensure_ascii=False) + "\n"

    # ── PUBLIC METHODS ────────────────────────────────────────────────────────

    async def respond(
        self,
        message: str,
        user_id: str,
        conversation_id: str,
        history: list = None,
        context_products: list = None,
        context_data: dict = None,       # Phase 3: real user data từ BE
        image_url: str = None,           # Vision: ảnh gửi kèm
    ) -> dict:
        # 1. Pre-filter — phân loại intent
        classification = classify_message(message)
        logger.info("[PharmacyAgent] Intent: %s | image=%s", classification, bool(image_url))

        # ── IMAGE PIPELINE ────────────────────────────────────────────────
        # Nếu có ảnh → dùng pipeline ảnh riêng (multimodal)
        if image_url:
            return await self._handle_image_request(
                image_url=image_url,
                text_message=message,
                user_id=user_id,
                conversation_id=conversation_id,
                history=history,
                context_data=context_data,
                streaming=False,
            )

        if classification == "image_only":
            logger.warning("[PharmacyAgent] image_only intent but image_url is missing")
            return {
                "reply": "Mình chưa nhận được ảnh trong yêu cầu này. Bạn vui lòng gửi lại ảnh rõ nét để mình có thể đọc và phân tích nhé.",
                "classification": "image_only_triage",
                "is_escalated": False,
                "products_suggested": [],
                "suggested_questions": ["Gửi lại ảnh đơn thuốc", "Kết nối Dược sĩ Medispace"],
            }

        # Các intent được xử lý bằng prefilter cứng (không gọi LLM)
        hard_prefilter_map = {
            'too_long':             (TOO_LONG_RESPONSE,       False),
            'greeting':             (GREETING_RESPONSE,      False),
            'emergency':            (EMERGENCY_RESPONSE,       True),
            'mental_health_crisis': (MENTAL_HEALTH_RESPONSE,  True),
            'prescription_request': (PRESCRIPTION_RESPONSE,   True),
            'personalized_dosage':  (PERSONALIZED_DOSAGE_RESPONSE, True),
        }
        if classification in hard_prefilter_map:
            reply, escalated = hard_prefilter_map[classification]
            return self._build_prefilter_response(classification, reply, escalated)

        # 2+3. [PARALLEL RAG] Gọi song song: products + articles cùng lúc
        # Trước: tuần tự ~2-4s (products xong mới gọi articles)
        # Sau:  song song ~1-2s (cả hai chạy đồng thời, đợi cái chậm nhất)
        needs_articles = classification in (
            "general", "drug_info_general", "product_search", "return_request", "coupon_inquiry"
        )

        async def _no_articles() -> list:
            return []

        (
            (enriched_products, rag_source),
            rag_articles,
        ) = await asyncio.gather(
            self._enrich_context_products(
                message=message,
                intent=classification,
                fe_products=context_products or [],
            ),
            search_articles_for_rag(message) if needs_articles else _no_articles(),
        )
        logger.info(
            "[PharmacyAgent] Parallel RAG done — products=%d articles=%d",
            len(enriched_products), len(rag_articles),
        )

        rag_context = self._build_rag_context(enriched_products, articles=rag_articles)
        messages    = self._build_messages(
            message, history or [], rag_context, classification,
            context_data=context_data,
        )

        # 4. Call LLM — [FIX-4] với retry exponential backoff
        temperature = self._get_llm_temperature(classification)
        endpoint = f"{LLM_BASE}/v1/chat/completions"
        payload  = {
            "model":       LLM_MODEL,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  LLM_MAX_TOKENS,
            "stream":      False,
        }
        logger.info(
            "[PharmacyAgent] LLM call → intent=%s temp=%.2f rag=%s products=%d context_data=%s",
            classification, temperature, rag_source, len(enriched_products),
            bool(context_data),
        )
        raw_reply = ""
        last_error: Exception | None = None
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(endpoint, json=payload)
                    resp.raise_for_status()
                    data      = resp.json()
                    raw_reply = data["choices"][0]["message"]["content"]
                    logger.debug("[PharmacyAgent] Raw reply: %s", raw_reply[:200])
                    break  # Thành công → thoát retry loop
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if attempt < LLM_MAX_RETRIES:
                    wait_s = 1.5 * (attempt + 1)  # 1.5s, 3.0s
                    logger.warning(
                        "[PharmacyAgent] LLM timeout (attempt %d/%d), retry sau %.1fs...",
                        attempt + 1, LLM_MAX_RETRIES + 1, wait_s
                    )
                    await asyncio.sleep(wait_s)
                else:
                    logger.error("[PharmacyAgent] LLM timeout sau %d lần thử: %s", LLM_MAX_RETRIES + 1, str(e))
                    raise e
            except Exception as e:
                # Lỗi khác (HTTP 4xx/5xx) → raise ngay, không retry
                logger.error("[PharmacyAgent] LLM API Error: %s", str(e))
                raise e

        # 5. Post-process
        result = self._process_final_reply(raw_reply, enriched_products)

        # [FIX-1] is_escalated dựa theo intent, KHÔNG phải was_sanitized
        # was_sanitized=True chỉ nghĩa là reply bị thay fallback — không nhất thiết cần DS
        # Các intent luôn escalate: emergency, mental_health_crisis, prescription_request
        ESCALATE_INTENTS = {"emergency", "mental_health_crisis", "prescription_request"}
        is_escalated = classification in ESCALATE_INTENTS

        return {
            "reply":               result["safe_reply"],
            "classification":      classification,
            "is_escalated":        is_escalated,
            "products_suggested":  result["products_suggested"],
            "suggested_questions": result["suggested_questions"],
        }

    async def stream_respond(
        self,
        message: str,
        user_id: str,
        conversation_id: str,
        history: list = None,
        context_products: list = None,
        context_data: dict = None,       # Phase 3: real user data từ BE
        image_url: str = None,           # Vision: ảnh gửi kèm
    ):
        # 1. Pre-filter
        classification = classify_message(message)
        logger.info("[PharmacyAgent Stream] Intent: %s | image=%s", classification, bool(image_url))

        # ── IMAGE PIPELINE (stream mode) ────────────────────────────────
        if image_url:
            # Stream: yield chunk + done từ image pipeline
            async for chunk in self._stream_image_request(
                image_url=image_url,
                text_message=message,
                user_id=user_id,
                conversation_id=conversation_id,
                history=history,
                context_data=context_data,
            ):
                yield chunk
            return

        if classification == "image_only":
            logger.warning("[PharmacyAgent Stream] image_only intent but image_url is missing")
            yield json.dumps({
                "type": "done",
                "reply": "Mình chưa nhận được ảnh trong yêu cầu này. Bạn vui lòng gửi lại ảnh rõ nét để mình có thể đọc và phân tích nhé.",
                "classification": "image_only_triage",
                "is_escalated": False,
                "products_suggested": [],
                "suggested_questions": ["Gửi lại ảnh đơn thuốc", "Kết nối Dược sĩ Medispace"],
            }, ensure_ascii=False) + "\n"
            return

        hard_prefilter_map = {
            'too_long':             (TOO_LONG_RESPONSE,       False),
            'greeting':             (GREETING_RESPONSE,      False),
            'emergency':            (EMERGENCY_RESPONSE,       True),
            'mental_health_crisis': (MENTAL_HEALTH_RESPONSE,  True),
            'prescription_request': (PRESCRIPTION_RESPONSE,   True),
            'personalized_dosage':  (PERSONALIZED_DOSAGE_RESPONSE, True),
        }
        if classification in hard_prefilter_map:
            reply, escalated = hard_prefilter_map[classification]
            yield json.dumps({
                "type": "done",
                **self._build_prefilter_response(classification, reply, escalated),
            }, ensure_ascii=False) + "\n"
            return

        # 2+3. [PARALLEL RAG] Gọi song song: products + articles cùng lúc
        needs_articles = classification in (
            "general", "drug_info_general", "product_search", "return_request", "coupon_inquiry"
        )

        async def _no_articles() -> list:
            return []

        (
            (enriched_products, rag_source),
            rag_articles,
        ) = await asyncio.gather(
            self._enrich_context_products(
                message=message,
                intent=classification,
                fe_products=context_products or [],
            ),
            search_articles_for_rag(message) if needs_articles else _no_articles(),
        )
        logger.info(
            "[PharmacyAgent Stream] Parallel RAG done — products=%d articles=%d",
            len(enriched_products), len(rag_articles),
        )

        rag_context = self._build_rag_context(enriched_products, articles=rag_articles)
        messages    = self._build_messages(
            message, history or [], rag_context, classification,
            context_data=context_data,
        )

        # 4. Call LLM (streaming)
        temperature   = self._get_llm_temperature(classification)
        full_raw_reply = ""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                endpoint = f"{LLM_BASE}/v1/chat/completions"
                payload  = {
                    "model":       LLM_MODEL,
                    "messages":    messages,
                    "temperature": temperature,
                    "max_tokens":  LLM_MAX_TOKENS,
                    "stream":      True,
                }
                logger.info(
                    "[PharmacyAgent Stream] LLM call → intent=%s temp=%.2f rag=%s products=%d context_data=%s",
                    classification, temperature, rag_source, len(enriched_products),
                    bool(context_data),
                )

                async with client.stream("POST", endpoint, json=payload) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_lines():
                        if not chunk.strip():
                            continue
                        if chunk.startswith("data: "):
                            data_str = chunk[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                choices   = data_json.get("choices", [])
                                if choices:
                                    delta   = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        full_raw_reply += content
                                        yield json.dumps({
                                            "type":    "chunk",
                                            "content": content,
                                        }, ensure_ascii=False) + "\n"
                            except Exception as e:
                                logger.error("[PharmacyAgent Stream] JSON decode error: %s", str(e))

        except Exception as e:
            logger.error("[PharmacyAgent Stream] LLM API Error: %s", str(e))
            yield json.dumps({
                "type":    "error",
                "message": f"Lỗi hệ thống AI: {str(e)}",
            }, ensure_ascii=False) + "\n"
            return

        # 5. Post-process full reply
        result = self._process_final_reply(full_raw_reply, enriched_products)
        logger.info("[PharmacyAgent Stream] Sanitized: %s", result["was_sanitized"])

        # 6. Yield final metadata
        # [FIX-1] is_escalated dựa theo intent (xem respond() để đồng nhất)
        ESCALATE_INTENTS = {"emergency", "mental_health_crisis", "prescription_request"}
        is_escalated_stream = classification in ESCALATE_INTENTS
        yield json.dumps({
            "type":               "done",
            "reply":              result["safe_reply"],
            "classification":     classification,
            "is_escalated":       is_escalated_stream,
            "products_suggested": result["products_suggested"],
            "suggested_questions": result["suggested_questions"],
        }, ensure_ascii=False) + "\n"

