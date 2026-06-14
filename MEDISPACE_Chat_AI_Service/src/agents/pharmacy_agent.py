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
import httpx
import asyncio
import logging
import re
from datetime import datetime
from src.guardrails.pre_filter import (
    classify_message,
    EMERGENCY_RESPONSE,
    PRESCRIPTION_RESPONSE,
    MENTAL_HEALTH_RESPONSE,
    TOO_LONG_RESPONSE,
)
from src.guardrails.post_filter import sanitize_response
from src.rag.typesense_client import search_products_for_rag, search_articles_for_rag

logger = logging.getLogger("chat_ai.agent")

LLM_BASE  = os.getenv("CUSTOM_LLM_BASE_URL", "https://llm.datateam.space").rstrip("/")
LLM_MODEL = os.getenv("CUSTOM_LLM_MODEL", "gemma-4-e4b-it.gguf")
LLM_MAX_TOKENS = int(os.getenv("CUSTOM_LLM_MAX_TOKENS", "1536"))
LLM_MAX_RETRIES = int(os.getenv("CUSTOM_LLM_MAX_RETRIES", "2"))  # [FIX-4]

# ── Context window limits ─────────────────────────────────────────────────────
# [FIX-7] Đồng bộ với BE: BE giới hạn limit(6) messages khi gửi history
MAX_HISTORY_TURNS = 6      # Tối đa 6 lượt hội thoại (user+assistant mỗi lượt)
MAX_HISTORY_CHARS = 3000    # Tối đa 3000 ký tự trong history

# ── RAG config ────────────────────────────────────────────────────────
# Nếu FE gửi ít hơn ngưỡng này, tự động query Typesense bổ sung
RAG_MIN_PRODUCTS = 2        # Dưới 2 sản phẩm → trigger Typesense auto-fetch
RAG_MAX_PRODUCTS = 6        # Tối đa sản phẩm đưa vào RAG context

# ── Intent → Temperature mapping ─────────────────────────────────────────────
INTENT_TEMPERATURE: dict[str, float] = {
    # An toàn y tế → ít sáng tạo, ít hallucination
    "general":              0.30,
    "product_search":       0.35,
    # Thông tin tra cứu → trả lời chính xác
    "order_tracking":       0.20,
    "prescription_status":  0.20,
    "loyalty_inquiry":      0.20,
    "coupon_inquiry":       0.25,
    "return_request":       0.20,
}


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared header (dùng trong mọi prompt) ─────────────────────────────────────
# [FIX-6] Inject ngày hiện tại để AI trả lời đúng về thời gian, hạn coupon...
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
INTENT_PROMPT_MAP: dict[str, str] = {
    "general":              GENERAL_PROMPT_TEMPLATE,
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
        self.timeout = httpx.Timeout(60.0, connect=10.0)

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
        Format real user data (orders, loyalty, history) thành chuỗi
        để inject vào system prompt trước khi gọi LLM.
        Chỉ inject khi có data thực tế và intent phù hợp.
        """
        if not context_data:
            return ""

        lines = []

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
                    status_vn = status_map.get(o.get("status", ""), o.get("status", "Không rõ"))
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
                    if o.get("trackingCode"):
                        line += f" | Mã vận đơn: {o.get('trackingCode')}"
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
                lines.append(
                    f"- Điểm hiện có: {loyalty.get('points', 0):,} điểm "
                    f"(= {loyalty.get('points', 0) * 100:,}đ) "
                    f"| Hạng: {tier_vn} "
                    f"| Tổng chi tiêu: {loyalty.get('totalSpent', 0):,}đ"
                )

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

        messages = [{"role": "system", "content": system_prompt}]

        # ── [FIX A1] Context window management ──────────────────────────────
        if history:
            # Chỉ lấy N lượt cuối (mỗi lượt = 1 user + 1 assistant)
            recent = history[-(MAX_HISTORY_TURNS * 2):]

            # Nếu tổng ký tự vẫn vượt giới hạn → cắt từ đầu
            total_chars = sum(len(m.get("content", "")) for m in recent)
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
        Phase 2 — RAG enrichment logic:
        Öu tiên sử dụng context_products từ FE nếu đủ (>= RAG_MIN_PRODUCTS).
        Nếu thiếu → query Typesense để bổ sung, merge deduped by mongoId.

        Returns:
            (merged_products, rag_source)
            rag_source: 'fe_only' | 'typesense_only' | 'merged' | 'none'
        """
        # Intent không cần RAG sản phẩm
        if intent in ("order_tracking", "loyalty_inquiry", "coupon_inquiry",
                      "return_request", "prescription_status"):
            return [], "none"

        fe_count = len(fe_products)

        # FE có đủ sản phẩm → dùng luôn (không tốn thêm network)
        if fe_count >= RAG_MIN_PRODUCTS:
            logger.info("[RAG] Dùng %d sản phẩm từ FE", fe_count)
            return fe_products[:RAG_MAX_PRODUCTS], "fe_only"

        # Thiếu → auto-fetch từ Typesense
        ts_products = await search_products_for_rag(
            message=message,
            intent=intent,
            limit=RAG_MAX_PRODUCTS,
        )

        if not ts_products and not fe_products:
            return [], "none"

        # Merge: FE trước, Typesense sau, dedup theo mongoId
        seen     = {p.get("mongoId") for p in fe_products if p.get("mongoId")}
        merged   = list(fe_products)  # giữ FE làm anchor
        for p in ts_products:
            mid = p.get("mongoId")
            if mid and mid not in seen:
                seen.add(mid)
                merged.append(p)
            if len(merged) >= RAG_MAX_PRODUCTS:
                break

        source = "merged" if fe_products else "typesense_only"
        logger.info(
            "[RAG] %s: FE=%d + TS=%d → tổng %d sản phẩm",
            source, fe_count, len(ts_products), len(merged),
        )
        return merged, source

    # ── PUBLIC METHODS ────────────────────────────────────────────────────────

    async def respond(
        self,
        message: str,
        user_id: str,
        conversation_id: str,
        history: list = None,
        context_products: list = None,
        context_data: dict = None,       # Phase 3: real user data từ BE
    ) -> dict:
        # 1. Pre-filter — phân loại intent
        classification = classify_message(message)
        logger.info("[PharmacyAgent] Intent: %s", classification)

        # Các intent được xử lý bằng prefilter cứng (không gọi LLM)
        hard_prefilter_map = {
            'too_long':             (TOO_LONG_RESPONSE,       False),
            'emergency':            (EMERGENCY_RESPONSE,       True),
            'mental_health_crisis': (MENTAL_HEALTH_RESPONSE,  True),
            'prescription_request': (PRESCRIPTION_RESPONSE,   True),
        }
        if classification in hard_prefilter_map:
            reply, escalated = hard_prefilter_map[classification]
            return self._build_prefilter_response(classification, reply, escalated)

        # 2. [Phase 2] RAG enrichment — bổ sung sản phẩm từ Typesense nếu cần
        enriched_products, rag_source = await self._enrich_context_products(
            message=message,
            intent=classification,
            fe_products=context_products or [],
        )

        # 3. [Phase 2+] RAG enrichment — products + articles
        # [FIX-8] Mở rộng thêm return_request và coupon_inquiry để AI có thêm
        # kiến thức sức khỏe khi tư vấn đổi trả dị ứng / khuyến mãi dinh dưỡng
        rag_articles = []
        if classification in ("general", "product_search", "return_request", "coupon_inquiry"):
            rag_articles = await search_articles_for_rag(message, limit=2)

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
    ):
        # 1. Pre-filter
        classification = classify_message(message)
        logger.info("[PharmacyAgent Stream] Intent: %s", classification)

        hard_prefilter_map = {
            'too_long':             (TOO_LONG_RESPONSE,       False),
            'emergency':            (EMERGENCY_RESPONSE,       True),
            'mental_health_crisis': (MENTAL_HEALTH_RESPONSE,  True),
            'prescription_request': (PRESCRIPTION_RESPONSE,   True),
        }
        if classification in hard_prefilter_map:
            reply, escalated = hard_prefilter_map[classification]
            yield json.dumps({
                "type": "done",
                **self._build_prefilter_response(classification, reply, escalated),
            }, ensure_ascii=False) + "\n"
            return

        # 2. [Phase 2] RAG enrichment — bổ sung sản phẩm từ Typesense nếu cần
        enriched_products, rag_source = await self._enrich_context_products(
            message=message,
            intent=classification,
            fe_products=context_products or [],
        )


        # 3. [Phase 2+] RAG enrichment — products + articles
        # [FIX-8] Mở rộng thêm return_request và coupon_inquiry
        rag_articles = []
        if classification in ("general", "product_search", "return_request", "coupon_inquiry"):
            rag_articles = await search_articles_for_rag(message, limit=2)

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
