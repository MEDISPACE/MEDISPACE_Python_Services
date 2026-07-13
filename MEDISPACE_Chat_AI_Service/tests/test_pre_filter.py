"""
tests/test_pre_filter.py
Unit tests for classify_message() — Intent classification (Phase 1).
Chạy: pytest tests/ -v
"""
import pytest
from src.guardrails.pre_filter import classify_message


# ════════════════════════════════════════════════════════════════
# Safety intents (giữ nguyên từ cũ, không được regression)
# ════════════════════════════════════════════════════════════════

class TestSafetyIntents:
    def test_too_long(self):
        assert classify_message("A" * 801) == "too_long"

    def test_too_long_boundary(self):
        # Đúng 800 ký tự → general (không block)
        assert classify_message("A" * 800) != "too_long"

    def test_emergency_chest_pain(self):
        assert classify_message("Tôi đang bị đau ngực dữ dội") == "emergency"

    def test_emergency_stroke(self):
        assert classify_message("Người nhà tôi bị đột quỵ") == "emergency"

    def test_emergency_unconscious(self):
        assert classify_message("Bệnh nhân mất ý thức, cần cấp cứu") == "emergency"

    def test_mental_health_crisis(self):
        assert classify_message("Tôi muốn chết, không muốn sống nữa") == "mental_health_crisis"

    def test_mental_health_self_harm(self):
        assert classify_message("Tôi đang tự làm hại bản thân") == "mental_health_crisis"

    def test_prescription_buy_antibiotic(self):
        assert classify_message("Tôi cần mua amoxicillin") == "prescription_request"

    def test_prescription_direct_request(self):
        assert classify_message("Bạn kê đơn cho tôi đi") == "prescription_request"

    def test_prescription_buy_augmentin(self):
        assert classify_message("Cho tôi mua Augmentin") == "prescription_request"


# ════════════════════════════════════════════════════════════════
# Intent mới: Order Tracking
# ════════════════════════════════════════════════════════════════

class TestOrderTracking:
    def test_order_where(self):
        assert classify_message("Đơn hàng của tôi đến đâu rồi?") == "order_tracking"

    def test_order_code(self):
        assert classify_message("ORD-1234567890 trạng thái thế nào?") == "order_tracking"

    def test_order_delivery_time(self):
        assert classify_message("Giao hàng mất bao lâu vậy?") == "order_tracking"

    def test_order_check(self):
        assert classify_message("Tôi muốn kiểm tra đơn hàng vừa đặt") == "order_tracking"

    def test_order_tracking_code(self):
        assert classify_message("Cho tôi mã tracking đơn hàng") == "order_tracking"

    def test_order_not_arrived(self):
        assert classify_message("Đặt hàng 3 ngày rồi chưa thấy giao") == "order_tracking"


# ════════════════════════════════════════════════════════════════
# Intent mới: Loyalty
# ════════════════════════════════════════════════════════════════

class TestLoyaltyInquiry:
    def test_points_balance(self):
        assert classify_message("Tôi còn bao nhiêu điểm thưởng?") == "loyalty_inquiry"

    def test_tier_upgrade(self):
        assert classify_message("Lên hạng Vàng cần bao nhiêu tiền?") == "loyalty_inquiry"

    def test_redeem_points(self):
        assert classify_message("Đổi điểm thưởng thế nào?") == "loyalty_inquiry"

    def test_points_expiry(self):
        assert classify_message("Điểm tích lũy của tôi sắp hết hạn chưa?") == "loyalty_inquiry"

    def test_tier_inquiry(self):
        assert classify_message("Hạng thành viên của tôi là gì?") == "loyalty_inquiry"


# ════════════════════════════════════════════════════════════════
# Intent mới: Coupon
# ════════════════════════════════════════════════════════════════

class TestCouponInquiry:
    def test_discount_code(self):
        assert classify_message("Có mã giảm giá nào không?") == "coupon_inquiry"

    def test_voucher(self):
        assert classify_message("Tôi có voucher SAVE10 dùng được không?") == "coupon_inquiry"

    def test_freeship(self):
        assert classify_message("Có miễn phí vận chuyển không ạ?") == "coupon_inquiry"

    def test_coupon_direct(self):
        assert classify_message("Nhập coupon không được, lý do gì?") == "coupon_inquiry"

    def test_flash_sale(self):
        assert classify_message("Flash sale hôm nay có gì?") == "coupon_inquiry"


# ════════════════════════════════════════════════════════════════
# Intent mới: Return Request
# ════════════════════════════════════════════════════════════════

class TestReturnRequest:
    def test_return_product(self):
        assert classify_message("Tôi muốn trả hàng") == "return_request"

    def test_broken_product(self):
        assert classify_message("Sản phẩm tôi nhận được bị vỡ, muốn đổi") == "return_request"

    def test_refund(self):
        assert classify_message("Tôi muốn hoàn tiền cho đơn hàng này") == "return_request"

    def test_recent_returned_orders(self):
        assert classify_message("Tôi muốn xem đơn hàng nào đã hoàn trả gần đây") == "return_request"

    def test_return_code(self):
        assert classify_message("Yêu cầu RET-123456 của tôi xử lý chưa?") == "return_request"

    def test_wrong_item(self):
        assert classify_message("Giao sai hàng rồi, cần đổi lại") == "return_request"

    def test_return_policy(self):
        assert classify_message("Chính sách trả hàng của Medispace là gì?") == "return_request"


# ════════════════════════════════════════════════════════════════
# Intent mới: Prescription Status
# ════════════════════════════════════════════════════════════════

class TestPrescriptionStatus:
    def test_prescription_approved(self):
        assert classify_message("Đơn thuốc tôi gửi lên duyệt chưa?") == "prescription_status"

    def test_prescription_code(self):
        assert classify_message("PRE-987654 trạng thái thế nào?") == "prescription_status"

    def test_prescription_check(self):
        assert classify_message("Kiểm tra đơn thuốc tôi đã gửi") == "prescription_status"


# ════════════════════════════════════════════════════════════════
# Intent mới: Product Search
# ════════════════════════════════════════════════════════════════

class TestProductSearch:
    def test_find_medicine(self):
        assert classify_message("Tìm thuốc ho cho trẻ em") == "product_search"

    def test_price_inquiry(self):
        assert classify_message("Vitamin C giá bao nhiêu?") == "product_search"

    def test_product_available(self):
        assert classify_message("Medispace có bán thuốc tiêu hóa không?") == "product_search"

    def test_compare_products(self):
        assert classify_message("So sánh Vitamin C Blackmores và Centrum") == "product_search"


# ════════════════════════════════════════════════════════════════
# General — Không bị nhầm intent
# ════════════════════════════════════════════════════════════════

class TestGeneralNotMisclassified:
    def test_otc_info(self):
        assert classify_message("Paracetamol có tác dụng gì?") == "drug_info_general"

    def test_symptom_query(self):
        assert classify_message("Tôi bị sổ mũi, đau đầu nhẹ thì dùng gì?") == "general"

    def test_rx_info_not_buy(self):
        # Hỏi thông tin Rx (không mua) → drug_info_general, để LLM xử lý qua system prompt
        assert classify_message("Amoxicillin điều trị bệnh gì?") == "drug_info_general"

    def test_supplement_info(self):
        assert classify_message("Vitamin D3 bổ sung có tốt không?") == "general"

    def test_health_question(self):
        assert classify_message("Uống nhiều vitamin C có hại không?") == "general"
