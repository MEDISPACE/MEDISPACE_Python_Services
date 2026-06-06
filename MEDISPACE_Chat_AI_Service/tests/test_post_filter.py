"""
tests/test_post_filter.py
Unit tests for sanitize_response() — Post-filter guardrail (Phase 1).
Chạy: pytest tests/ -v
"""
import pytest
from src.guardrails.post_filter import sanitize_response


# ════════════════════════════════════════════════════════════════
# Câu HỢP LỆ — KHÔNG được block (false positive đã giảm)
# ════════════════════════════════════════════════════════════════

class TestNotBlocked:
    """Các câu trả lời hợp lệ phải đi qua không bị block."""

    def test_general_product_description(self):
        text = "Paracetamol 500mg là thuốc hạ sốt, giảm đau phổ biến."
        _, blocked = sanitize_response(text)
        assert not blocked

    def test_medispace_sells_info(self):
        text = "Medispace bán nhiều sản phẩm chăm sóc sức khỏe chất lượng cao."
        _, blocked = sanitize_response(text)
        assert not blocked

    def test_read_manual_advice(self):
        text = "Bạn đọc kỹ tờ hướng dẫn sử dụng để biết liều lượng phù hợp nhé."
        _, blocked = sanitize_response(text)
        assert not blocked

    def test_vitamin_general_info(self):
        text = "Vitamin C 1000mg có thể hỗ trợ tăng cường miễn dịch khi cơ thể cần."
        _, blocked = sanitize_response(text)
        assert not blocked

    def test_product_recommendation(self):
        text = "Medispace có bán Kẹo ngậm Bảo Thanh giúp giảm rát họng hiệu quả."
        _, blocked = sanitize_response(text)
        assert not blocked

    def test_pharmacist_consult_advice(self):
        text = "Bạn nên tham khảo Dược sĩ Medispace để được tư vấn đúng nhất."
        _, blocked = sanitize_response(text)
        assert not blocked

    def test_otc_side_effects_general(self):
        text = "Ibuprofen có thể gây kích ứng dạ dày nếu uống lúc đói, nên dùng sau ăn."
        _, blocked = sanitize_response(text)
        assert not blocked

    def test_supplement_description(self):
        text = "Omega-3 hỗ trợ sức khỏe tim mạch và não bộ. Thường dùng 1-2 viên/ngày theo hướng dẫn trên nhãn."
        _, blocked = sanitize_response(text)
        # "1-2 viên/ngày theo hướng dẫn" là thông tin chung từ nhãn, không phải AI kê đơn
        assert not blocked


# ════════════════════════════════════════════════════════════════
# Câu NGUY HIỂM — PHẢI bị block
# ════════════════════════════════════════════════════════════════

class TestBlocked:
    """Các câu AI kê liều dùng cá nhân hóa phải bị block."""

    def test_dosage_mg_per_day(self):
        text = "Bạn nên uống 500mg/ngày để điều trị tình trạng này."
        _, blocked = sanitize_response(text)
        assert blocked

    def test_dosage_tablets_per_day(self):
        text = "Hãy uống 2 viên mỗi ngày để nhanh khỏi bệnh."
        _, blocked = sanitize_response(text)
        assert blocked

    def test_ai_prescribes_antibiotic(self):
        text = "Tôi kê cho bạn Amoxicillin để điều trị nhiễm khuẩn này."
        _, blocked = sanitize_response(text)
        assert blocked

    def test_dosage_ml_per_dose(self):
        text = "Uống 10ml/lần, ngày 3 lần để giảm triệu chứng."
        _, blocked = sanitize_response(text)
        assert blocked


# ════════════════════════════════════════════════════════════════
# Brand name correction
# ════════════════════════════════════════════════════════════════

class TestBrandCorrection:
    def test_medis_period(self):
        text = "Liên hệ Medis. để được hỗ trợ."
        cleaned, _ = sanitize_response(text)
        assert "Medispace" in cleaned
        assert "Medis." not in cleaned

    def test_medis_comma(self):
        text = "Dược sĩ của Medis, sẽ tư vấn bạn."
        cleaned, _ = sanitize_response(text)
        assert "Medispace" in cleaned

    def test_medis_space(self):
        text = "Medispace hỗ trợ bạn. Liên hệ Medis nhé."
        cleaned, _ = sanitize_response(text)
        assert cleaned.count("Medispace") >= 1

    def test_full_medispace_unchanged(self):
        # "Medispace" đầy đủ không bị đổi
        text = "Liên hệ Medispace để được hỗ trợ."
        cleaned, _ = sanitize_response(text)
        assert "Medispace" in cleaned


# ════════════════════════════════════════════════════════════════
# Markdown cleanup
# ════════════════════════════════════════════════════════════════

class TestMarkdownCleanup:
    def test_removes_code_block(self):
        text = "```Paracetamol là thuốc hạ sốt```"
        cleaned, _ = sanitize_response(text)
        assert "```" not in cleaned
        assert "Paracetamol" in cleaned

    def test_removes_bold(self):
        text = "**Paracetamol** là thuốc hạ sốt."
        cleaned, _ = sanitize_response(text)
        assert "**" not in cleaned
        assert "Paracetamol" in cleaned

    def test_removes_italic(self):
        text = "Bạn nên dùng *Vitamin C* mỗi ngày."
        cleaned, _ = sanitize_response(text)
        assert "*Vitamin C*" not in cleaned
        assert "Vitamin C" in cleaned

    def test_removes_code_block_with_lang(self):
        text = "```json\n{\"key\": \"value\"}\n```"
        cleaned, _ = sanitize_response(text)
        assert "```" not in cleaned

    def test_normalizes_extra_newlines(self):
        text = "Dòng 1\n\n\n\n\nDòng 2"
        cleaned, _ = sanitize_response(text)
        assert "\n\n\n" not in cleaned


# ════════════════════════════════════════════════════════════════
# Return value structure
# ════════════════════════════════════════════════════════════════

class TestReturnStructure:
    def test_returns_tuple(self):
        result = sanitize_response("Paracetamol là thuốc hạ sốt.")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_string_and_bool(self):
        cleaned, blocked = sanitize_response("Bình thường.")
        assert isinstance(cleaned, str)
        assert isinstance(blocked, bool)

    def test_not_blocked_returns_false(self):
        _, blocked = sanitize_response("Thông tin thuốc chung.")
        assert blocked is False

    def test_blocked_returns_true(self):
        _, blocked = sanitize_response("Uống 500mg/ngày nhé.")
        assert blocked is True
