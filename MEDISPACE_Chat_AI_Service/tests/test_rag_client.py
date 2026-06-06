"""
tests/test_rag_client.py
Unit tests for Typesense RAG client — Phase 2.
Dùng mock để không cần Typesense server thật.
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.rag.typesense_client import (
    _extract_search_query,
    search_products_for_rag,
    INTENT_RAG_CONFIG,
)


# ════════════════════════════════════════════════════════════════
# _extract_search_query — Query extraction logic
# ════════════════════════════════════════════════════════════════

class TestExtractSearchQuery:
    def test_short_message(self):
        result = _extract_search_query("Paracetamol", "general")
        assert "paracetamol" in result.lower() or "Paracetamol" in result

    def test_removes_stop_words(self):
        result = _extract_search_query("Tôi bị đau đầu, có thuốc gì không?", "general")
        assert "tôi" not in result.lower()
        # Nên giữ lại từ có nghĩa
        assert len(result) > 0

    def test_long_message_truncated(self):
        long_msg = "A" * 200
        result = _extract_search_query(long_msg, "general")
        assert len(result) <= 150

    def test_empty_message(self):
        result = _extract_search_query("", "general")
        assert result == ""

    def test_meaningful_words_kept(self):
        result = _extract_search_query("Vitamin C giảm cảm cúm", "general")
        # "Vitamin C" hoặc "giảm cảm cúm" nên còn trong result
        assert len(result) > 0

    def test_whitespace_trimmed(self):
        result = _extract_search_query("  Paracetamol  ", "general")
        assert result == result.strip()


# ════════════════════════════════════════════════════════════════
# INTENT_RAG_CONFIG — Configuration mapping
# ════════════════════════════════════════════════════════════════

class TestIntentRagConfig:
    def test_general_has_otc_filter(self):
        cfg = INTENT_RAG_CONFIG["general"]
        assert cfg is not None
        assert "requiresPrescription:=false" in cfg["filter_by"]

    def test_product_search_no_rx_filter(self):
        cfg = INTENT_RAG_CONFIG["product_search"]
        assert cfg is not None
        # product_search không filter OTC-only
        assert "requiresPrescription:=false" not in cfg["filter_by"]

    def test_order_tracking_no_rag(self):
        assert INTENT_RAG_CONFIG.get("order_tracking") is None

    def test_loyalty_no_rag(self):
        assert INTENT_RAG_CONFIG.get("loyalty_inquiry") is None

    def test_coupon_no_rag(self):
        assert INTENT_RAG_CONFIG.get("coupon_inquiry") is None

    def test_return_no_rag(self):
        assert INTENT_RAG_CONFIG.get("return_request") is None

    def test_prescription_status_no_rag(self):
        assert INTENT_RAG_CONFIG.get("prescription_status") is None

    def test_general_searches_indications(self):
        cfg = INTENT_RAG_CONFIG["general"]
        assert "indications" in cfg["query_by"]

    def test_general_searches_active_ingredients(self):
        cfg = INTENT_RAG_CONFIG["general"]
        assert "activeIngredients" in cfg["query_by"]


# ════════════════════════════════════════════════════════════════
# search_products_for_rag — Main RAG function (mocked)
# ════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestSearchProductsForRag:
    async def test_returns_empty_for_non_rag_intent(self):
        """Intent không cần RAG trả về [] ngay, không gọi Typesense."""
        result = await search_products_for_rag("Đơn hàng đâu?", "order_tracking")
        assert result == []

    async def test_returns_empty_for_loyalty_intent(self):
        result = await search_products_for_rag("Bao nhiêu điểm?", "loyalty_inquiry")
        assert result == []

    async def test_returns_empty_when_no_api_key(self):
        """Không có API key → trả về [], không raise exception."""
        with patch.dict("os.environ", {"TYPESENSE_API_KEY": ""}):
            # Reimport để env được đọc lại
            from importlib import reload
            import src.rag.typesense_client as rag_mod
            reload(rag_mod)
            result = await rag_mod.search_products_for_rag("Paracetamol", "general")
            assert result == []

    async def test_returns_empty_on_timeout(self):
        """Typesense timeout → trả về [], không crash."""
        import httpx
        with patch("src.rag.typesense_client.TYPESENSE_API_KEY", "test-key"):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
                mock_client_cls.return_value = mock_client

                result = await search_products_for_rag("Paracetamol", "general")
                assert result == []

    async def test_returns_products_on_success(self):
        """Typesense trả về hits → map đúng format."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "hits": [
                {
                    "document": {
                        "mongoId": "abc123",
                        "name": "Paracetamol 500mg",
                        "slug": "paracetamol-500mg",
                        "price": 25000,
                        "featuredImage": "https://example.com/img.jpg",
                        "activeIngredients": "Paracetamol 500mg",
                        "indications": "Hạ sốt, giảm đau",
                        "requiresPrescription": False,
                        "categoryName": "Thuốc hạ sốt",
                        "brandName": "DHG",
                        "rating": 4.5,
                    }
                }
            ]
        })

        with patch("src.rag.typesense_client.TYPESENSE_API_KEY", "test-key"):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client_cls.return_value = mock_client

                result = await search_products_for_rag("đau đầu hạ sốt", "general")

                assert len(result) == 1
                assert result[0]["mongoId"] == "abc123"
                assert result[0]["name"] == "Paracetamol 500mg"
                assert result[0]["price"] == 25000
                assert result[0]["activeIngredients"] == "Paracetamol 500mg"
                assert result[0]["indications"] == "Hạ sốt, giảm đau"
                assert result[0]["requiresPrescription"] is False

    async def test_returns_empty_list_on_empty_hits(self):
        """Typesense trả về hits rỗng → []."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"hits": []})

        with patch("src.rag.typesense_client.TYPESENSE_API_KEY", "test-key"):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client_cls.return_value = mock_client

                result = await search_products_for_rag("xyz không tồn tại", "general")
                assert result == []
