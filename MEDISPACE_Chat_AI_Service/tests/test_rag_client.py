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
    _expand_semantic_query,
    _is_irrelevant_for_fever_query,
    _normalize_ascii,
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

    def test_expands_nong_trong_nguoi_to_liver_support_terms(self):
        result = _expand_semantic_query("Tôi cảm thấy nóng trong người")
        normalized = _normalize_ascii(result)
        assert "thanh nhiet" in normalized
        assert "mat gan" in normalized
        assert "giai doc gan" in normalized
        assert "nóng" not in result

    def test_expands_mat_nuoc_to_oral_rehydration_terms(self):
        result = _expand_semantic_query("Tôi bị mất nước sau tiêu chảy")
        normalized = _normalize_ascii(result)
        assert "oresol" in normalized
        assert "dien giai" in normalized
        assert "bu nuoc" in normalized

    def test_expands_fever_to_antipyretic_terms(self):
        result = _expand_semantic_query("Tôi cảm thấy sốt, mệt mỏi trong người")
        normalized = _normalize_ascii(result)
        assert "paracetamol" in normalized
        assert "ha sot" in normalized

    def test_fever_query_filters_acne_products(self):
        acne_doc = {
            "name": "Miếng dán mụn Somaderm Spot",
            "categoryName": "Chăm sóc da mụn",
            "indications": "Giúp che phủ và làm dịu vùng da mụn",
        }
        assert _is_irrelevant_for_fever_query("sốt mệt mỏi", acne_doc) is True

    def test_fever_query_keeps_antipyretic_products(self):
        fever_doc = {
            "name": "Panadol",
            "categoryName": "Thuốc giảm đau hạ sốt",
            "activeIngredients": "Paracetamol",
            "indications": "Giảm đau và hạ sốt",
        }
        assert _is_irrelevant_for_fever_query("sốt mệt mỏi", fever_doc) is False

    def test_fever_query_filters_detox_related_paracetamol_docs(self):
        detox_doc = {
            "name": "Dung dịch uống Acetuss",
            "categoryName": "Thuốc trị ho cảm",
            "activeIngredients": "N-acetylcysteine",
            "indications": "Được làm thuốc giải độc trong quá liều paracetamol",
        }
        assert _is_irrelevant_for_fever_query("sốt mệt mỏi", detox_doc) is True


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

                result = await search_products_for_rag("đau đầu", "general")

                assert len(result) == 1
                assert result[0]["mongoId"] == "abc123"
                assert result[0]["name"] == "Paracetamol 500mg"
                assert result[0]["price"] == 25000
                assert result[0]["activeIngredients"] == "Paracetamol 500mg"
                assert result[0]["indications"] == "Hạ sốt, giảm đau"
                assert result[0]["requiresPrescription"] is False

                params = mock_client.get.await_args.kwargs["params"]
                assert "embedding" in params["query_by"].split(",")
                assert len(params["query_by"].split(",")) == len(params["query_by_weights"].split(","))
                assert len(params["query_by"].split(",")) == len(params["num_typos"].split(","))
                assert len(params["query_by"].split(",")) == len(params["prefix"].split(","))

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

    async def test_retries_bm25_when_vector_search_fails(self):
        """Vector query loi thi retry BM25-only truoc khi tra rong."""
        import httpx

        error_response = MagicMock()
        error_response.status_code = 400
        error_response.text = "Unknown field embedding"
        vector_error = httpx.HTTPStatusError(
            "vector error",
            request=MagicMock(),
            response=error_response,
        )

        vector_response = MagicMock()
        vector_response.raise_for_status = MagicMock(side_effect=vector_error)

        bm25_response = MagicMock()
        bm25_response.raise_for_status = MagicMock()
        bm25_response.json = MagicMock(return_value={
            "hits": [
                {
                    "document": {
                        "mongoId": "bm25-1",
                        "name": "Vitamin C",
                        "slug": "vitamin-c",
                        "price": 100000,
                        "requiresPrescription": False,
                    }
                }
            ]
        })

        with patch("src.rag.typesense_client.TYPESENSE_API_KEY", "test-key"):
            with patch("src.rag.typesense_client._VECTOR_SEARCH_ENABLED", True):
                with patch("httpx.AsyncClient") as mock_client_cls:
                    mock_client = AsyncMock()
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client.get = AsyncMock(side_effect=[vector_response, bm25_response])
                    mock_client_cls.return_value = mock_client

                    result = await search_products_for_rag("vitamin c", "general")

                    assert len(result) == 1
                    assert result[0]["mongoId"] == "bm25-1"
                    assert mock_client.get.await_count == 2
                    first_params = mock_client.get.await_args_list[0].kwargs["params"]
                    second_params = mock_client.get.await_args_list[1].kwargs["params"]
                    assert "vector_query" in first_params
                    assert "vector_query" not in second_params

    async def test_fever_queries_skip_vector_search(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"hits": []})

        with patch("src.rag.typesense_client.TYPESENSE_API_KEY", "test-key"):
            with patch("src.rag.typesense_client._VECTOR_SEARCH_ENABLED", True):
                with patch("httpx.AsyncClient") as mock_client_cls:
                    mock_client = AsyncMock()
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client.get = AsyncMock(return_value=mock_response)
                    mock_client_cls.return_value = mock_client

                    await search_products_for_rag("Tôi cảm thấy sốt, mệt mỏi trong người", "general")

                    params = mock_client.get.await_args.kwargs["params"]
                    assert "vector_query" not in params
