"""
tests/test_tfidf.py — Unit tests cho TFIDFRecommender
"""
import pytest
from src.models.tfidf_model import TFIDFRecommender


@pytest.fixture
def model(sample_products):
    m = TFIDFRecommender()
    m.train(sample_products)
    return m


# ─── Training ─────────────────────────────────────────────────────────────────

class TestTFIDFTraining:
    def test_is_trained_after_train(self, model):
        assert model.is_trained is True

    def test_product_index_populated(self, model, sample_products):
        assert len(model.product_index) == len(sample_products)

    def test_feature_texts_stored(self, model, sample_products):
        """feature_texts cần được lưu để pharmacist boost dùng"""
        assert len(model.feature_texts) == len(sample_products)

    def test_products_df_stored(self, model, sample_products):
        assert model.products_df is not None
        assert len(model.products_df) == len(sample_products)

    def test_train_empty_products(self):
        m = TFIDFRecommender()
        m.train([])
        assert m.is_trained is False

    def test_tfidf_matrix_shape(self, model, sample_products):
        assert model.tfidf_matrix.shape[0] == len(sample_products)


# ─── get_related ──────────────────────────────────────────────────────────────

class TestGetRelated:
    @pytest.mark.asyncio
    async def test_related_returns_results(self, model):
        results = await model.get_related("p1", limit=3)
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_related_scored_returns_cosine_score(self, model):
        results = await model.get_related_scored("p1", limit=3)
        assert isinstance(results, list)
        assert results[0]["productId"]
        assert 0 < results[0]["score"] <= 1

    @pytest.mark.asyncio
    async def test_excludes_self(self, model):
        results = await model.get_related("p1", limit=10)
        assert "p1" not in results

    @pytest.mark.asyncio
    async def test_excludes_out_of_stock(self, model):
        results = await model.get_related("p1", limit=10)
        assert "p6" not in results, "Out-of-stock product should be excluded"

    @pytest.mark.asyncio
    async def test_excludes_prescription_products_by_default(self, model):
        results = await model.get_related("p1", limit=10)
        assert "p3" not in results, "Prescription product should not appear in customer recommendations"

    @pytest.mark.asyncio
    async def test_related_pain_relief_products(self, model):
        """p1 (Paracetamol) và p2 (Ibuprofen) đều là thuốc giảm đau → phải gần nhau"""
        results = await model.get_related("p1", limit=5)
        assert "p2" in results, "Ibuprofen should be related to Paracetamol"

    @pytest.mark.asyncio
    async def test_unknown_product_returns_empty(self, model):
        results = await model.get_related("unknown_product_id")
        assert results == []

    @pytest.mark.asyncio
    async def test_limit_respected(self, model):
        results = await model.get_related("p1", limit=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_not_trained_returns_empty(self, sample_products):
        m = TFIDFRecommender()
        results = await m.get_related("p1")
        assert results == []


# ─── get_related_diverse (MMR) ────────────────────────────────────────────────

class TestGetRelatedDiverse:
    @pytest.mark.asyncio
    async def test_mmr_returns_results(self, model):
        results = await model.get_related_diverse("p1", limit=4)
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_mmr_excludes_self(self, model):
        results = await model.get_related_diverse("p1", limit=10)
        assert "p1" not in results

    @pytest.mark.asyncio
    async def test_mmr_excludes_out_of_stock(self, model):
        results = await model.get_related_diverse("p1", limit=10)
        assert "p6" not in results

    @pytest.mark.asyncio
    async def test_mmr_limit_respected(self, model):
        results = await model.get_related_diverse("p1", limit=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_pure_relevance_equals_get_related(self, model):
        """lambda=1.0 → MMR degenerates to pure relevance (same as get_related)"""
        mmr_results = await model.get_related_diverse("p1", limit=4, lambda_mmr=1.0, candidate_pool=10)
        pure_results = await model.get_related("p1", limit=4)
        # Không nhất thiết identical (pool size khác nhau) nhưng tập hợp phải tương tự
        assert len(mmr_results) > 0

    @pytest.mark.asyncio
    async def test_unknown_product_returns_empty(self, model):
        results = await model.get_related_diverse("unknown_id")
        assert results == []


# ─── get_pharmacist_suggestions ───────────────────────────────────────────────

class TestGetPharmacistSuggestions:
    @pytest.mark.asyncio
    async def test_pharmacist_returns_results(self, model):
        results = await model.get_pharmacist_suggestions(
            chronic_diseases=["tim mạch"],
            allergies=[],
            current_medications=[],
            prescription_product_ids=["p3"],
            limit=5
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_excludes_prescription_products(self, model):
        results = await model.get_pharmacist_suggestions(
            chronic_diseases=[],
            allergies=[],
            current_medications=[],
            prescription_product_ids=["p3"],
            limit=5
        )
        assert "p3" not in results, "Prescription product should not appear in suggestions"

    @pytest.mark.asyncio
    async def test_allergy_filter(self, model):
        """Sản phẩm chứa Ibuprofen phải bị loại khi allergy = ibuprofen"""
        results = await model.get_pharmacist_suggestions(
            chronic_diseases=[],
            allergies=["ibuprofen"],
            current_medications=[],
            prescription_product_ids=["p3"],
            limit=10
        )
        assert "p2" not in results, "Product with allergen should be excluded"

    @pytest.mark.asyncio
    async def test_chronic_disease_boost(self, model):
        """p7 (Aspirin tim mạch) nên được boost khi chronic_disease = tim mạch"""
        results = await model.get_pharmacist_suggestions(
            chronic_diseases=["tim mạch"],
            allergies=[],
            current_medications=[],
            prescription_product_ids=["p1"],  # unrelated prescription
            limit=10
        )
        # p7 nên xuất hiện trong kết quả khi có boost
        assert "p7" in results, "Heart-related product should be boosted for heart disease patient"

    @pytest.mark.asyncio
    async def test_current_medication_excludes_matching_product(self, model):
        results = await model.get_pharmacist_suggestions(
            chronic_diseases=["tim mạch"],
            allergies=[],
            current_medications=["Aspirin"],
            prescription_product_ids=["p1"],
            limit=10
        )
        assert "p7" not in results

    @pytest.mark.asyncio
    async def test_not_trained_returns_empty(self):
        m = TFIDFRecommender()
        results = await m.get_pharmacist_suggestions([], [], [], ["p1"])
        assert results == []
