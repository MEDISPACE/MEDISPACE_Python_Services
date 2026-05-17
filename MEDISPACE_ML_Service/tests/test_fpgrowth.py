"""
tests/test_fpgrowth.py — Unit tests cho FPGrowthRecommender
"""
import pytest
from src.models.fpgrowth_model import FPGrowthRecommender


def make_baskets():
    """
    Transaction baskets giả lập:
    - p1 và p2 thường mua cùng nhau
    - p3 và p4 thường mua cùng nhau
    Cần >= 50 transactions để FP-Growth train
    """
    base = [
        ["p1", "p2", "p4"],
        ["p1", "p2"],
        ["p1", "p2", "p5"],
        ["p3", "p4"],
        ["p3", "p4", "p5"],
        ["p1", "p2", "p3"],
        ["p2", "p3", "p4"],
        ["p1", "p2", "p4", "p5"],
        ["p3", "p4", "p5"],
        ["p1", "p2"],
    ]
    # Repeat để vượt ngưỡng 50 transactions
    return base * 6  # 60 transactions


@pytest.fixture
def trained_model():
    m = FPGrowthRecommender()
    baskets = make_baskets()
    m.train(baskets)
    return m


class TestFPGrowthTraining:
    def test_trained_after_enough_transactions(self):
        m = FPGrowthRecommender()
        m.train(make_baskets())
        assert m.is_trained is True

    def test_not_trained_on_empty(self):
        m = FPGrowthRecommender()
        m.train([])
        assert m.is_trained is False

    def test_requires_min_transactions(self):
        m = FPGrowthRecommender()
        # ít hơn 50 transactions mặc định → không train
        m.train([["p1", "p2"]] * 3)
        assert m.is_trained is False

    def test_rules_generated(self, trained_model):
        assert trained_model.rules is not None
        assert len(trained_model.rules) > 0


class TestGetAssociated:
    @pytest.mark.asyncio
    async def test_p1_associated_with_p2(self, trained_model):
        """p1 và p2 xuất hiện cùng nhau nhiều nhất"""
        results = await trained_model.get_associated("p1", limit=5)
        assert "p2" in results

    @pytest.mark.asyncio
    async def test_excludes_self(self, trained_model):
        results = await trained_model.get_associated("p1", limit=10)
        assert "p1" not in results

    @pytest.mark.asyncio
    async def test_unknown_product_returns_empty(self, trained_model):
        results = await trained_model.get_associated("unknown_id", limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_not_trained_returns_empty(self):
        m = FPGrowthRecommender()
        results = await m.get_associated("p1")
        assert results == []

    @pytest.mark.asyncio
    async def test_limit_respected(self, trained_model):
        results = await trained_model.get_associated("p1", limit=2)
        assert len(results) <= 2
