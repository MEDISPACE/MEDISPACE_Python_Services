"""
tests/test_nmf.py — Unit tests cho NMFTrendingRecommender
"""
import pytest
from src.models.nmf_trending import NMFTrendingRecommender


def make_interactions():
    """Ma trận user-item interactions giả lập"""
    import pandas as pd
    rows = [
        ("user1", "p1", 5.0),
        ("user1", "p2", 3.0),
        ("user1", "p4", 2.0),
        ("user2", "p1", 4.0),
        ("user2", "p3", 5.0),
        ("user3", "p2", 3.0),
        ("user3", "p3", 4.0),
        ("user3", "p5", 2.0),
        ("user4", "p1", 5.0),
        ("user4", "p4", 3.0),
        ("user5", "p2", 4.0),
        ("user5", "p5", 5.0),
    ]
    return pd.DataFrame(rows, columns=["user_id", "product_id", "score"])


def make_products():
    return [
        {"_id": "p1", "categoryId": "cat1", "isActive": True, "stockQuantity": 100, "rating": 4.5, "reviewCount": 50},
        {"_id": "p2", "categoryId": "cat1", "isActive": True, "stockQuantity": 80,  "rating": 4.0, "reviewCount": 30},
        {"_id": "p3", "categoryId": "cat2", "isActive": True, "stockQuantity": 60,  "rating": 4.2, "reviewCount": 20},
        {"_id": "p4", "categoryId": "cat1", "isActive": True, "stockQuantity": 0,   "rating": 3.5, "reviewCount": 10},  # OOS
        {"_id": "p5", "categoryId": "cat2", "isActive": True, "stockQuantity": 40,  "rating": 3.8, "reviewCount": 15},
    ]


@pytest.fixture
def trained_nmf():
    m = NMFTrendingRecommender()
    interactions = make_interactions()
    products = make_products()
    m.train(interactions, products)
    return m


class TestNMFTraining:
    def test_trained_after_train(self):
        m = NMFTrendingRecommender()
        m.train(make_interactions(), make_products())
        assert m.is_trained is True

    def test_not_trained_on_empty(self):
        m = NMFTrendingRecommender()
        import pandas as pd
        m.train(pd.DataFrame(), [])
        assert m.is_trained is False


class TestGetTrending:
    @pytest.mark.asyncio
    async def test_returns_list(self, trained_nmf):
        results = await trained_nmf.get_trending(category_id=None, limit=5)
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_excludes_out_of_stock(self, trained_nmf):
        results = await trained_nmf.get_trending(limit=10)
        assert "p4" not in results, "OOS product should not appear in trending"

    @pytest.mark.asyncio
    async def test_limit_respected(self, trained_nmf):
        results = await trained_nmf.get_trending(limit=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_category_filter(self, trained_nmf):
        results = await trained_nmf.get_trending(category_id="cat1", limit=10)
        # All returned products should be in cat1
        for pid in results:
            product = next((p for p in make_products() if p["_id"] == pid), None)
            if product:
                assert product["categoryId"] == "cat1"

    @pytest.mark.asyncio
    async def test_not_trained_returns_empty(self):
        m = NMFTrendingRecommender()
        results = await m.get_trending()
        assert results == []
