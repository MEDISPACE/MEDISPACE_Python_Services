"""
tests/test_svd.py — Unit tests cho SVDRecommender
"""
import pytest
import pandas as pd
from src.models.svd_model import SVDRecommender


def make_interactions(n_users: int = 15, n_items: int = 10):
    """Tạo interaction matrix đủ lớn cho SVD (cần ≥ SVD_MIN_USERS)"""
    import numpy as np
    rows = []
    np.random.seed(42)
    for u in range(n_users):
        uid = f"user{u}"
        for p in range(n_items):
            pid = f"prod{p}"
            if np.random.random() > 0.4:  # 60% density
                rows.append((uid, pid, float(np.random.randint(1, 6))))
    return pd.DataFrame(rows, columns=["user_id", "product_id", "score"])


@pytest.fixture
def trained_svd():
    m = SVDRecommender(min_users=5, n_components=5)
    interactions = make_interactions()
    m.train(interactions)
    return m


class TestSVDTraining:
    def test_trained_with_enough_users(self):
        m = SVDRecommender(min_users=5, n_components=5)
        m.train(make_interactions(n_users=10))
        assert m.is_trained is True

    def test_not_trained_with_few_users(self):
        m = SVDRecommender(min_users=10, n_components=5)
        m.train(make_interactions(n_users=3))
        assert m.is_trained is False

    def test_not_trained_on_empty(self):
        m = SVDRecommender()
        m.train(pd.DataFrame())
        assert m.is_trained is False


class TestGetRecommendations:
    @pytest.mark.asyncio
    async def test_returns_results_for_known_user(self, trained_svd):
        results = await trained_svd.get_recommendations("user0", limit=5)
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_unknown_user_returns_empty(self, trained_svd):
        results = await trained_svd.get_recommendations("user_not_exist", limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_limit_respected(self, trained_svd):
        results = await trained_svd.get_recommendations("user0", limit=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_no_duplicate_recommendations(self, trained_svd):
        results = await trained_svd.get_recommendations("user0", limit=10)
        assert len(results) == len(set(results)), "Duplicates found in SVD recommendations"

    @pytest.mark.asyncio
    async def test_not_trained_returns_empty(self):
        m = SVDRecommender()
        results = await m.get_recommendations("user0")
        assert results == []
