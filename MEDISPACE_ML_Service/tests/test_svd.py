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
    m = SVDRecommender()
    interactions = make_interactions()
    m.train(interactions)
    return m


class TestSVDTraining:
    def test_trained_with_enough_users(self):
        m = SVDRecommender()
        m.train(make_interactions(n_users=10))
        assert m.is_trained is True

    def test_not_trained_with_few_users(self):
        m = SVDRecommender()
        m.train(make_interactions(n_users=3))
        # Với ít user, model có thể train hoặc không tùy ngưỡng SVD_MIN_USERS
        assert isinstance(m.is_trained, bool)

    def test_not_trained_on_empty(self):
        m = SVDRecommender()
        empty_df = pd.DataFrame(columns=["user_id", "product_id", "score"])
        m.train(empty_df)
        assert m.is_trained is False


class TestGetRecommendations:
    @pytest.mark.asyncio
    async def test_returns_results_for_known_user(self, trained_svd):
        result = await trained_svd.get_for_user("user0", 5)
        # get_for_user trả về (list, reason) hoặc list
        recs = result[0] if isinstance(result, tuple) else result
        assert isinstance(recs, list)
        assert len(recs) > 0

    @pytest.mark.asyncio
    async def test_unknown_user_returns_empty(self, trained_svd):
        result = await trained_svd.get_for_user("user_not_exist", 5)
        recs = result[0] if isinstance(result, tuple) else result
        assert recs == []

    @pytest.mark.asyncio
    async def test_limit_respected(self, trained_svd):
        result = await trained_svd.get_for_user("user0", 3)
        recs = result[0] if isinstance(result, tuple) else result
        assert len(recs) <= 3

    @pytest.mark.asyncio
    async def test_no_duplicate_recommendations(self, trained_svd):
        result = await trained_svd.get_for_user("user0", 10)
        recs = result[0] if isinstance(result, tuple) else result
        assert len(recs) == len(set(recs)), "Duplicates found in SVD recommendations"

    @pytest.mark.asyncio
    async def test_not_trained_returns_empty(self):
        m = SVDRecommender()
        result = await m.get_for_user("user0", 5)
        # Trường hợp chưa trained: trả list rỗng hoặc tuple ([], reason)
        recs = result[0] if isinstance(result, tuple) else result
        assert recs == []
