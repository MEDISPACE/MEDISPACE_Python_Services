"""
svd_model.py - Collaborative Filtering dung SVD (Matrix Factorization)
Use case: "Danh Cho Ban" khi co du du lieu (>= 10 users co orders)
"""
import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from typing import List, Dict, Optional, Tuple

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../saved_models")
SVD_MIN_USERS = int(os.getenv("SVD_MIN_USERS", "10"))
N_FACTORS = 20  # So latent factors


class SVDRecommender:
    def __init__(self):
        self.is_trained = False
        self.predicted_matrix: Optional[np.ndarray] = None
        self.user_index: Dict[str, int] = {}    # userId -> row
        self.product_index: Dict[str, int] = {} # productId -> col
        self.index_product: Dict[int, str] = {} # col -> productId
        self.user_product_df: Optional[pd.DataFrame] = None  # original for filtering

    def train(self, interaction_df: pd.DataFrame) -> None:
        """
        Train SVD model tren interaction matrix.
        interaction_df: DataFrame [user_id, product_id, score]
        """
        unique_users = interaction_df['user_id'].nunique()
        unique_products = interaction_df['product_id'].nunique()

        if unique_users < SVD_MIN_USERS:
            print(f"[SVD] Not enough users ({unique_users} < {SVD_MIN_USERS}). Skipping training.")
            self.is_trained = False
            return

        print(f"[SVD] Training on {unique_users} users x {unique_products} products...")

        try:
            # Build pivot matrix
            pivot = interaction_df.pivot_table(
                index='user_id',
                columns='product_id',
                values='score',
                fill_value=0
            )

            self.user_index = {uid: i for i, uid in enumerate(pivot.index)}
            self.product_index = {pid: j for j, pid in enumerate(pivot.columns)}
            self.index_product = {j: pid for pid, j in self.product_index.items()}

            # Store original interactions (de filter da mua)
            self.user_product_df = interaction_df

            # Convert to sparse matrix
            matrix = csr_matrix(pivot.values, dtype=np.float64)

            # SVD decomposition
            k = min(N_FACTORS, min(matrix.shape) - 1)
            U, sigma, Vt = svds(matrix, k=k)

            # Reconstruct full predicted matrix
            sigma_diag = np.diag(sigma)
            self.predicted_matrix = np.dot(np.dot(U, sigma_diag), Vt)

            self.is_trained = True
            self._save()
            print(f"[SVD] Trained! Factors: {k}, Matrix: {self.predicted_matrix.shape}")

        except Exception as e:
            print(f"[SVD] Training error: {e}")
            self.is_trained = False

    def _save(self):
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        np.save(os.path.join(SAVED_MODELS_DIR, "svd_matrix.npy"), self.predicted_matrix)
        joblib.dump(self.user_index, os.path.join(SAVED_MODELS_DIR, "svd_user_index.pkl"))
        joblib.dump(self.product_index, os.path.join(SAVED_MODELS_DIR, "svd_product_index.pkl"))
        joblib.dump(self.index_product, os.path.join(SAVED_MODELS_DIR, "svd_index_product.pkl"))
        if self.user_product_df is not None:
            self.user_product_df.to_pickle(os.path.join(SAVED_MODELS_DIR, "svd_interactions.pkl"))

    def load(self) -> bool:
        try:
            self.predicted_matrix = np.load(os.path.join(SAVED_MODELS_DIR, "svd_matrix.npy"))
            self.user_index = joblib.load(os.path.join(SAVED_MODELS_DIR, "svd_user_index.pkl"))
            self.product_index = joblib.load(os.path.join(SAVED_MODELS_DIR, "svd_product_index.pkl"))
            self.index_product = joblib.load(os.path.join(SAVED_MODELS_DIR, "svd_index_product.pkl"))
            self.user_product_df = pd.read_pickle(os.path.join(SAVED_MODELS_DIR, "svd_interactions.pkl"))
            self.is_trained = True
            print(f"[SVD] Loaded from disk ({len(self.user_index)} users, {len(self.product_index)} products)")
            return True
        except Exception as e:
            print(f"[SVD] Could not load: {e}")
            return False

    def can_predict_for_user(self, user_id: str) -> bool:
        return self.is_trained and user_id in self.user_index

    async def get_for_user(self, user_id: str, limit: int = 12) -> Tuple[List[str], str]:
        """
        Tra ve personalized recommendations cho user_id.
        Returns: (list of productIds, algorithm_name)
        """
        if not self.is_trained or self.predicted_matrix is None:
            return [], "svd_not_ready"

        if user_id not in self.user_index:
            return [], "svd_user_not_found"

        user_idx = self.user_index[user_id]
        user_predictions = self.predicted_matrix[user_idx]

        # Get products user da tuong tac (de exclude)
        already_interacted = set()
        if self.user_product_df is not None:
            interacted = self.user_product_df[
                self.user_product_df['user_id'] == user_id
            ]['product_id'].tolist()
            already_interacted = set(interacted)

        # Sort by predicted score DESC, exclude da tuong tac
        sorted_indices = np.argsort(user_predictions)[::-1]

        recommendations = []
        for idx in sorted_indices:
            pid = self.index_product.get(idx)
            if pid and pid not in already_interacted:
                recommendations.append(pid)
            if len(recommendations) >= limit:
                break

        return recommendations, "svd"

    async def get_for_user_scored(self, user_id: str, limit: int = 12) -> Tuple[List[Dict], str]:
        """Return personalized recommendations with raw SVD predicted scores."""
        if not self.is_trained or self.predicted_matrix is None:
            return [], "svd_not_ready"

        if user_id not in self.user_index:
            return [], "svd_user_not_found"

        user_idx = self.user_index[user_id]
        user_predictions = self.predicted_matrix[user_idx]

        already_interacted = set()
        if self.user_product_df is not None:
            interacted = self.user_product_df[
                self.user_product_df['user_id'] == user_id
            ]['product_id'].tolist()
            already_interacted = set(interacted)

        sorted_indices = np.argsort(user_predictions)[::-1]

        recommendations = []
        for idx in sorted_indices:
            pid = self.index_product.get(idx)
            if pid and pid not in already_interacted:
                recommendations.append({"productId": pid, "score": round(float(user_predictions[idx]), 6)})
            if len(recommendations) >= limit:
                break

        return recommendations, "svd"
