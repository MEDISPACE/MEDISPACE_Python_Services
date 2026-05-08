"""
nmf_trending.py - NMF + Time-weighted Trending Score
Use case: "Xu Huong / Ban Chay" tren Home Page va fallback cho "Danh Cho Ban"
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from typing import List, Dict, Optional

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../saved_models")


class NMFTrendingRecommender:
    def __init__(self):
        self.is_trained = False
        self.trending_scores: pd.DataFrame = pd.DataFrame()  # productId, score, categoryId
        self.nmf_model: Optional[NMF] = None
        self.scaler = MinMaxScaler()
        self.category_trending: Dict[str, List[str]] = {}  # categoryId -> [productIds sorted by score]
        self.global_trending: List[str] = []

    def train(self, interaction_df: pd.DataFrame, products: List[Dict]) -> None:
        """
        Train NMF model va tinh trending scores.
        interaction_df: DataFrame voi columns [user_id, product_id, score]
        products: list products de lay metadata
        """
        if interaction_df.empty or len(interaction_df) < 5:
            print("[NMF] Not enough interactions. Using sales-count fallback.")
            self._fallback_train(products)
            return

        print(f"[NMF] Training on {len(interaction_df)} interactions...")

        try:
            # Pivot: user x product matrix
            pivot = interaction_df.pivot_table(
                index='user_id',
                columns='product_id',
                values='score',
                fill_value=0
            )

            n_users, n_products = pivot.shape
            n_components = min(20, n_users - 1, n_products - 1)
            if n_components < 2:
                self._fallback_train(products)
                return

            # Train NMF
            self.nmf_model = NMF(n_components=n_components, random_state=42, max_iter=200)
            W = self.nmf_model.fit_transform(pivot.values)  # user-topic
            H = self.nmf_model.components_                  # topic-product

            # Product popularity = sum of H columns (how much each product appears across topics)
            product_scores_nmf = H.sum(axis=0)

            # Build DataFrame
            product_ids = pivot.columns.tolist()
            scores_df = pd.DataFrame({
                'product_id': product_ids,
                'nmf_score': product_scores_nmf
            })

            # Merge voi product metadata
            product_meta = pd.DataFrame([{
                'product_id': str(p['_id']),
                'category_id': str(p.get('categoryId', '')),
                'rating': float(p.get('rating', 0)),
                'stock': int(p.get('stockQuantity', 0)),
            } for p in products])

            merged = scores_df.merge(product_meta, on='product_id', how='right')
            merged['nmf_score'] = merged['nmf_score'].fillna(0)

            # Filter: chi lay products con hang
            merged = merged[merged['stock'] > 0]

            # Final score = NMF score + rating bonus
            merged['final_score'] = (
                self.scaler.fit_transform(merged[['nmf_score']]).flatten() * 0.7 +
                (merged['rating'] / 5.0) * 0.3
            )

            self.trending_scores = merged.sort_values('final_score', ascending=False)
            self._build_indexes()
            self.is_trained = True
            self._save()
            print(f"[NMF] Trained! {len(self.trending_scores)} products scored")

        except Exception as e:
            print(f"[NMF] Training error: {e}. Using fallback.")
            self._fallback_train(products)

    def _fallback_train(self, products: List[Dict]) -> None:
        """Fallback: sort theo rating * solluong ban (khong can NMF)"""
        print("[NMF] Using rating-based fallback trending...")
        rows = []
        for p in products:
            stock = int(p.get('stockQuantity', 0))
            if stock <= 0:
                continue
            rows.append({
                'product_id': str(p['_id']),
                'category_id': str(p.get('categoryId', '')),
                'rating': float(p.get('rating', 0)),
                'stock': stock,
                'final_score': float(p.get('rating', 0))
            })
        if rows:
            self.trending_scores = pd.DataFrame(rows).sort_values('final_score', ascending=False)
            self._build_indexes()
            self.is_trained = True
            self._save()

    def _build_indexes(self):
        """Build global va per-category trending indexes"""
        self.global_trending = self.trending_scores['product_id'].tolist()

        self.category_trending = {}
        for _, row in self.trending_scores.iterrows():
            cat = row['category_id']
            if cat not in self.category_trending:
                self.category_trending[cat] = []
            self.category_trending[cat].append(row['product_id'])

    def _save(self):
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        self.trending_scores.to_pickle(os.path.join(SAVED_MODELS_DIR, "nmf_trending.pkl"))
        joblib.dump(self.global_trending, os.path.join(SAVED_MODELS_DIR, "nmf_global.pkl"))
        joblib.dump(self.category_trending, os.path.join(SAVED_MODELS_DIR, "nmf_category.pkl"))

    def load(self) -> bool:
        try:
            self.trending_scores = pd.read_pickle(os.path.join(SAVED_MODELS_DIR, "nmf_trending.pkl"))
            self.global_trending = joblib.load(os.path.join(SAVED_MODELS_DIR, "nmf_global.pkl"))
            self.category_trending = joblib.load(os.path.join(SAVED_MODELS_DIR, "nmf_category.pkl"))
            self.is_trained = True
            print(f"[NMF] Loaded from disk ({len(self.global_trending)} products)")
            return True
        except Exception as e:
            print(f"[NMF] Could not load: {e}")
            return False

    async def get_trending(self, category_id: Optional[str] = None, limit: int = 12) -> List[str]:
        """Tra ve trending products, optionally filtered by category"""
        if not self.is_trained:
            return []

        if category_id and category_id in self.category_trending:
            return self.category_trending[category_id][:limit]

        return self.global_trending[:limit]

    async def get_for_new_user(self, limit: int = 12) -> List[str]:
        """Cold-start: global trending cho user moi"""
        return await self.get_trending(limit=limit)

    async def get_filtered_by_categories(self, category_ids: List[str], limit: int = 12) -> List[str]:
        """Trending filtered theo nhieu categories (cho personalized fallback)"""
        if not self.is_trained or not category_ids:
            return self.global_trending[:limit]

        candidates = []
        seen = set()
        for cat_id in category_ids:
            for pid in self.category_trending.get(cat_id, []):
                if pid not in seen:
                    seen.add(pid)
                    candidates.append(pid)

        return candidates[:limit] if candidates else self.global_trending[:limit]
