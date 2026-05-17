"""
hybrid_engine.py - Dieu phoi logic giua cac ML models va fallback chain
"""
import os
from typing import List, Tuple, Dict
from src.data.mongo_loader import mongo_loader, runtime_loader
from src.models.tfidf_model import TFIDFRecommender
from src.models.fpgrowth_model import FPGrowthRecommender
from src.models.nmf_trending import NMFTrendingRecommender
from src.models.svd_model import SVDRecommender


class HybridEngine:
    def __init__(
        self,
        tfidf: TFIDFRecommender,
        fpgrowth: FPGrowthRecommender,
        nmf: NMFTrendingRecommender,
        svd: SVDRecommender,
    ):
        self.tfidf = tfidf
        self.fpgrowth = fpgrowth
        self.nmf = nmf
        self.svd = svd

    async def train_all(self) -> None:
        """Train tat ca models theo thu tu. Goi khi khoi dong va retraining."""
        print("\n[HybridEngine] === START TRAINING ALL MODELS ===")
        mongo_loader.connect()

        try:
            # 1. Load data
            products = mongo_loader.load_products()
            baskets = mongo_loader.build_transaction_baskets()
            interaction_df = mongo_loader.build_interaction_matrix()

            # 2. TF-IDF (luon train duoc)
            self.tfidf.train(products)

            # 3. FP-Growth (can du baskets)
            self.fpgrowth.train(baskets)

            # 4. NMF Trending (train tren interaction hoac fallback)
            self.nmf.train(interaction_df, products)

            # 5. SVD (chi train khi du users)
            if not interaction_df.empty:
                self.svd.train(interaction_df)
            else:
                print("[SVD] No interaction data. Skipping.")

        except Exception as e:
            print(f"[HybridEngine] Training error: {e}")
            # Try loading from disk as fallback
            self._load_from_disk()
        finally:
            mongo_loader.disconnect()

        print("[HybridEngine] === TRAINING COMPLETE ===\n")

    def _load_from_disk(self):
        """Thu load models tu disk khi training that bai"""
        print("[HybridEngine] Attempting to load models from disk...")
        self.tfidf.load()
        self.fpgrowth.load()
        self.nmf.load()
        self.svd.load()

    async def get_personalized(self, user_id: str, limit: int = 12) -> Tuple[List[str], str]:
        """
        Fallback chain cho 'Danh Cho Ban':
        SVD (neu du data) → NMF filtered by user categories → NMF global
        """
        # Try SVD first
        if self.svd.can_predict_for_user(user_id):
            results, algo = await self.svd.get_for_user(user_id, limit)
            if results:
                return results, algo

        # Try NMF filtered by user's top categories
        # Sử dụng runtime_loader (persistent connection) — mongo_loader đã disconnect sau train_all()
        top_categories = runtime_loader.get_user_top_categories(user_id)
        if top_categories:
            results = await self.nmf.get_filtered_by_categories(top_categories, limit)
            if results:
                return results, "nmf_personalized"

        # Fallback to global trending
        results = await self.nmf.get_for_new_user(limit)
        return results, "nmf_trending"

    async def get_post_purchase(self, order_product_ids: List[str], limit: int = 8) -> List[str]:
        """
        Goi y sau khi dat hang.
        Strategy: FP-Growth associated + TF-IDF related, khong trung lap
        """
        seen = set(order_product_ids)
        results = []

        # FP-Growth: bought together
        for pid in order_product_ids:
            associated = await self.fpgrowth.get_associated(pid, limit=4)
            for r in associated:
                if r not in seen:
                    seen.add(r)
                    results.append(r)

        # Fill bang TF-IDF neu chua du
        if len(results) < limit:
            for pid in order_product_ids:
                related = await self.tfidf.get_related(pid, limit=6)
                for r in related:
                    if r not in seen:
                        seen.add(r)
                        results.append(r)
                    if len(results) >= limit:
                        break
                if len(results) >= limit:
                    break

        return results[:limit]

    async def get_pharmacist_suggestions(
        self,
        chronic_diseases: List[str],
        allergies: List[str],
        current_medications: List[str],
        prescription_product_ids: List[str],
        limit: int = 10
    ) -> List[str]:
        """
        Goi y cho Pharmacist dua tren medical context.
        Dung TF-IDF medical context tu tfidf_model.
        """
        return await self.tfidf.get_pharmacist_suggestions(
            chronic_diseases=chronic_diseases,
            allergies=allergies,
            current_medications=current_medications,
            prescription_product_ids=prescription_product_ids,
            limit=limit
        )
