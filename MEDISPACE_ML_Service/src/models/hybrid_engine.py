"""
hybrid_engine.py - Dieu phoi logic giua cac ML models va fallback chain
"""
import os
import asyncio
import threading
from datetime import datetime, timezone
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
        self.model_version = "untrained"
        self.last_evaluation: Dict[str, object] = {}
        self._swap_lock = threading.Lock()

    async def train_all(self) -> None:
        """Train models off the FastAPI event loop."""
        await asyncio.to_thread(self._train_all_sync)

    def _train_all_sync(self) -> None:
        """Train tat ca models theo thu tu. Goi khi khoi dong va retraining."""
        print("\n[HybridEngine] === START TRAINING ALL MODELS ===")
        mongo_loader.connect()

        try:
            # 1. Load data
            products = mongo_loader.load_products()
            baskets = mongo_loader.build_transaction_baskets()
            interaction_df = mongo_loader.build_interaction_matrix()
            otc_products = [
                product for product in products
                if not bool(product.get("requiresPrescription", False))
            ]
            otc_product_ids = {str(product["_id"]) for product in otc_products}
            otc_baskets = [
                [product_id for product_id in basket if product_id in otc_product_ids]
                for basket in baskets
            ]
            otc_baskets = [basket for basket in otc_baskets if len(basket) >= 2]
            otc_interaction_df = interaction_df
            if not interaction_df.empty:
                otc_interaction_df = interaction_df[
                    interaction_df["product_id"].isin(otc_product_ids)
                ]

            candidate_tfidf = TFIDFRecommender()
            candidate_fpgrowth = FPGrowthRecommender()
            candidate_nmf = NMFTrendingRecommender()
            candidate_svd = SVDRecommender()

            # Train a shadow bundle. Serving models remain untouched until validation passes.
            candidate_tfidf.train(products)

            # 3. FP-Growth (can du baskets)
            candidate_fpgrowth.train(otc_baskets)

            # 4. NMF Trending (train tren interaction hoac fallback)
            candidate_nmf.train(otc_interaction_df, otc_products)

            # 5. SVD (chi train khi du users)
            if not otc_interaction_df.empty:
                candidate_svd.train(otc_interaction_df)
            else:
                print("[SVD] No interaction data. Skipping.")

            evaluation = {
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "catalog_size": len(products),
                "otc_catalog_size": len(otc_products),
                "interaction_count": len(otc_interaction_df),
                "basket_count": len(otc_baskets),
                "tfidf_ready": candidate_tfidf.is_trained,
                "nmf_ready": candidate_nmf.is_trained,
                "svd_ready": candidate_svd.is_trained,
                "fpgrowth_ready": candidate_fpgrowth.is_trained,
            }
            if not candidate_tfidf.is_trained or not candidate_nmf.is_trained:
                raise RuntimeError("Candidate bundle failed readiness gate")

            with self._swap_lock:
                self.tfidf = candidate_tfidf
                self.fpgrowth = candidate_fpgrowth
                self.nmf = candidate_nmf
                self.svd = candidate_svd
                self.model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                self.last_evaluation = evaluation

        except Exception as e:
            print(f"[HybridEngine] Training error: {e}")
            # Keep the currently serving bundle. Load from disk only during initial startup.
            if self.model_version == "untrained":
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
        if self.tfidf.is_trained and self.nmf.is_trained:
            self.model_version = "disk-fallback"

    @property
    def is_ready(self) -> bool:
        return self.tfidf.is_trained and self.nmf.is_trained

    def metrics(self) -> Dict[str, object]:
        return {
            "model_version": self.model_version,
            "ready": self.is_ready,
            "evaluation": self.last_evaluation,
            "coverage": {
                "tfidf_products": len(self.tfidf.product_index),
                "trending_products": len(self.nmf.global_trending),
                "svd_users": len(self.svd.user_index),
                "association_antecedents": len(self.fpgrowth.rules_dict),
            },
        }

    async def get_personalized(self, user_id: str, limit: int = 12) -> Tuple[List[str], str]:
        """
        Fallback chain cho 'Danh Cho Ban':
        SVD (neu du data) → NMF filtered by user categories → NMF global
        """
        feedback = await asyncio.to_thread(runtime_loader.get_user_feedback_exclusions, user_id)
        excluded = feedback["dismissed"] | feedback["snoozed"]

        def visible(values: List[str]) -> List[str]:
            return [value for value in values if value not in excluded][:limit]

        # Try SVD first
        if self.svd.can_predict_for_user(user_id):
            results, algo = await self.svd.get_for_user(user_id, limit * 3)
            if results:
                return visible(results), algo

        # Try NMF filtered by user's top categories
        # Sử dụng runtime_loader (persistent connection) — mongo_loader đã disconnect sau train_all()
        top_categories = await asyncio.to_thread(runtime_loader.get_user_top_categories, user_id)
        if top_categories:
            results = await self.nmf.get_filtered_by_categories(top_categories, limit * 3)
            if results:
                return visible(results), "nmf_personalized"

        # Fallback to global trending
        results = await self.nmf.get_for_new_user(limit * 3)
        return visible(results), "nmf_trending"

    async def get_post_purchase(self, order_product_ids: List[str], limit: int = 8) -> List[str]:
        """
        Gợi ý sau khi đặt hàng.
        Strategy: FP-Growth associated + TF-IDF MMR fill, không trùng lặp
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

        # Fill bằng TF-IDF MMR nếu chưa đủ
        if len(results) < limit:
            for pid in order_product_ids:
                related = await self.tfidf.get_related_diverse(pid, limit=6, lambda_mmr=0.65)
                for r in related:
                    if r not in seen:
                        seen.add(r)
                        results.append(r)
                    if len(results) >= limit:
                        break
                if len(results) >= limit:
                    break

        return results[:limit]

    async def get_replenishment(self, user_id: str, limit: int = 5) -> List[str]:
        """
        Predictive Replenishment: sản phẩm user cần mua lại.

        Phân tích chu kỳ mua hàng:
        - Nếu user mua sản phẩm X định kỳ (≥2 lần), tính avg interval
        - Nếu đã qua ≥80% interval kể từ lần mua cuối → gợi ý reorder

        Use case: thuốc uống hàng ngày, vitamin, thực phẩm chức năng.
        """
        from datetime import datetime, timedelta

        runtime_loader._ensure_connected()
        db = runtime_loader._loader.db
        if db is None:
            return []

        try:
            from bson import ObjectId
            uid = ObjectId(user_id)
        except Exception:
            return []

        try:
            # Chỉ dùng giao dịch đã giao thành công; ngày nhận hàng mới bắt đầu chu kỳ sử dụng.
            def _load_orders():
                return list(db["orders"].find(
                    {"userId": uid, "orderStatus": "delivered"},
                    {"items": 1, "deliveredAt": 1, "createdAt": 1}
                ).sort("deliveredAt", 1))

            orders = await asyncio.to_thread(_load_orders)

            if len(orders) < 2:
                return []

            # Tính purchase timeline per product
            purchase_timeline: Dict[str, List[datetime]] = {}
            for order in orders:
                purchase_at = order.get("deliveredAt") or order.get("createdAt")
                if not purchase_at:
                    continue
                for item in order.get("items", []):
                    pid = str(item.get("productId", ""))
                    if not pid:
                        continue
                    if pid not in purchase_timeline:
                        purchase_timeline[pid] = []
                    purchase_timeline[pid].append(purchase_at)

            # Tìm sản phẩm đến hạn mua lại
            now = datetime.now()
            due_products: List[tuple] = []  # (pid, overdue_ratio)

            for pid, dates in purchase_timeline.items():
                if len(dates) < 2:
                    continue  # Cần ≥2 lần mua để tính interval

                # Tính avg interval (ngày)
                intervals = [
                    (dates[i + 1] - dates[i]).days
                    for i in range(len(dates) - 1)
                ]
                avg_interval = sum(intervals) / len(intervals)
                if avg_interval < 7:  # Bỏ qua interval < 1 tuần (noise)
                    continue

                last_purchase = dates[-1]
                days_since_last = (now - last_purchase).days
                overdue_ratio = days_since_last / avg_interval

                # Gợi ý khi đã qua ≥80% chu kỳ
                if overdue_ratio >= 0.8:
                    due_products.append((pid, overdue_ratio))

            # Sort theo mức độ "quá hạn" — sản phẩm cần nhất lên đầu
            due_products.sort(key=lambda x: x[1], reverse=True)
            due_product_ids = [pid for pid, _ in due_products]

            def _load_eligible_product_ids():
                object_ids = []
                for product_id in due_product_ids:
                    try:
                        object_ids.append(ObjectId(product_id))
                    except Exception:
                        continue
                products = db["products"].find(
                    {
                        "_id": {"$in": object_ids},
                        "isActive": True,
                        "stockQuantity": {"$gt": 0},
                        "requiresPrescription": {"$ne": True}
                    },
                    {"_id": 1}
                )
                return {str(product["_id"]) for product in products}

            eligible_product_ids = await asyncio.to_thread(_load_eligible_product_ids)
            feedback = await asyncio.to_thread(runtime_loader.get_user_feedback_exclusions, user_id)
            excluded = feedback["dismissed"] | feedback["snoozed"]
            return [
                product_id for product_id in due_product_ids
                if product_id in eligible_product_ids and product_id not in excluded
            ][:limit]

        except Exception as e:
            print(f"[HybridEngine] get_replenishment error: {e}")
            return []

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
