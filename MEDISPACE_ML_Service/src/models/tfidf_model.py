"""
tfidf_model.py - Content-Based Filtering dùng TF-IDF + Cosine Similarity
Use case: "San pham lien quan" tren Product Detail Page va Pharmacist panel
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Tuple

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../saved_models")


class TFIDFRecommender:
    def __init__(self):
        self.is_trained = False
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            analyzer='word',
            token_pattern=r"(?u)\b\w+\b",  # handle Vietnamese
        )
        self.tfidf_matrix = None
        self.product_index: Dict[str, int] = {}   # productId -> row index
        self.index_product: Dict[int, str] = {}   # row index -> productId
        self.products_df: Optional[pd.DataFrame] = None
        self.feature_texts: List[str] = []        # raw text per product (for keyword matching)

    def _build_feature_text(self, product: Dict) -> str:
        """
        Ghep cac truong text thanh 1 document cho TF-IDF.
        Weight: ten x3, hoat chat x3, chi dinh x2, category x2
        """
        parts = []
        name = product.get("name", "")
        parts.extend([name] * 3)

        details = product.get("details") or {}
        ingredients = details.get("activeIngredients", "")
        if ingredients:
            parts.extend([ingredients] * 3)

        indications = details.get("indications", "")
        if indications:
            parts.extend([indications] * 2)

        category = product.get("categoryName", "")
        if category:
            parts.extend([category] * 2)

        brand = product.get("brandName", "")
        if brand:
            parts.append(brand)

        dosage_form = details.get("dosageForm", "")
        if dosage_form:
            parts.append(dosage_form)

        description = product.get("shortDescription", "")
        if description:
            parts.append(description)

        return " ".join(filter(None, parts))

    def train(self, products: List[Dict]) -> None:
        """Train TF-IDF model tren danh sach products"""
        if not products:
            print("[TF-IDF] No products to train on")
            return

        print(f"[TF-IDF] Training on {len(products)} products...")

        # Build product index
        self.product_index = {}
        self.index_product = {}
        self.feature_texts = []

        for i, product in enumerate(products):
            pid = str(product["_id"])
            self.product_index[pid] = i
            self.index_product[i] = pid
            self.feature_texts.append(self._build_feature_text(product))

        # Store metadata
        self.products_df = pd.DataFrame([{
            "_id": str(p["_id"]),
            "name": p.get("name", ""),
            "categoryId": str(p.get("categoryId", "")),
            "brandId": str(p.get("brandId", "")),
            "requiresPrescription": p.get("requiresPrescription", False),
            "rating": p.get("rating", 0),
            "stockQuantity": p.get("stockQuantity", 0),
            "basePrice": (p.get("priceVariants") or [{}])[0].get("price", 0) if p.get("priceVariants") else 0
        } for p in products])

        # Fit TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(self.feature_texts)
        self.is_trained = True

        # Save to disk
        self._save()
        print(f"[TF-IDF] Trained! Matrix shape: {self.tfidf_matrix.shape}")

    def _save(self):
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(SAVED_MODELS_DIR, "tfidf_vectorizer.pkl"))
        joblib.dump(self.tfidf_matrix, os.path.join(SAVED_MODELS_DIR, "tfidf_matrix.pkl"))
        joblib.dump(self.product_index, os.path.join(SAVED_MODELS_DIR, "tfidf_index.pkl"))
        joblib.dump(self.index_product, os.path.join(SAVED_MODELS_DIR, "tfidf_index_rev.pkl"))
        joblib.dump(self.feature_texts, os.path.join(SAVED_MODELS_DIR, "tfidf_feature_texts.pkl"))
        if self.products_df is not None:
            self.products_df.to_pickle(os.path.join(SAVED_MODELS_DIR, "tfidf_products_df.pkl"))

    def load(self) -> bool:
        try:
            self.vectorizer = joblib.load(os.path.join(SAVED_MODELS_DIR, "tfidf_vectorizer.pkl"))
            self.tfidf_matrix = joblib.load(os.path.join(SAVED_MODELS_DIR, "tfidf_matrix.pkl"))
            self.product_index = joblib.load(os.path.join(SAVED_MODELS_DIR, "tfidf_index.pkl"))
            self.index_product = joblib.load(os.path.join(SAVED_MODELS_DIR, "tfidf_index_rev.pkl"))
            self.products_df = pd.read_pickle(os.path.join(SAVED_MODELS_DIR, "tfidf_products_df.pkl"))
            # feature_texts optional — có thể không tồn tại trên model cũ
            try:
                self.feature_texts = joblib.load(os.path.join(SAVED_MODELS_DIR, "tfidf_feature_texts.pkl"))
            except Exception:
                self.feature_texts = []
            self.is_trained = True
            print(f"[TF-IDF] Loaded from disk ({len(self.product_index)} products)")
            return True
        except Exception as e:
            print(f"[TF-IDF] Could not load from disk: {e}")
            return False

    async def get_related(
        self,
        product_id: str,
        limit: int = 8,
        exclude_prescription_mismatch: bool = True,
        exclude_prescription: bool = True
    ) -> List[str]:
        """Tra ve danh sach productId lien quan nhat"""
        if not self.is_trained or self.tfidf_matrix is None:
            return []

        idx = self.product_index.get(product_id)
        if idx is None:
            return []

        # Compute cosine similarity cho product nay
        product_vec = self.tfidf_matrix[idx]
        sim_scores = cosine_similarity(product_vec, self.tfidf_matrix).flatten()
        sim_scores[idx] = 0  # exclude chinh no

        # Filter: loai products het hang
        if self.products_df is not None:
            out_of_stock = self.products_df[self.products_df["stockQuantity"] <= 0]["_id"].tolist()
            for oos_id in out_of_stock:
                oos_idx = self.product_index.get(oos_id)
                if oos_idx is not None:
                    sim_scores[oos_idx] = 0

            if exclude_prescription:
                prescription_mask = self.products_df["requiresPrescription"].fillna(False).astype(bool)
                prescription_ids = self.products_df[prescription_mask]["_id"].tolist()
                for prescription_id in prescription_ids:
                    prescription_idx = self.product_index.get(prescription_id)
                    if prescription_idx is not None:
                        sim_scores[prescription_idx] = 0

            # Filter: uu tien cung requiresPrescription flag
            if exclude_prescription_mismatch and self.products_df is not None:
                current_rx = self.products_df.iloc[idx]["requiresPrescription"]
                diff_rx = self.products_df[self.products_df["requiresPrescription"] != current_rx]["_id"].tolist()
                for diff_id in diff_rx:
                    diff_idx = self.product_index.get(diff_id)
                    if diff_idx is not None:
                        sim_scores[diff_idx] *= 0.3  # giam trong so, khong loai han

        # Get top-N (raw candidates trước MMR)
        top_indices = np.argsort(sim_scores)[::-1][:limit]
        return [self.index_product[i] for i in top_indices if sim_scores[i] > 0]

    async def get_related_scored(
        self,
        product_id: str,
        limit: int = 8,
        exclude_prescription_mismatch: bool = True,
        exclude_prescription: bool = True
    ) -> List[Dict]:
        """Return related products with true cosine similarity scores."""
        if not self.is_trained or self.tfidf_matrix is None:
            return []

        idx = self.product_index.get(product_id)
        if idx is None:
            return []

        product_vec = self.tfidf_matrix[idx]
        sim_scores = cosine_similarity(product_vec, self.tfidf_matrix).flatten()
        sim_scores[idx] = 0

        if self.products_df is not None:
            out_of_stock = self.products_df[self.products_df["stockQuantity"] <= 0]["_id"].tolist()
            for oos_id in out_of_stock:
                oos_idx = self.product_index.get(oos_id)
                if oos_idx is not None:
                    sim_scores[oos_idx] = 0

            if exclude_prescription:
                prescription_mask = self.products_df["requiresPrescription"].fillna(False).astype(bool)
                prescription_ids = self.products_df[prescription_mask]["_id"].tolist()
                for prescription_id in prescription_ids:
                    prescription_idx = self.product_index.get(prescription_id)
                    if prescription_idx is not None:
                        sim_scores[prescription_idx] = 0

            if exclude_prescription_mismatch:
                current_rx = self.products_df.iloc[idx]["requiresPrescription"]
                diff_rx = self.products_df[self.products_df["requiresPrescription"] != current_rx]["_id"].tolist()
                for diff_id in diff_rx:
                    diff_idx = self.product_index.get(diff_id)
                    if diff_idx is not None:
                        sim_scores[diff_idx] *= 0.3

        top_indices = np.argsort(sim_scores)[::-1][:limit]
        return [
            {"productId": self.index_product[i], "score": round(float(sim_scores[i]), 6)}
            for i in top_indices
            if sim_scores[i] > 0
        ]

    async def get_related_diverse(
        self,
        product_id: str,
        limit: int = 8,
        lambda_mmr: float = 0.7,
        candidate_pool: int = 30,
        exclude_prescription_mismatch: bool = True,
        exclude_prescription: bool = True
    ) -> List[str]:
        """
        Sản phẩm liên quan với diversity via Maximal Marginal Relevance (MMR).

        MMR score = λ × relevance(c, query) − (1−λ) × max_similarity(c, selected)

        lambda_mmr = 1.0 → pure relevance (giống get_related)
        lambda_mmr = 0.0 → pure diversity
        lambda_mmr = 0.7 → 70% relevance, 30% diversity (mặc định khuyến nghị)

        Tránh filter bubble: không show 8 sản phẩm gần như giống nhau.
        """
        if not self.is_trained or self.tfidf_matrix is None:
            return []

        idx = self.product_index.get(product_id)
        if idx is None:
            return []

        # Lấy pool candidates rộng hơn từ get_related
        candidates = await self.get_related(
            product_id,
            limit=candidate_pool,
            exclude_prescription_mismatch=exclude_prescription_mismatch,
            exclude_prescription=exclude_prescription
        )
        if not candidates:
            return []

        # Lấy vector của query product
        query_vec = self.tfidf_matrix[idx]

        # Tính relevance score cho từng candidate
        candidate_indices = [self.product_index[pid] for pid in candidates if pid in self.product_index]
        if not candidate_indices:
            return candidates[:limit]

        # Ma trận similarity giữa candidates và query
        candidate_matrix = self.tfidf_matrix[candidate_indices]
        relevance_scores = cosine_similarity(query_vec, candidate_matrix).flatten()

        # Ma trận similarity giữa candidates với nhau (dùng cho diversity penalty)
        inter_sim = cosine_similarity(candidate_matrix, candidate_matrix)

        # MMR selection
        selected_local_indices: List[int] = []  # index trong candidate_indices
        remaining = list(range(len(candidate_indices)))

        # Chọn candidate đầu tiên: relevance cao nhất
        best = int(np.argmax(relevance_scores))
        selected_local_indices.append(best)
        remaining.remove(best)

        while len(selected_local_indices) < min(limit, len(candidate_indices)) and remaining:
            best_mmr = -1.0
            best_idx = remaining[0]

            for i in remaining:
                rel = float(relevance_scores[i])
                # Max similarity với các item đã chọn
                max_sim_to_selected = max(
                    float(inter_sim[i, s]) for s in selected_local_indices
                )
                mmr = lambda_mmr * rel - (1.0 - lambda_mmr) * max_sim_to_selected
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected_local_indices.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected_local_indices]

    async def get_related_diverse_scored(
        self,
        product_id: str,
        limit: int = 8,
        lambda_mmr: float = 0.7,
        candidate_pool: int = 30,
        exclude_prescription_mismatch: bool = True,
        exclude_prescription: bool = True
    ) -> List[Dict]:
        """Return MMR-selected related products with cosine relevance scores."""
        if not self.is_trained or self.tfidf_matrix is None:
            return []

        idx = self.product_index.get(product_id)
        if idx is None:
            return []

        candidate_items = await self.get_related_scored(
            product_id,
            limit=candidate_pool,
            exclude_prescription_mismatch=exclude_prescription_mismatch,
            exclude_prescription=exclude_prescription
        )
        candidates = [item["productId"] for item in candidate_items]
        score_by_id = {item["productId"]: float(item["score"]) for item in candidate_items}
        if not candidates:
            return []

        query_vec = self.tfidf_matrix[idx]
        candidate_indices = [self.product_index[pid] for pid in candidates if pid in self.product_index]
        if not candidate_indices:
            return candidate_items[:limit]

        candidate_matrix = self.tfidf_matrix[candidate_indices]
        relevance_scores = cosine_similarity(query_vec, candidate_matrix).flatten()
        inter_sim = cosine_similarity(candidate_matrix, candidate_matrix)

        selected_local_indices: List[int] = []
        remaining = list(range(len(candidate_indices)))
        best = int(np.argmax(relevance_scores))
        selected_local_indices.append(best)
        remaining.remove(best)

        while len(selected_local_indices) < min(limit, len(candidate_indices)) and remaining:
            best_mmr = -1.0
            best_idx = remaining[0]
            for i in remaining:
                rel = float(relevance_scores[i])
                max_sim_to_selected = max(float(inter_sim[i, s]) for s in selected_local_indices)
                mmr = lambda_mmr * rel - (1.0 - lambda_mmr) * max_sim_to_selected
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            selected_local_indices.append(best_idx)
            remaining.remove(best_idx)

        return [
            {"productId": candidates[i], "score": round(score_by_id.get(candidates[i], float(relevance_scores[i])), 6)}
            for i in selected_local_indices
        ]

    async def get_pharmacist_suggestions(
        self,
        chronic_diseases: List[str],
        allergies: List[str],
        current_medications: List[str],
        prescription_product_ids: List[str],
        limit: int = 10
    ) -> List[str]:
        """
        Gợi ý cho Pharmacist dựa trên medical context.

        Chiến lược:
          1. Tìm sản phẩm liên quan (TF-IDF) tới các thuốc trong đơn
          2. Boost +0.3 nếu sản phẩm liên quan đến bệnh mãn tính của bệnh nhân
          3. Loại bỏ sản phẩm nếu tên/thành phần có trong danh sách dị ứng
          4. Loại bỏ sản phẩm đã có trong đơn thuốc
        """
        if not self.is_trained:
            return []

        candidate_scores: Dict[str, float] = {}

        # 1. Lấy related products từ các sản phẩm trong đơn thuốc
        for pid in prescription_product_ids:
            related = await self.get_related(
                pid,
                limit=20,
                exclude_prescription_mismatch=False,
                exclude_prescription=False
            )
            for r_pid in related:
                p_idx = self.product_index.get(pid)
                r_idx = self.product_index.get(r_pid)
                if p_idx is not None and r_idx is not None:
                    sim = float(
                        cosine_similarity(self.tfidf_matrix[p_idx], self.tfidf_matrix[r_idx]).flatten()[0]
                    )
                    candidate_scores[r_pid] = max(candidate_scores.get(r_pid, 0), sim)

        # 2. Boost theo chronic diseases (keyword match trong feature text)
        if chronic_diseases and self.feature_texts:
            for disease in chronic_diseases:
                disease_lower = disease.lower().strip()
                if len(disease_lower) < 3:
                    continue
                for idx, text in enumerate(self.feature_texts):
                    pid = self.index_product.get(idx)
                    if pid and pid not in prescription_product_ids and disease_lower in text.lower():
                        candidate_scores[pid] = candidate_scores.get(pid, 0) + 0.3

        # 3. Loại bỏ sản phẩm khớp với dị ứng của bệnh nhân
        if allergies and self.feature_texts:
            allergy_blocked = set()
            for allergy in allergies:
                allergy_lower = allergy.lower().strip()
                if len(allergy_lower) < 3:
                    continue
                for idx, text in enumerate(self.feature_texts):
                    pid = self.index_product.get(idx)
                    if pid and allergy_lower in text.lower():
                        allergy_blocked.add(pid)
            for pid in allergy_blocked:
                candidate_scores.pop(pid, None)

        # 4. Loại sản phẩm trùng với thuốc bệnh nhân đang sử dụng.
        # Đây chỉ là guardrail từ khóa, không thay thế drug-interaction checker.
        if current_medications and self.feature_texts:
            for medication in current_medications:
                medication_lower = medication.lower().strip()
                if len(medication_lower) < 3:
                    continue
                for idx, text in enumerate(self.feature_texts):
                    pid = self.index_product.get(idx)
                    if pid and medication_lower in text.lower():
                        candidate_scores.pop(pid, None)

        # 5. Loại bỏ sản phẩm đã có trong đơn thuốc
        for pid in prescription_product_ids:
            candidate_scores.pop(pid, None)

        # Automatic pharmacist suggestions remain OTC-only. Prescription items
        # must be selected explicitly and independently reviewed.
        if self.products_df is not None:
            for _, product in self.products_df.iterrows():
                if bool(product.get("requiresPrescription", False)):
                    candidate_scores.pop(str(product["_id"]), None)

        # Sort và return top-N
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in sorted_candidates[:limit]]

    async def get_pharmacist_suggestions_scored(
        self,
        chronic_diseases: List[str],
        allergies: List[str],
        current_medications: List[str],
        prescription_product_ids: List[str],
        limit: int = 10
    ) -> List[Dict]:
        if not self.is_trained:
            return []

        ids = await self.get_pharmacist_suggestions(
            chronic_diseases,
            allergies,
            current_medications,
            prescription_product_ids,
            limit
        )
        if not ids:
            return []

        candidate_scores: Dict[str, float] = {}
        for pid in prescription_product_ids:
            p_idx = self.product_index.get(pid)
            if p_idx is None:
                continue
            related = await self.get_related_scored(
                pid,
                limit=20,
                exclude_prescription_mismatch=False,
                exclude_prescription=False
            )
            for item in related:
                candidate_scores[item["productId"]] = max(candidate_scores.get(item["productId"], 0), float(item["score"]))

        if chronic_diseases and self.feature_texts:
            for disease in chronic_diseases:
                disease_lower = disease.lower().strip()
                if len(disease_lower) < 3:
                    continue
                for idx, text in enumerate(self.feature_texts):
                    pid = self.index_product.get(idx)
                    if pid and pid in ids and disease_lower in text.lower():
                        candidate_scores[pid] = candidate_scores.get(pid, 0) + 0.3

        return [
            {"productId": pid, "score": round(float(candidate_scores.get(pid, 0)), 6)}
            for pid in ids
        ]
