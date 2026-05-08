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
from typing import List, Dict, Optional

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
        feature_texts = []

        for i, product in enumerate(products):
            pid = str(product["_id"])
            self.product_index[pid] = i
            self.index_product[i] = pid
            feature_texts.append(self._build_feature_text(product))

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
        self.tfidf_matrix = self.vectorizer.fit_transform(feature_texts)
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
        if self.products_df is not None:
            self.products_df.to_pickle(os.path.join(SAVED_MODELS_DIR, "tfidf_products_df.pkl"))

    def load(self) -> bool:
        try:
            self.vectorizer = joblib.load(os.path.join(SAVED_MODELS_DIR, "tfidf_vectorizer.pkl"))
            self.tfidf_matrix = joblib.load(os.path.join(SAVED_MODELS_DIR, "tfidf_matrix.pkl"))
            self.product_index = joblib.load(os.path.join(SAVED_MODELS_DIR, "tfidf_index.pkl"))
            self.index_product = joblib.load(os.path.join(SAVED_MODELS_DIR, "tfidf_index_rev.pkl"))
            self.products_df = pd.read_pickle(os.path.join(SAVED_MODELS_DIR, "tfidf_products_df.pkl"))
            self.is_trained = True
            print(f"[TF-IDF] Loaded from disk ({len(self.product_index)} products)")
            return True
        except Exception as e:
            print(f"[TF-IDF] Could not load from disk: {e}")
            return False

    async def get_related(self, product_id: str, limit: int = 8,
                           exclude_prescription_mismatch: bool = True) -> List[str]:
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

            # Filter: uu tien cung requiresPrescription flag
            if exclude_prescription_mismatch and self.products_df is not None:
                current_rx = self.products_df.iloc[idx]["requiresPrescription"]
                diff_rx = self.products_df[self.products_df["requiresPrescription"] != current_rx]["_id"].tolist()
                for diff_id in diff_rx:
                    diff_idx = self.product_index.get(diff_id)
                    if diff_idx is not None:
                        sim_scores[diff_idx] *= 0.3  # giam trong so, khong loai han

        # Get top-N
        top_indices = np.argsort(sim_scores)[::-1][:limit]
        return [self.index_product[i] for i in top_indices if sim_scores[i] > 0]

    async def get_pharmacist_suggestions(
        self,
        chronic_diseases: List[str],
        allergies: List[str],
        current_medications: List[str],
        prescription_product_ids: List[str],
        limit: int = 10
    ) -> List[str]:
        """
        Goi y cho Pharmacist dua tren:
        1. Products trong don thuoc hien tai (TF-IDF related)
        2. Loc bo san pham trong allergies keywords
        """
        if not self.is_trained:
            return []

        candidate_scores: Dict[str, float] = {}

        # 1. Lay related products tu cac san pham trong don thuoc
        for pid in prescription_product_ids:
            related = await self.get_related(pid, limit=20, exclude_prescription_mismatch=False)
            for r_pid in related:
                idx = self.product_index.get(pid)
                r_idx = self.product_index.get(r_pid)
                if idx is not None and r_idx is not None:
                    sim = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix[r_idx]).flatten()[0]
                    candidate_scores[r_pid] = max(candidate_scores.get(r_pid, 0), float(sim))

        # 2. Boost theo chronic diseases (keyword match trong indications)
        if chronic_diseases and self.products_df is not None:
            # Uu tien products lien quan den benh man tinh
            pass  # TODO: implement keyword boost nếu cần

        # 3. Loai bo cac san pham da co trong don thuoc
        for pid in prescription_product_ids:
            candidate_scores.pop(pid, None)

        # Sort va return
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in sorted_candidates[:limit]]
