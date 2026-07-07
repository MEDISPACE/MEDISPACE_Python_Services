"""
fpgrowth_model.py - Association Rule Mining dung FP-Growth
Use case: "Thuong Mua Kem" tren Product Detail Page
"""
import os
import joblib
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from typing import List, Dict, Tuple

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../saved_models")

MIN_SUPPORT = float(os.getenv("FPGROWTH_MIN_SUPPORT", "0.01"))
MIN_CONFIDENCE = float(os.getenv("FPGROWTH_MIN_CONFIDENCE", "0.3"))
MIN_TRANSACTIONS = int(os.getenv("FPGROWTH_MIN_TRANSACTIONS", "50"))


class FPGrowthRecommender:
    def __init__(self):
        self.is_trained = False
        self.rules: pd.DataFrame = pd.DataFrame()
        self.rules_dict: Dict[str, List[Tuple[str, float]]] = {}  # productId -> [(assoc_productId, lift)]

    def train(self, baskets: List[List[str]]) -> None:
        """
        Train FP-Growth tren transaction baskets.
        baskets: [[productId1, productId2, ...], ...]
        """
        if len(baskets) < MIN_TRANSACTIONS:
            print(f"[FP-Growth] Not enough transactions ({len(baskets)} < {MIN_TRANSACTIONS}). Skipping.")
            self.is_trained = False
            return

        print(f"[FP-Growth] Training on {len(baskets)} baskets...")

        try:
            te = TransactionEncoder()
            te_array = te.fit(baskets).transform(baskets)
            df = pd.DataFrame(te_array, columns=te.columns_)

            # Mine frequent itemsets
            frequent_itemsets = fpgrowth(df, min_support=MIN_SUPPORT, use_colnames=True)

            if frequent_itemsets.empty:
                print("[FP-Growth] No frequent itemsets found. Try lowering min_support.")
                return

            # Generate association rules
            self.rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=MIN_CONFIDENCE
            )

            # Filter: lift > 1 (co correlation thuc su)
            self.rules = self.rules[self.rules["lift"] > 1.0]

            # Build lookup dict: antecedent productId -> [(consequent, lift)]
            self.rules_dict = {}
            for _, row in self.rules.iterrows():
                antecedents = list(row["antecedents"])
                consequents = list(row["consequents"])
                lift = float(row["lift"])
                confidence = float(row["confidence"])

                for ant in antecedents:
                    if ant not in self.rules_dict:
                        self.rules_dict[ant] = []
                    for cons in consequents:
                        self.rules_dict[ant].append((cons, lift * confidence))

            # Sort each entry by score DESC
            for ant in self.rules_dict:
                self.rules_dict[ant].sort(key=lambda x: x[1], reverse=True)

            self.is_trained = True
            self._save()
            print(f"[FP-Growth] Trained! {len(self.rules)} rules, {len(self.rules_dict)} unique antecedents")

        except Exception as e:
            print(f"[FP-Growth] Training error: {e}")
            self.is_trained = False

    def _save(self):
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        joblib.dump(self.rules_dict, os.path.join(SAVED_MODELS_DIR, "fp_rules_dict.pkl"))
        self.rules.to_pickle(os.path.join(SAVED_MODELS_DIR, "fp_rules.pkl"))

    def load(self) -> bool:
        try:
            self.rules_dict = joblib.load(os.path.join(SAVED_MODELS_DIR, "fp_rules_dict.pkl"))
            self.rules = pd.read_pickle(os.path.join(SAVED_MODELS_DIR, "fp_rules.pkl"))
            self.is_trained = True
            print(f"[FP-Growth] Loaded from disk ({len(self.rules_dict)} antecedents)")
            return True
        except Exception as e:
            print(f"[FP-Growth] Could not load: {e}")
            return False

    async def get_associated(self, product_id: str, limit: int = 6) -> List[str]:
        """Tra ve cac productId thuong mua kem voi product_id"""
        if not self.is_trained or product_id not in self.rules_dict:
            return []

        results = self.rules_dict[product_id]
        seen = set()
        unique_results = []
        for pid, _ in results:
            if pid != product_id and pid not in seen:
                seen.add(pid)
                unique_results.append(pid)
            if len(unique_results) >= limit:
                break

        return unique_results

    async def get_associated_scored(self, product_id: str, limit: int = 6) -> List[Dict]:
        """Return associated products with lift * confidence scores."""
        if not self.is_trained or product_id not in self.rules_dict:
            return []

        results = self.rules_dict[product_id]
        seen = set()
        unique_results = []
        for pid, score in results:
            if pid != product_id and pid not in seen:
                seen.add(pid)
                unique_results.append({"productId": pid, "score": round(float(score), 6)})
            if len(unique_results) >= limit:
                break

        return unique_results
