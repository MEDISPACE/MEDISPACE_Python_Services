"""
mongo_loader.py - Fetch và transform data từ MongoDB cho ML training
"""
import os
from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Optional


class MongoLoader:
    """
    Quản lý kết nối và load dữ liệu từ MongoDB.

    Có 2 chế độ sử dụng:
      1. Batch Training: connect() → load data → disconnect() (trong train_all)
      2. Runtime Query: connect_persistent() → queries → (không disconnect ngay)

    get_user_top_categories() yêu cầu connection đang hoạt động.
    HybridEngine phải gọi runtime_loader (xem RuntimeMongoLoader) thay vì dùng
    mongo_loader singleton đã bị disconnect sau training.
    """

    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db = None

    def connect(self):
        uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("MONGODB_DB_NAME", "medispace")
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        print(f"[MongoLoader] Connected to MongoDB: {db_name}")

    def disconnect(self):
        if self.client:
            self.client.close()
            self.client = None
            self.db = None

    @property
    def is_connected(self) -> bool:
        return self.db is not None

    def load_products(self) -> List[Dict]:
        """Load tất cả active products kèm chi tiết (join categories, brands, details)"""
        pipeline = [
            {"$match": {"isActive": True}},
            {
                "$lookup": {
                    "from": "productDetails",
                    "localField": "_id",
                    "foreignField": "productId",
                    "as": "details"
                }
            },
            {
                "$lookup": {
                    "from": "categories",
                    "localField": "categoryId",
                    "foreignField": "_id",
                    "as": "category"
                }
            },
            {
                "$lookup": {
                    "from": "brands",
                    "localField": "brandId",
                    "foreignField": "_id",
                    "as": "brand"
                }
            },
            {
                "$addFields": {
                    "details": {"$arrayElemAt": ["$details", 0]},
                    "categoryName": {"$arrayElemAt": ["$category.name", 0]},
                    "brandName": {"$arrayElemAt": ["$brand.name", 0]},
                }
            },
        ]
        products = list(self.db["products"].aggregate(pipeline))
        print(f"[MongoLoader] Loaded {len(products)} active products")
        return products

    def load_orders(self, days: int = 365) -> List[Dict]:
        """Load orders trong N ngày gần nhất (loại bỏ cancelled)"""
        since = datetime.now() - timedelta(days=days)
        orders = list(self.db["orders"].find(
            {
                "createdAt": {"$gte": since},
                "orderStatus": {"$nin": ["cancelled"]}
            },
            {"_id": 1, "userId": 1, "items": 1, "createdAt": 1, "totalAmount": 1}
        ))
        print(f"[MongoLoader] Loaded {len(orders)} orders (last {days} days)")
        return orders

    def load_reviews(self) -> List[Dict]:
        """Load approved reviews"""
        reviews = list(self.db["reviews"].find(
            {"status": "approved"},
            {"_id": 1, "userId": 1, "productId": 1, "rating": 1, "createdAt": 1}
        ))
        print(f"[MongoLoader] Loaded {len(reviews)} approved reviews")
        return reviews

    def load_carts(self) -> List[Dict]:
        """Load active cart items (implicit signal)"""
        carts = list(self.db["carts"].find(
            {"items": {"$exists": True, "$ne": []}},
            {"_id": 1, "userId": 1, "items": 1}
        ))
        print(f"[MongoLoader] Loaded {len(carts)} active carts")
        return carts

    def load_wishlists(self) -> List[Dict]:
        """
        Load user wishlists — strong implicit interest signal.
        Data đã tồn tại trong User.wishlist (ObjectId[]).
        """
        users = list(self.db["users"].find(
            {"wishlist": {"$exists": True, "$not": {"$size": 0}}},
            {"_id": 1, "wishlist": 1}
        ))
        print(f"[MongoLoader] Loaded {len(users)} users with wishlists")
        return users

    def build_interaction_matrix(self) -> pd.DataFrame:
        """
        Xây dựng User-Item interaction matrix cho SVD/NMF.

        Signal weights (cao hơn = stronger intent):
          - Purchase : 5.0 × recency_decay  (confirmed intent)
          - Review   : 0–4.0 × recency_decay (explicit feedback)
          - Wishlist : 2.0                   (explicit save, no decay — still relevant)
          - Cart     : 1.0                   (weak implicit — user may abandon)
        """
        interactions: Dict[tuple, float] = {}
        now = datetime.now()

        # ── 1. Purchase (weight = 5.0 × recency) ────────────────────────────
        orders = self.load_orders()
        for order in orders:
            user_id = str(order.get("userId", ""))
            if not user_id or user_id == "None":
                continue
            days_ago = (now - order.get("createdAt", now)).days
            recency_factor = np.exp(-0.005 * days_ago)

            for item in order.get("items", []):
                product_id = str(item.get("productId", ""))
                if not product_id:
                    continue
                key = (user_id, product_id)
                interactions[key] = max(interactions.get(key, 0), 5.0 * recency_factor)

        # ── 2. Review (weight = rating/5 × 4 × recency) ─────────────────────
        reviews = self.load_reviews()
        for review in reviews:
            user_id = str(review.get("userId", ""))
            product_id = str(review.get("productId", ""))
            if not user_id or not product_id:
                continue
            rating = review.get("rating", 3)
            days_ago = (now - review.get("createdAt", now)).days
            recency_factor = np.exp(-0.005 * days_ago)
            weight = (rating / 5.0) * 4.0 * recency_factor
            key = (user_id, product_id)
            interactions[key] = max(interactions.get(key, 0), weight)

        # ── 3. Wishlist (weight = 2.0) ────────────────────────────────────────
        wishlists = self.load_wishlists()
        for user in wishlists:
            user_id = str(user.get("_id", ""))
            if not user_id:
                continue
            for product_id in user.get("wishlist", []):
                key = (user_id, str(product_id))
                # Chỉ set nếu chưa có signal mạnh hơn (purchase/review > 2.0)
                if interactions.get(key, 0) < 2.0:
                    interactions[key] = 2.0

        # ── 4. Cart (weight = 1.0) ────────────────────────────────────────────
        carts = self.load_carts()
        for cart in carts:
            user_id = str(cart.get("userId", ""))
            if not user_id:
                continue
            for item in cart.get("items", []):
                product_id = str(item.get("productId", ""))
                if not product_id:
                    continue
                key = (user_id, product_id)
                if key not in interactions:
                    interactions[key] = 1.0

        # ── Build DataFrame ───────────────────────────────────────────────────
        if not interactions:
            return pd.DataFrame()

        rows = [(uid, pid, score) for (uid, pid), score in interactions.items()]
        df = pd.DataFrame(rows, columns=["user_id", "product_id", "score"])
        print(
            f"[MongoLoader] Interaction matrix: "
            f"{df['user_id'].nunique()} users × {df['product_id'].nunique()} products"
        )
        return df

    def build_transaction_baskets(self) -> List[List[str]]:
        """
        Xây dựng transaction baskets cho FP-Growth.
        Mỗi order = 1 basket (list productIds). Bỏ qua đơn chỉ có 1 sản phẩm.
        """
        orders = self.load_orders()
        baskets = []
        for order in orders:
            items = order.get("items", [])
            basket = [str(item["productId"]) for item in items if item.get("productId")]
            if len(basket) >= 2:
                baskets.append(basket)
        print(f"[MongoLoader] Built {len(baskets)} transaction baskets for FP-Growth")
        return baskets

    def get_user_order_count(self) -> Dict[str, int]:
        """Đếm số orders per user (để quyết định SVD vs fallback)"""
        pipeline = [
            {"$match": {"orderStatus": {"$nin": ["cancelled"]}}},
            {"$group": {"_id": "$userId", "count": {"$sum": 1}}}
        ]
        result = list(self.db["orders"].aggregate(pipeline))
        return {str(r["_id"]): r["count"] for r in result}

    def get_user_top_categories(self, user_id: str, top_n: int = 3) -> List[str]:
        """
        Lấy top N categories user mua nhiều nhất.

        ⚠️  Yêu cầu self.db đang connected.
        Phải được gọi qua RuntimeMongoLoader (không phải sau disconnect).
        """
        from bson import ObjectId

        if not self.is_connected:
            print("[MongoLoader] get_user_top_categories: not connected, returning []")
            return []

        try:
            uid = ObjectId(user_id)
        except Exception:
            print(f"[MongoLoader] Invalid user_id for ObjectId: '{user_id}'")
            return []

        pipeline = [
            {"$match": {"userId": uid}},
            {"$unwind": "$items"},
            {
                "$lookup": {
                    "from": "products",
                    "localField": "items.productId",
                    "foreignField": "_id",
                    "as": "product"
                }
            },
            {"$unwind": "$product"},
            {"$group": {"_id": "$product.categoryId", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": top_n}
        ]
        try:
            result = list(self.db["orders"].aggregate(pipeline))
            return [str(r["_id"]) for r in result]
        except Exception as e:
            print(f"[MongoLoader] get_user_top_categories error: {e}")
            return []


# ── Runtime loader: kết nối persistent cho query tại request time ─────────────
class RuntimeMongoLoader:
    """
    Kết nối MongoDB persistent cho runtime queries (personalization, etc.).
    Khác với mongo_loader, KHÔNG disconnect sau mỗi lần dùng.
    Khởi tạo lazy khi lần đầu được gọi.
    """

    def __init__(self):
        self._loader: Optional[MongoLoader] = None

    def _ensure_connected(self):
        if self._loader is None or not self._loader.is_connected:
            self._loader = MongoLoader()
            self._loader.connect()

    def get_user_top_categories(self, user_id: str, top_n: int = 3) -> List[str]:
        self._ensure_connected()
        return self._loader.get_user_top_categories(user_id, top_n)


# Singleton runtime loader — dùng trong HybridEngine.get_personalized()
runtime_loader = RuntimeMongoLoader()

# Singleton instance
mongo_loader = MongoLoader()
