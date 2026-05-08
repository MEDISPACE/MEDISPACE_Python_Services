"""
mongo_loader.py - Fetch và transform data từ MongoDB cho ML training
"""
import os
from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any


class MongoLoader:
    def __init__(self):
        self.client = None
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

    def load_products(self) -> List[Dict]:
        """Load tất cả products với details (join)"""
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
        """Load orders trong N ngày gần nhất"""
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

    def build_interaction_matrix(self) -> pd.DataFrame:
        """
        Xây dựng User-Item interaction matrix cho SVD/NMF.
        Weights: purchase=5, review(normalized)=0-4, cart=1
        Với recency decay: recent interactions có weight cao hơn
        """
        interactions = {}

        # 1. Purchase interactions (weight = 5, highest)
        orders = self.load_orders()
        now = datetime.now()
        for order in orders:
            user_id = str(order.get("userId", ""))
            if not user_id:
                continue
            days_ago = (now - order.get("createdAt", now)).days
            recency_factor = np.exp(-0.005 * days_ago)  # decay

            for item in order.get("items", []):
                product_id = str(item.get("productId", ""))
                if not product_id:
                    continue
                key = (user_id, product_id)
                current = interactions.get(key, 0)
                interactions[key] = max(current, 5.0 * recency_factor)

        # 2. Review interactions (weight = rating/5 * 4)
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

        # 3. Cart interactions (weight = 1)
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

        # Build DataFrame
        if not interactions:
            return pd.DataFrame()

        rows = [(uid, pid, score) for (uid, pid), score in interactions.items()]
        df = pd.DataFrame(rows, columns=["user_id", "product_id", "score"])
        print(f"[MongoLoader] Interaction matrix: {df['user_id'].nunique()} users × {df['product_id'].nunique()} products")
        return df

    def build_transaction_baskets(self) -> List[List[str]]:
        """
        Xây dựng transaction baskets cho FP-Growth.
        Mỗi order = 1 basket, chứa list productIds
        """
        orders = self.load_orders()
        baskets = []
        for order in orders:
            items = order.get("items", [])
            if len(items) < 2:
                continue  # FP-Growth cần ít nhất 2 items
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
        """Lấy top categories user hay mua nhất"""
        pipeline = [
            {"$match": {"userId": {"$oid": user_id} if len(user_id) == 24 else user_id}},
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
            from bson import ObjectId
            pipeline[0]["$match"]["userId"] = ObjectId(user_id)
            result = list(self.db["orders"].aggregate(pipeline))
            return [str(r["_id"]) for r in result]
        except Exception:
            return []


# Singleton instance
mongo_loader = MongoLoader()
