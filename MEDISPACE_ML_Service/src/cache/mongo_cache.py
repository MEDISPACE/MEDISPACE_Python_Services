"""
mongo_cache.py - TTL Cache vao MongoDB cho recommendation results
"""
import os
from datetime import datetime, timedelta
from typing import Any, Optional
from pymongo import MongoClient
from pymongo.collection import Collection


class MongoCache:
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.collection: Optional[Collection] = None

    async def connect(self):
        uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("MONGODB_DB_NAME", "medispace")
        self.client = MongoClient(uri)
        db = self.client[db_name]
        self.collection = db["recommendation_cache"]

        # Tao TTL index (tu dong xoa sau khi het han)
        self.collection.create_index("expiresAt", expireAfterSeconds=0)
        self.collection.create_index("key", unique=True)
        print(f"[MongoCache] Connected. TTL index ready.")

    async def disconnect(self):
        if self.client:
            self.client.close()

    async def get(self, key: str) -> Optional[Any]:
        """Lay cache theo key. Tra ve None neu khong co hoac het han."""
        if self.collection is None:
            return None
        try:
            doc = self.collection.find_one({"key": key})
            if doc and doc.get("expiresAt", datetime.min) > datetime.utcnow():
                return doc.get("data")
        except Exception as e:
            print(f"[MongoCache] Get error: {e}")
        return None

    async def set(self, key: str, data: Any, ttl_hours: int = 6) -> bool:
        """Luu cache voi TTL."""
        if self.collection is None:
            return False
        try:
            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
            self.collection.update_one(
                {"key": key},
                {"$set": {
                    "key": key,
                    "data": data,
                    "expiresAt": expires_at,
                    "updatedAt": datetime.utcnow()
                }},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"[MongoCache] Set error: {e}")
            return False

    async def invalidate(self, key: str) -> bool:
        """Xoa 1 key cu the (vi du khi user dat hang moi)."""
        if self.collection is None:
            return False
        try:
            self.collection.delete_one({"key": key})
            return True
        except Exception as e:
            print(f"[MongoCache] Invalidate error: {e}")
            return False

    async def invalidate_pattern(self, prefix: str) -> int:
        """Xoa tat ca cache co key bat dau bang prefix."""
        if not self.collection:
            return 0
        try:
            import re
            result = self.collection.delete_many({"key": {"$regex": f"^{re.escape(prefix)}"}})
            return result.deleted_count
        except Exception as e:
            print(f"[MongoCache] Invalidate pattern error: {e}")
            return 0
