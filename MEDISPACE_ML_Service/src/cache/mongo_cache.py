"""
mongo_cache.py - TTL Cache vào MongoDB cho recommendation results.

Fix: Các phương thức async trước đây dùng pymongo đồng bộ, gây block event loop
của FastAPI. Đã chuyển sang asyncio.to_thread() để wrap sync I/O.
"""
import os
import asyncio
import re as _re
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

        def _connect():
            client = MongoClient(uri)
            db = client[db_name]
            coll = db["recommendation_cache"]
            # TTL index: MongoDB tự xoá document khi expiresAt đã qua
            coll.create_index("expiresAt", expireAfterSeconds=0)
            coll.create_index("key", unique=True)
            return client, coll

        self.client, self.collection = await asyncio.to_thread(_connect)
        print("[MongoCache] Connected. TTL index ready.")

    async def disconnect(self):
        if self.client:
            await asyncio.to_thread(self.client.close)
            self.client = None
            self.collection = None

    async def get(self, key: str) -> Optional[Any]:
        """Lấy cache theo key. Trả về None nếu không có hoặc hết hạn."""
        if self.collection is None:
            return None

        def _get():
            doc = self.collection.find_one({"key": key})
            if doc and doc.get("expiresAt", datetime.min) > datetime.utcnow():
                return doc.get("data")
            return None

        try:
            return await asyncio.to_thread(_get)
        except Exception as e:
            print(f"[MongoCache] Get error: {e}")
            return None

    async def set(self, key: str, data: Any, ttl_hours: int = 6) -> bool:
        """Lưu cache với TTL (giờ)."""
        if self.collection is None:
            return False

        expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)

        def _set():
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

        try:
            await asyncio.to_thread(_set)
            return True
        except Exception as e:
            print(f"[MongoCache] Set error: {e}")
            return False

    async def invalidate(self, key: str) -> bool:
        """Xoá 1 key cụ thể (ví dụ khi user đặt hàng mới → xoá for-you cache)."""
        if self.collection is None:
            return False

        def _del():
            self.collection.delete_one({"key": key})

        try:
            await asyncio.to_thread(_del)
            return True
        except Exception as e:
            print(f"[MongoCache] Invalidate error: {e}")
            return False

    async def invalidate_pattern(self, prefix: str) -> int:
        """Xoá tất cả cache có key bắt đầu bằng prefix."""
        if self.collection is None:
            return 0

        def _del_many():
            result = self.collection.delete_many(
                {"key": {"$regex": f"^{_re.escape(prefix)}"}}
            )
            return result.deleted_count

        try:
            return await asyncio.to_thread(_del_many)
        except Exception as e:
            print(f"[MongoCache] Invalidate pattern error: {e}")
            return 0

    async def invalidate_user(self, user_id: str) -> int:
        """Invalidate personalized and replenishment cache entries for one user."""
        deleted = 0
        for prefix in (f"fyt_{user_id}_", f"replenish_{user_id}_"):
            deleted += await self.invalidate_pattern(prefix)
        return deleted
