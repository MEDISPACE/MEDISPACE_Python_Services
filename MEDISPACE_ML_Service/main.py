"""
MEDISPACE ML Service - Main Entry Point
Port: 8002
"""
import os
import asyncio
import httpx
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.models.tfidf_model import TFIDFRecommender
from src.models.fpgrowth_model import FPGrowthRecommender
from src.models.nmf_trending import NMFTrendingRecommender
from src.models.svd_model import SVDRecommender
from src.models.hybrid_engine import HybridEngine
from src.cache.mongo_cache import MongoCache

# ─── Global instances ────────────────────────────────────────────
tfidf_model = TFIDFRecommender()
fpgrowth_model = FPGrowthRecommender()
nmf_model = NMFTrendingRecommender()
svd_model = SVDRecommender()
hybrid_engine = HybridEngine(tfidf_model, fpgrowth_model, nmf_model, svd_model)
cache = MongoCache()
scheduler = AsyncIOScheduler()

BE_SERVICE_URL = os.getenv("BE_SERVICE_URL", "")
# Support nhiều URL (cách nhau bằng dấu phẩy) — notify tất cả sau retrain
BE_SERVICE_URLS = [
    url.strip()
    for url in os.getenv("BE_SERVICE_URLS", BE_SERVICE_URL).split(",")
    if url.strip()
]

# ─── Retrain helper ───────────────────────────────────────────────
async def _retrain_and_notify():
    """Train tất cả models, sau đó notify tất cả BE instances để flush Redis cache."""
    await hybrid_engine.train_all()
    # Invalidate toàn bộ ML cache sau khi retrain
    await cache.invalidate_pattern("")
    print("[ML Service] ML cache invalidated after retrain.")

    # Notify tất cả BE URLs song song (fire-and-forget, non-blocking)
    if not BE_SERVICE_URLS:
        return

    async def _notify(url: str):
        endpoint = f"{url.rstrip('/')}/internal/flush-recommendation-cache"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(endpoint)
            print(f"[ML Service] Notified {url} to flush cache. ✓")
        except Exception as e:
            print(f"[ML Service] Could not notify {url} (non-critical): {e}")

    await asyncio.gather(*[_notify(url) for url in BE_SERVICE_URLS])


# ─── Startup / Shutdown ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[ML Service] Starting up...")

    # Connect cache
    await cache.connect()

    # Initial training
    print("[ML Service] Running initial model training...")
    await hybrid_engine.train_all()
    print("[ML Service] All models trained and ready!")

    # Schedule periodic retraining (6h)
    scheduler.add_job(
        _retrain_and_notify,
        'interval',
        hours=int(os.getenv("ML_RETRAIN_INTERVAL_HOURS", "6")),
        id='retrain_all'
    )
    scheduler.start()
    print(f"[ML Service] Scheduler started (retrain every {os.getenv('ML_RETRAIN_INTERVAL_HOURS', '6')}h)")

    yield

    scheduler.shutdown()
    await cache.disconnect()
    print("[ML Service] Shutdown complete.")


app = FastAPI(
    title="MEDISPACE ML Recommendation Service",
    description="Hệ thống gợi ý sản phẩm dựa trên Machine Learning",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ───────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "MEDISPACE ML Recommendation Service",
        "status": "running",
        "version": "1.1.0",
        "models": {
            "tfidf": tfidf_model.is_trained,
            "fpgrowth": fpgrowth_model.is_trained,
            "nmf_trending": nmf_model.is_trained,
            "svd": svd_model.is_trained,
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}


# ─── Manual retrain (admin) ───────────────────────────────────────
@app.post("/train")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Trigger retraining thủ công (admin).
    Chạy trong background để không block response.
    """
    background_tasks.add_task(_retrain_and_notify)
    return {"message": "Retraining started in background", "models": {
        "tfidf": tfidf_model.is_trained,
        "fpgrowth": fpgrowth_model.is_trained,
        "nmf_trending": nmf_model.is_trained,
        "svd": svd_model.is_trained,
    }}


# ─── Recommendation Endpoints ────────────────────────────────────

@app.get("/recommend/related/{product_id}")
async def get_related(product_id: str, limit: int = 8, diverse: bool = True, lambda_mmr: float = 0.7):
    """
    TF-IDF Content-Based: Sản phẩm liên quan.
    diverse=True  → MMR (Maximal Marginal Relevance) — giảm filter bubble
    diverse=False → Pure relevance ranking
    """
    cached = await cache.get(f"related_{product_id}_{diverse}")
    if cached:
        return {"source": "cache", "algorithm": "tfidf_mmr" if diverse else "tfidf", "products": cached}

    if diverse:
        results = await tfidf_model.get_related_diverse(product_id, limit=limit, lambda_mmr=lambda_mmr)
        algo = "tfidf_mmr"
    else:
        results = await tfidf_model.get_related(product_id, limit=limit)
        algo = "tfidf"

    await cache.set(f"related_{product_id}_{diverse}", results, ttl_hours=24)
    return {"source": "computed", "algorithm": algo, "products": results}


@app.get("/recommend/bought-together/{product_id}")
async def get_bought_together(product_id: str, limit: int = 6):
    """FP-Growth: Thường mua kèm"""
    cached = await cache.get(f"fbt_{product_id}")
    if cached:
        return {"source": "cache", "algorithm": "fpgrowth", "products": cached}

    results = await fpgrowth_model.get_associated(product_id, limit)
    if not results:
        # Fallback to TF-IDF MMR if no FP-Growth rules found
        results = await tfidf_model.get_related_diverse(product_id, limit=limit, lambda_mmr=0.6)
        await cache.set(f"fbt_{product_id}", results, ttl_hours=6)
        return {"source": "computed", "algorithm": "tfidf_mmr_fallback", "products": results}

    await cache.set(f"fbt_{product_id}", results, ttl_hours=6)
    return {"source": "computed", "algorithm": "fpgrowth", "products": results}


@app.get("/recommend/trending")
async def get_trending(category_id: str = None, limit: int = 12):
    """NMF: Xu hướng & bán chạy"""
    key = f"trending_{category_id or 'all'}"
    cached = await cache.get(key)
    if cached:
        return {"source": "cache", "algorithm": "nmf", "products": cached}

    results = await nmf_model.get_trending(category_id, limit)
    await cache.set(key, results, ttl_hours=2)
    return {"source": "computed", "algorithm": "nmf", "products": results}


@app.get("/recommend/for-you/{user_id}")
async def get_for_you(user_id: str, limit: int = 12):
    """SVD or NMF fallback: Dành cho bạn"""
    cached = await cache.get(f"fyt_{user_id}")
    if cached:
        return {"source": "cache", "algorithm": cached.get("algorithm"), "products": cached.get("products")}

    results, algorithm = await hybrid_engine.get_personalized(user_id, limit)
    await cache.set(f"fyt_{user_id}", {"algorithm": algorithm, "products": results}, ttl_hours=3)
    return {"source": "computed", "algorithm": algorithm, "products": results}


@app.get("/recommend/post-purchase")
async def get_post_purchase(order_ids: str, limit: int = 8):
    """Hybrid: Gợi ý sau khi đặt hàng (FP-Growth + TF-IDF MMR)"""
    ids = [i for i in order_ids.split(",") if i.strip()]
    results = await hybrid_engine.get_post_purchase(ids, limit)
    return {"algorithm": "hybrid", "products": results}


@app.get("/recommend/pharmacist")
async def get_pharmacist_suggestions(
    chronic_diseases: str = "",
    allergies: str = "",
    current_medications: str = "",
    prescription_product_ids: str = "",
    limit: int = 10
):
    """TF-IDF Medical Context: Gợi ý cho pharmacist — với chronic disease boost & allergy filter"""
    results = await hybrid_engine.get_pharmacist_suggestions(
        chronic_diseases=chronic_diseases.split(",") if chronic_diseases else [],
        allergies=allergies.split(",") if allergies else [],
        current_medications=current_medications.split(",") if current_medications else [],
        prescription_product_ids=prescription_product_ids.split(",") if prescription_product_ids else [],
        limit=limit
    )
    return {"algorithm": "tfidf_medical", "products": results}


@app.get("/recommend/replenishment/{user_id}")
async def get_replenishment(user_id: str, limit: int = 5):
    """
    Predictive Replenishment: Gợi ý sản phẩm cần mua lại.
    Phân tích chu kỳ mua hàng của user, tìm sản phẩm đến hạn reorder.
    Đặc biệt hữu ích cho thuốc uống thường xuyên, vitamin, mỹ phẩm.
    """
    cached = await cache.get(f"replenish_{user_id}")
    if cached:
        return {"source": "cache", "algorithm": "replenishment", "products": cached}

    results = await hybrid_engine.get_replenishment(user_id, limit)
    # Cache ngắn hơn (1h) vì phụ thuộc vào ngày hiện tại
    await cache.set(f"replenish_{user_id}", results, ttl_hours=1)
    return {"source": "computed", "algorithm": "replenishment", "products": results}
