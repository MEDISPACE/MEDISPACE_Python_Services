"""
MEDISPACE ML Service - Main Entry Point
Port: 8002
"""
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
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

    # Schedule periodic retraining
    scheduler.add_job(hybrid_engine.train_all, 'interval', hours=6, id='retrain_all')
    scheduler.start()
    print("[ML Service] Scheduler started (retrain every 6h)")

    yield

    scheduler.shutdown()
    await cache.disconnect()
    print("[ML Service] Shutdown complete.")


app = FastAPI(
    title="MEDISPACE ML Recommendation Service",
    description="Hệ thống gợi ý sản phẩm dựa trên Machine Learning",
    version="1.0.0",
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
        "version": "1.0.0",
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
async def trigger_retrain():
    await hybrid_engine.train_all()
    return {"message": "Retraining completed", "models": {
        "tfidf": tfidf_model.is_trained,
        "fpgrowth": fpgrowth_model.is_trained,
        "nmf_trending": nmf_model.is_trained,
        "svd": svd_model.is_trained,
    }}


# ─── Recommendation Endpoints ────────────────────────────────────

@app.get("/recommend/related/{product_id}")
async def get_related(product_id: str, limit: int = 8):
    """TF-IDF Content-Based: Sản phẩm liên quan"""
    cached = await cache.get(f"related_{product_id}")
    if cached:
        return {"source": "cache", "algorithm": "tfidf", "products": cached}
    
    results = await tfidf_model.get_related(product_id, limit)
    await cache.set(f"related_{product_id}", results, ttl_hours=24)
    return {"source": "computed", "algorithm": "tfidf", "products": results}


@app.get("/recommend/bought-together/{product_id}")
async def get_bought_together(product_id: str, limit: int = 6):
    """FP-Growth: Thường mua kèm"""
    cached = await cache.get(f"fbt_{product_id}")
    if cached:
        return {"source": "cache", "algorithm": "fpgrowth", "products": cached}
    
    results = await fpgrowth_model.get_associated(product_id, limit)
    if not results:
        # Fallback to TF-IDF if no FP-Growth rules found
        results = await tfidf_model.get_related(product_id, limit)
        await cache.set(f"fbt_{product_id}", results, ttl_hours=6)
        return {"source": "computed", "algorithm": "tfidf_fallback", "products": results}
    
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
    """Hybrid: Gợi ý sau khi đặt hàng"""
    ids = order_ids.split(",")
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
    """TF-IDF Medical Context: Gợi ý cho pharmacist"""
    results = await hybrid_engine.get_pharmacist_suggestions(
        chronic_diseases=chronic_diseases.split(",") if chronic_diseases else [],
        allergies=allergies.split(",") if allergies else [],
        current_medications=current_medications.split(",") if current_medications else [],
        prescription_product_ids=prescription_product_ids.split(",") if prescription_product_ids else [],
        limit=limit
    )
    return {"algorithm": "tfidf_medical", "products": results}
