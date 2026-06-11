"""
MEDISPACE ML Service - Main Entry Point
Port: 8002
"""
import os
import asyncio
import httpx
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, BackgroundTasks, Depends, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel, Field

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
ML_SERVICE_TOKEN = os.getenv("ML_SERVICE_TOKEN", "")
if not ML_SERVICE_TOKEN and os.getenv("ENVIRONMENT", "development") != "production":
    ML_SERVICE_TOKEN = "medispace-local-ml-token"
retrain_lock = asyncio.Lock()
serving_metrics = {"requests": 0, "errors": 0, "latency_ms_total": 0.0}

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
    if retrain_lock.locked():
        print("[ML Service] Retrain already in progress, skipping duplicate trigger.")
        return

    async with retrain_lock:
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
                await client.post(endpoint, headers={"x-service-token": ML_SERVICE_TOKEN})
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
    allow_origins=[origin.strip() for origin in os.getenv("ML_CORS_ORIGINS", "").split(",") if origin.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["content-type", "x-service-token"],
)

@app.middleware("http")
async def observe_serving(request, call_next):
    started = time.perf_counter()
    serving_metrics["requests"] += 1
    try:
        response_value = await call_next(request)
        if response_value.status_code >= 500:
            serving_metrics["errors"] += 1
        return response_value
    except Exception:
        serving_metrics["errors"] += 1
        raise
    finally:
        serving_metrics["latency_ms_total"] += (time.perf_counter() - started) * 1000

async def require_service_token(x_service_token: str = Header(default="")):
    if not ML_SERVICE_TOKEN or x_service_token != ML_SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized service request")

class PharmacistRecommendationRequest(BaseModel):
    chronic_diseases: list[str] = Field(default_factory=list, max_length=50)
    allergies: list[str] = Field(default_factory=list, max_length=50)
    current_medications: list[str] = Field(default_factory=list, max_length=50)
    prescription_product_ids: list[str] = Field(default_factory=list, max_length=50)
    limit: int = Field(default=10, ge=1, le=45)

class PostPurchaseRequest(BaseModel):
    product_ids: list[str] = Field(default_factory=list, max_length=50)
    limit: int = Field(default=8, ge=1, le=60)

class InvalidateUserRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)

def recommendation_items(product_ids: list[str], reason: str, evidence: list[str] | None = None):
    total = max(len(product_ids), 1)
    return [
        {
            "productId": product_id,
            "score": round(max(0.0, 1.0 - index / total), 6),
            "reason": reason,
            "evidence": evidence or [],
        }
        for index, product_id in enumerate(product_ids)
    ]

def response(algorithm: str, product_ids: list[str], reason: str, source: str = "computed", evidence: list[str] | None = None):
    return {
        "source": source,
        "algorithm": algorithm,
        "model_version": hybrid_engine.model_version,
        "products": recommendation_items(product_ids, reason, evidence),
    }


# ─── Health ───────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "MEDISPACE ML Recommendation Service",
        "status": "running",
        "version": "1.1.0",
        "models": {
            "tfidf": hybrid_engine.tfidf.is_trained,
            "fpgrowth": hybrid_engine.fpgrowth.is_trained,
            "nmf_trending": hybrid_engine.nmf.is_trained,
            "svd": hybrid_engine.svd.is_trained,
        },
        **hybrid_engine.metrics(),
    }

@app.get("/health")
async def health():
    if not hybrid_engine.is_ready:
        raise HTTPException(status_code=503, detail="Models are not ready")
    return {"status": "healthy", "model_version": hybrid_engine.model_version}

@app.get("/metrics", dependencies=[Depends(require_service_token)])
async def metrics():
    requests = serving_metrics["requests"]
    return {
        **hybrid_engine.metrics(),
        "serving": {
            **serving_metrics,
            "average_latency_ms": serving_metrics["latency_ms_total"] / requests if requests else 0,
            "error_rate": serving_metrics["errors"] / requests if requests else 0,
        },
        "safety": {
            "automatic_prescription_recommendations_enabled": False,
            "interaction_database_configured": False,
        },
    }

@app.post("/events/invalidate-user", dependencies=[Depends(require_service_token)])
async def invalidate_user(payload: InvalidateUserRequest):
    deleted = await cache.invalidate_user(payload.user_id)
    return {"invalidated": deleted, "user_id": payload.user_id}


# ─── Manual retrain (admin) ───────────────────────────────────────
@app.post("/train", dependencies=[Depends(require_service_token)])
async def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Trigger retraining thủ công (admin).
    Chạy trong background để không block response.
    """
    background_tasks.add_task(_retrain_and_notify)
    return {"message": "Retraining started in background", "models": {
        "tfidf": hybrid_engine.tfidf.is_trained,
        "fpgrowth": hybrid_engine.fpgrowth.is_trained,
        "nmf_trending": hybrid_engine.nmf.is_trained,
        "svd": hybrid_engine.svd.is_trained,
    }}


# ─── Recommendation Endpoints ────────────────────────────────────

@app.get("/recommend/related/{product_id}", dependencies=[Depends(require_service_token)])
async def get_related(
    product_id: str,
    limit: int = Query(default=8, ge=1, le=60),
    diverse: bool = True,
    lambda_mmr: float = Query(default=0.7, ge=0.0, le=1.0)
):
    """
    TF-IDF Content-Based: Sản phẩm liên quan.
    diverse=True  → MMR (Maximal Marginal Relevance) — giảm filter bubble
    diverse=False → Pure relevance ranking
    """
    cache_key = f"related_{product_id}_{limit}_{diverse}_{lambda_mmr}"
    cached = await cache.get(cache_key)
    if cached:
        return response("tfidf_mmr" if diverse else "tfidf", cached, "Tương đồng nội dung sản phẩm", "cache", ["catalog_content"])

    if diverse:
        results = await hybrid_engine.tfidf.get_related_diverse(product_id, limit=limit, lambda_mmr=lambda_mmr)
        algo = "tfidf_mmr"
    else:
        results = await hybrid_engine.tfidf.get_related(product_id, limit=limit)
        algo = "tfidf"

    await cache.set(cache_key, results, ttl_hours=24)
    return response(algo, results, "Tương đồng nội dung và mức độ đa dạng", evidence=["catalog_content", "mmr_diversity"])


@app.get("/recommend/bought-together/{product_id}", dependencies=[Depends(require_service_token)])
async def get_bought_together(product_id: str, limit: int = Query(default=6, ge=1, le=60)):
    """FP-Growth: Thường mua kèm"""
    cache_key = f"fbt_{product_id}_{limit}"
    cached = await cache.get(cache_key)
    if cached:
        return response("fpgrowth", cached, "Thường xuất hiện cùng nhau trong đơn đã giao", "cache", ["delivered_orders"])

    results = await hybrid_engine.fpgrowth.get_associated(product_id, limit)
    if not results:
        # Fallback to TF-IDF MMR if no FP-Growth rules found
        results = await hybrid_engine.tfidf.get_related_diverse(product_id, limit=limit, lambda_mmr=0.6)
        await cache.set(cache_key, results, ttl_hours=6)
        return response("tfidf_mmr_fallback", results, "Sản phẩm liên quan dùng khi chưa đủ lịch sử mua kèm", evidence=["catalog_content"])

    await cache.set(cache_key, results, ttl_hours=6)
    return response("fpgrowth", results, "Thường xuất hiện cùng nhau trong đơn đã giao", evidence=["delivered_orders"])


@app.get("/recommend/trending", dependencies=[Depends(require_service_token)])
async def get_trending(category_id: str = None, limit: int = Query(default=12, ge=1, le=60)):
    """NMF: Xu hướng & bán chạy"""
    key = f"trending_{category_id or 'all'}_{limit}"
    cached = await cache.get(key)
    if cached:
        return response("nmf", cached, "Sản phẩm nổi bật dựa trên tương tác và đánh giá", "cache", ["interactions", "ratings"])

    results = await hybrid_engine.nmf.get_trending(category_id, limit)
    await cache.set(key, results, ttl_hours=2)
    return response("nmf", results, "Sản phẩm nổi bật dựa trên tương tác và đánh giá", evidence=["interactions", "ratings"])

@app.get("/recommend/popular", dependencies=[Depends(require_service_token)])
async def get_popular(category_id: str = None, limit: int = Query(default=12, ge=1, le=60)):
    results = await hybrid_engine.nmf.get_trending(category_id, limit)
    return response("popular", results, "Phổ biến trong dữ liệu giao dịch và tương tác", evidence=["delivered_orders", "interactions"])


@app.get("/recommend/for-you/{user_id}", dependencies=[Depends(require_service_token)])
async def get_for_you(user_id: str, limit: int = Query(default=12, ge=1, le=60)):
    """SVD or NMF fallback: Dành cho bạn"""
    cache_key = f"fyt_{user_id}_{limit}"
    cached = await cache.get(cache_key)
    if cached:
        return response(cached.get("algorithm"), cached.get("products"), "Phù hợp với lịch sử và sở thích gần đây", "cache", ["user_interactions"])

    results, algorithm = await hybrid_engine.get_personalized(user_id, limit)
    await cache.set(cache_key, {"algorithm": algorithm, "products": results}, ttl_hours=3)
    return response(algorithm, results, "Phù hợp với lịch sử và sở thích gần đây", evidence=["user_interactions"])


@app.post("/recommend/post-purchase", dependencies=[Depends(require_service_token)])
async def get_post_purchase(payload: PostPurchaseRequest):
    """Hybrid: Gợi ý sau khi đặt hàng (FP-Growth + TF-IDF MMR)"""
    results = await hybrid_engine.get_post_purchase(payload.product_ids, payload.limit)
    return response("hybrid", results, "Bổ trợ cho các sản phẩm vừa mua", evidence=["delivered_orders", "catalog_content"])


@app.post("/recommend/pharmacist", dependencies=[Depends(require_service_token)])
async def get_pharmacist_suggestions(payload: PharmacistRecommendationRequest):
    """TF-IDF Medical Context: Gợi ý cho pharmacist — với chronic disease boost & allergy filter"""
    results = await hybrid_engine.get_pharmacist_suggestions(
        chronic_diseases=payload.chronic_diseases,
        allergies=payload.allergies,
        current_medications=payload.current_medications,
        prescription_product_ids=payload.prescription_product_ids,
        limit=payload.limit
    )
    return response("tfidf_medical", results, "Gợi ý OTC tham khảo theo ngữ cảnh; cần dược sĩ rà soát độc lập", evidence=["catalog_content", "patient_context"])


@app.get("/recommend/replenishment/{user_id}", dependencies=[Depends(require_service_token)])
async def get_replenishment(user_id: str, limit: int = Query(default=5, ge=1, le=24)):
    """
    Predictive Replenishment: Gợi ý sản phẩm cần mua lại.
    Phân tích chu kỳ mua hàng của user, tìm sản phẩm đến hạn reorder.
    Đặc biệt hữu ích cho thuốc uống thường xuyên, vitamin, mỹ phẩm.
    """
    cache_key = f"replenish_{user_id}_{limit}"
    cached = await cache.get(cache_key)
    if cached:
        return response("replenishment", cached, "Có thể đến chu kỳ mua lại", "cache", ["delivered_orders", "purchase_interval"])

    results = await hybrid_engine.get_replenishment(user_id, limit)
    # Cache ngắn hơn (1h) vì phụ thuộc vào ngày hiện tại
    await cache.set(cache_key, results, ttl_hours=1)
    return response("replenishment", results, "Có thể đến chu kỳ mua lại", evidence=["delivered_orders", "purchase_interval"])
