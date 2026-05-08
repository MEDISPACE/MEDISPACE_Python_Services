"""
jobs.py - APScheduler background jobs cho ML retraining
"""
from src.models.hybrid_engine import HybridEngine


def setup_scheduler(scheduler, engine: HybridEngine):
    """
    Dang ky cac background jobs.
    Goi trong lifespan cua FastAPI.
    """
    retrain_hours = 6

    # Retrain tat ca models moi 6 gio
    scheduler.add_job(
        _retrain_job,
        'interval',
        hours=retrain_hours,
        id='retrain_all_models',
        args=[engine],
        replace_existing=True
    )

    print(f"[Scheduler] Retrain job registered (every {retrain_hours}h)")


async def _retrain_job(engine: HybridEngine):
    """Wrapper async cho scheduler."""
    print("\n[Scheduler] Starting scheduled retrain...")
    try:
        await engine.train_all()
        print("[Scheduler] Scheduled retrain completed.")
    except Exception as e:
        print(f"[Scheduler] Retrain failed: {e}")
