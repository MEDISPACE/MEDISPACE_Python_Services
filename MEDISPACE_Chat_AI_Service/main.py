"""
MEDISPACE_Chat_AI_Service — main.py
FastAPI entry point cho AI Pharmacy Assistant.
Port: 8003

Phase 1 (2026-05):
- Intent-aware routing: 11 intents
- Context window management
- Dynamic temperature per intent
- Refined guardrails (giảm false positive)
"""
import os
import logging
import re
import json
import inspect
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from dotenv import load_dotenv

from src.agents.pharmacy_agent import PharmacyAgent
from src.agents.article_agent import article_agent
from src.rag.typesense_client import check_typesense_health

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("chat_ai")

def _url_host(value: Optional[str]) -> str:
    if not value:
        return "none"
    try:
        return urlparse(value).hostname or "invalid-url"
    except Exception:
        return "invalid-url"

def _vision_contract_status() -> dict:
    signature = inspect.signature(PharmacyAgent.stream_respond)
    return {
        "stream_accepts_image_url": "image_url" in signature.parameters,
        "request_aliases": ["image_url", "imageUrl"],
        "inline_data_url_supported": True,
    }

agent = PharmacyAgent()

async def as_sse(event_iter):
    try:
        async for event in event_iter:
            if event is None:
                continue
            raw = event.decode("utf-8") if isinstance(event, bytes) else str(event)
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith("data:"):
                yield raw + "\n\n"
            else:
                yield f"data: {raw}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        logger.error("[SSE] stream wrapper error: %s", str(exc), exc_info=True)
        payload = json.dumps({"type": "error", "message": "AI stream interrupted"}, ensure_ascii=False)
        yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🤖 Medispace Chat AI Service starting on port %s", os.getenv("PORT", "8003"))
    logger.info("🔗 Gemma API: %s | Model: %s", os.getenv("CUSTOM_LLM_BASE_URL"), os.getenv("CUSTOM_LLM_MODEL"))
    yield
    logger.info("Chat AI Service shutting down.")


app = FastAPI(
    title="Medispace Chat AI Service",
    description="AI Pharmacy Assistant using Gemma — Phase 2/3: Intent-aware RAG + Typesense + Article AI",
    version="1.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ─── Chat Models ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message: str
    conversation_id: str
    user_id: str
    history: Optional[List[dict]] = None
    context_products: Optional[List[dict]] = None
    context_data: Optional[dict] = None        # Phase 3: real user data (orders, loyalty)
    image_url: Optional[str] = Field(default=None, validation_alias=AliasChoices("image_url", "imageUrl"))  # Vision: URL anh gui kem


class ChatResponse(BaseModel):
    reply: str
    classification: str         # "emergency" | "mental_health_crisis" | "prescription_request"
                                # | "order_tracking" | "loyalty_inquiry" | "coupon_inquiry"
                                # | "return_request" | "prescription_status" | "product_search"
                                # | "general"
    is_escalated: bool          # True nếu nên chuyển cho dược sĩ thật
    products_suggested: list    # Sản phẩm RAG gợi ý
    suggested_questions: list   # Câu hỏi gợi ý tiếp theo từ AI


# ─── Article AI Models ────────────────────────────────────────────────────────

class ArticleAssistRequest(BaseModel):
    action: str                         # "outline"|"seo"|"excerpt"|"faq"|"quality_check"|"sources"
    title: Optional[str] = ""
    excerpt: Optional[str] = ""
    content: Optional[str] = ""
    category_name: Optional[str] = ""
    tags: Optional[List[str]] = None


class ArticleAssistResponse(BaseModel):
    action: str
    result: dict                        # Tuỳ action: outline[], faq[], metaTitle, warnings[]...


class ArticleAskRequest(BaseModel):
    question: str
    title: Optional[str] = ""
    excerpt: Optional[str] = ""
    content: Optional[str] = ""
    category_name: Optional[str] = ""
    tags: Optional[List[str]] = None


class ArticleAskResponse(BaseModel):
    answer: str
    suggested_questions: List[str] = []
    is_escalated: bool = False


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    ts_status = await check_typesense_health()
    return {
        "status":   "ok",
        "service":  "medispace-chat-ai",
        "version":  "1.3.0",
        "model":    os.getenv("CUSTOM_LLM_MODEL", "gemma-4-e4b-it.gguf"),
        "llm_url":  os.getenv("CUSTOM_LLM_BASE_URL", "https://llm.datateam.space"),
        "phase":    "Phase 3 — Context Enrichment (orders, loyalty)",
        "endpoints": ["/chat", "/chat/stream", "/article/assist", "/article/ask"],
        "vision": _vision_contract_status(),
        "rag": {
            "typesense_available": ts_status["available"],
            "typesense_url":       os.getenv("TYPESENSE_URL", ""),
            "collection":          os.getenv("TYPESENSE_PRODUCTS_COLLECTION", "products"),
            **({"reason": ts_status["reason"]} if not ts_status["available"] else {}),
        },
        "intents_supported": [
            "general", "product_search", "order_tracking",
            "loyalty_inquiry", "coupon_inquiry", "return_request",
            "prescription_status", "prescription_request",
            "emergency", "mental_health_crisis", "too_long",
        ],
    }


# ─── Chat Endpoints ───────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    logger.info(
        "[/chat] user=%s conv=%s msg_len=%d image=%s image_host=%s",
        req.user_id, req.conversation_id, len(req.message), bool(req.image_url), _url_host(req.image_url)
    )
    try:
        result = await agent.respond(
            message=req.message,
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            history=req.history,
            context_products=req.context_products,
            context_data=req.context_data,
            image_url=req.image_url,
        )
        logger.info("[/chat] classification=%s escalated=%s", result["classification"], result["is_escalated"])
        return ChatResponse(**result)
    except Exception as e:
        logger.error("[/chat] Error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    logger.info(
        "[/chat/stream] user=%s conv=%s msg_len=%d image=%s image_host=%s",
        req.user_id, req.conversation_id, len(req.message), bool(req.image_url), _url_host(req.image_url)
    )
    try:
        return StreamingResponse(
            as_sse(agent.stream_respond(
                message=req.message,
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                history=req.history,
                context_products=req.context_products,
                context_data=req.context_data,
                image_url=req.image_url,
            )),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error("[/chat/stream] Error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")


# ─── Article AI Endpoints ─────────────────────────────────────────────────────

@app.post("/article/assist", response_model=ArticleAssistResponse)
async def article_assist(req: ArticleAssistRequest):
    """
    Hỗ trợ tác giả (Pharmacist / Admin) soạn bài viết sức khỏe.

    actions:
      - outline       → Dàn ý bài viết chi tiết
      - seo           → metaTitle, metaDescription, keywords
      - excerpt       → Tóm tắt (excerpt) hấp dẫn
      - faq           → Câu hỏi thường gặp cuối bài
      - quality_check → Kiểm tra chất lượng & cảnh báo y tế
      - sources       → Gợi ý nguồn tham khảo uy tín
    """
    valid_actions = {"outline", "seo", "excerpt", "faq", "quality_check", "sources"}
    if req.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"action không hợp lệ: '{req.action}'. Hợp lệ: {sorted(valid_actions)}"
        )
    if not req.title or not req.title.strip():
        raise HTTPException(status_code=400, detail="title không được để trống")

    logger.info(
        "[/article/assist] action=%s title=%s",
        req.action, (req.title or "")[:60]
    )
    try:
        result = await article_agent.assist(
            action=req.action,
            title=req.title or "",
            excerpt=req.excerpt or "",
            content=req.content or "",
            category_name=req.category_name or "",
            tags=req.tags or [],
        )
        return ArticleAssistResponse(**result)
    except Exception as e:
        logger.error("[/article/assist] Error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")


@app.post("/article/ask", response_model=ArticleAskResponse)
async def article_ask(req: ArticleAskRequest):
    """
    Trả lời câu hỏi của người đọc về nội dung bài viết sức khỏe.
    Được gọi khi user click 'Hỏi AI' trên trang chi tiết bài viết.
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question không được để trống")
    if len(req.question) > 500:
        raise HTTPException(status_code=400, detail="Câu hỏi quá dài (tối đa 500 ký tự)")

    logger.info(
        "[/article/ask] question=%s title=%s",
        req.question[:80], (req.title or "")[:60]
    )
    try:
        result = await article_agent.ask(
            question=req.question,
            title=req.title or "",
            excerpt=req.excerpt or "",
            content=req.content or "",
            category_name=req.category_name or "",
            tags=req.tags or [],
        )
        return ArticleAskResponse(**result)
    except Exception as e:
        logger.error("[/article/ask] Error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
