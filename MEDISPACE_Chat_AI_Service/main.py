"""
MEDISPACE_Chat_AI_Service — main.py
FastAPI entry point cho AI Pharmacy Assistant.
Port: 8003
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

from src.agents.pharmacy_agent import PharmacyAgent

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("chat_ai")

agent = PharmacyAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🤖 Medispace Chat AI Service starting on port %s", os.getenv("PORT", "8001"))
    logger.info("🔗 Gemma API: %s | Model: %s", os.getenv("CUSTOM_LLM_BASE_URL"), os.getenv("CUSTOM_LLM_MODEL"))
    yield
    logger.info("Chat AI Service shutting down.")


app = FastAPI(
    title="Medispace Chat AI Service",
    description="AI Pharmacy Assistant using Gemma — fallback khi dược sĩ offline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    user_id: str
    history: Optional[List[dict]] = None
    context_products: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    reply: str
    classification: str         # "emergency" | "prescription_request" | "general"
    is_escalated: bool          # True nếu nên chuyển cho dược sĩ thật
    products_suggested: list    # Sản phẩm RAG gợi ý (Phase 2)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "medispace-chat-ai",
        "model": os.getenv("CUSTOM_LLM_MODEL", "gemma-4-e4b-it.gguf"),
        "llm_url": os.getenv("CUSTOM_LLM_BASE_URL", "https://llm.datateam.space"),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    logger.info("[/chat] user=%s conv=%s msg_len=%d", req.user_id, req.conversation_id, len(req.message))
    try:
        result = await agent.respond(
            message=req.message,
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            history=req.history,
            context_products=req.context_products,
        )
        logger.info("[/chat] classification=%s escalated=%s", result["classification"], result["is_escalated"])
        return ChatResponse(**result)
    except Exception as e:
        logger.error("[/chat] Error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    logger.info("[/chat/stream] user=%s conv=%s msg_len=%d", req.user_id, req.conversation_id, len(req.message))
    try:
        return StreamingResponse(
            agent.stream_respond(
                message=req.message,
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                history=req.history,
                context_products=req.context_products,
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error("[/chat/stream] Error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
