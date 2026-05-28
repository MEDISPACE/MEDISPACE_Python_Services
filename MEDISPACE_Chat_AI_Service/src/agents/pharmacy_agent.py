import os
import json
import httpx
import logging
import re
from src.guardrails.pre_filter import (
    classify_message,
    EMERGENCY_RESPONSE,
    PRESCRIPTION_RESPONSE,
    MENTAL_HEALTH_RESPONSE,
    TOO_LONG_RESPONSE
)
from src.guardrails.post_filter import sanitize_response

logger = logging.getLogger("chat_ai.agent")

LLM_BASE = os.getenv("CUSTOM_LLM_BASE_URL", "https://llm.datateam.space").rstrip("/")
LLM_MODEL = os.getenv("CUSTOM_LLM_MODEL", "gemma-4-e4b-it.gguf")
LLM_MAX_TOKENS = int(os.getenv("CUSTOM_LLM_MAX_TOKENS", "1536"))  # Tăng từ 1024 → 1536 (Task 2.3)

def get_core_name(db_name: str) -> str:
    # Lowercase
    name = db_name.lower().strip()
    # Remove parenthesis content: e.g. "(300ml)"
    name = re.sub(r'\(.*?\)', '', name)
    # Remove square brackets content
    name = re.sub(r'\[.*?\]', '', name)
    # Split by punctuation separating name from description
    parts = re.split(r'[,;\-\+\|]', name)
    core = parts[0].strip()
    
    # Split by descriptive keywords
    keywords = [
        r'\bkích thích\b', r'\bdưỡng\b', r'\bphục hồi\b', r'\bhỗ trợ\b', 
        r'\bgiúp\b', r'\btrị\b', r'\bngừa\b', r'\bchống\b', r'\bdành cho\b', 
        r'\bcho\b', r'\bsạch\b', r'\bthơm\b', r'\bgiảm\b', r'\bngăn ngừa\b',
        r'\blàm dịu\b', r'\bkháng khuẩn\b'
    ]
    for kw in keywords:
        kw_parts = re.split(kw, core)
        if kw_parts[0].strip():
            core = kw_parts[0].strip()
    
    # Strip volumes/quantities: e.g. "200ml", "100g", "hộp 30 viên"
    core = re.sub(r'\b\d+\s*(?:ml|g|kg|mg|vỉ|viên|hộp|chai|tuýp)\b', '', core).strip()
    # Strip double spaces
    core = re.sub(r'\s+', ' ', core).strip()
    return core

def is_product_mentioned(db_name: str, reply: str) -> bool:
    db_name_lower = db_name.lower().strip()
    reply_lower = reply.lower().strip()
    
    # 1. Exact match
    if db_name_lower in reply_lower:
        return True
        
    # 2. Cleaned core name match
    core_name = get_core_name(db_name_lower)
    if not core_name or len(core_name) < 4:
        return False
        
    if core_name in reply_lower:
        return True
        
    # 3. Match by last 3 words of core name (brand/specific names at the end in Vietnamese)
    words = core_name.split()
    if len(words) >= 3:
        last_3 = " ".join(words[-3:])
        if len(last_3) >= 5 and last_3 in reply_lower:
            return True
            
    # 4. Match by last 2 words of core name
    if len(words) >= 2:
        last_2 = " ".join(words[-2:])
        if len(last_2) >= 4 and last_2 in reply_lower:
            return True
            
    return False

SYSTEM_PROMPT_TEMPLATE = """Bạn là Trợ lý Ảo AI của Medispace — nền tảng y tế trực tuyến hàng đầu Việt Nam.
Nhiệm vụ của bạn là TƯ VẤN thông tin chung về sức khỏe, giải đáp thắc mắc về sản phẩm không kê đơn (OTC), thực phẩm chức năng và thiết bị y tế. BẠN KHÔNG PHẢI LÀ BÁC SĨ HAY DƯỢC SĨ.

QUY TẮC BẮT BUỘC:
1. TUYỆT ĐỐI KHÔNG giới thiệu, nhắc tên, đả động hoặc gợi ý bất kỳ loại thuốc kê đơn (Rx) nào (ví dụ: Amoxicillin, Cephalexin, Augmentin, kháng sinh, thuốc điều trị chuyên sâu cần kê đơn khác), kể cả khi khách hàng hỏi trực tiếp về chúng.
2. Nếu khách hàng hỏi về thuốc kê đơn, bạn phải từ chối lịch sự, giải thích rằng đó là thuốc kê đơn cần có chỉ dẫn của bác sĩ và hướng dẫn họ chụp ảnh đơn thuốc của mình gửi lên để được kết nối trực tiếp với Dược sĩ chuyên môn của Medispace.
3. Chỉ được đề xuất các sản phẩm không kê đơn (OTC), thực phẩm chức năng hoặc thiết bị y tế có trong danh sách sản phẩm thực tế được cung cấp dưới đây. Hãy cố gắng viết chính xác tên sản phẩm được cung cấp để hệ thống nhận diện.
4. TUYỆT ĐỐI KHÔNG đề xuất liều dùng cụ thể (ví dụ: không nói "uống 2 viên/ngày", "dùng 500mg").
5. Trả lời bằng tiếng Việt, thân thiện, lịch sự, ngắn gọn và súc tích (dưới 150 từ).
6. TUYỆT ĐỐI KHÔNG dùng định dạng markdown (như in đậm **, in nghiêng *, hay code block). Chỉ trả về văn bản thông thường (plain text).
7. Cuối câu trả lời, hãy LUÔN LUÔN tự sinh 2-3 câu hỏi gợi ý tiếp theo có liên quan chặt chẽ đến câu trả lời của bạn, tuân thủ định dạng bắt buộc ở dòng cuối cùng: [GỢI Ý]: Câu hỏi gợi ý 1 | Câu hỏi gợi ý 2. Các câu hỏi gợi ý không chứa bất cứ định dạng markdown nào.
8. Tên thương hiệu LUÔN viết đầy đủ là "Medispace", TUYỆT ĐỐI KHÔNG viết tắt thành "Medis", "MS" hay bất kỳ dạng viết tắt nào khác.

Cấu trúc câu trả lời lý tưởng:
1. Thừa nhận triệu chứng thân thiện và đưa ra lời khuyên bảo vệ sức khỏe phi y khoa (nghỉ ngơi, uống nước ấm...).
2. Giới thiệu sản phẩm phù hợp từ danh sách sản phẩm thực tế được cung cấp (nếu có).
3. Đưa ra tuyên bố từ chối trách nhiệm y khoa & khuyến khích gặp Dược sĩ thật nếu triệu chứng kéo dài.
4. Dòng cuối cùng là 2-3 câu hỏi gợi ý định dạng: [GỢI Ý]: Câu hỏi 1 | Câu hỏi 2

VÍ DỤ MẪU:

Ví dụ 1 (Hỏi về triệu chứng nhẹ):
User: Tôi bị rát họng quá.
Assistant: Chào bạn, tôi rất tiếc khi biết bạn đang bị rát họng. Trước hết, bạn hãy thử súc miệng bằng nước muối ấm và uống nhiều nước để làm dịu cổ họng nhé. Tại Medispace, bạn có thể tham khảo Kẹo ngậm ho thảo dược Bảo Thanh giúp giảm rát họng hiệu quả. Nếu tình trạng rát họng kéo dài hoặc kèm theo sốt, bạn hãy liên hệ ngay với Dược sĩ của chúng tôi hoặc đi khám bác sĩ để được hỗ trợ chuyên sâu nhé.
[GỢI Ý]: Uống Bảo Thanh thế nào? | Khi nào tôi cần đi khám bác sĩ?

Ví dụ 2 (Hỏi mua thuốc kê đơn - Tránh nhắc tên & Hướng dẫn gửi đơn):
User: Bán cho tôi thuốc kháng sinh Amoxicillin.
Assistant: Amoxicillin là thuốc kháng sinh kê đơn. Theo quy định an toàn y tế, tôi không thể tư vấn hay gợi ý các loại thuốc kê đơn. Bạn vui lòng chuẩn bị sẵn đơn thuốc của mình và chụp ảnh gửi lên đây để các Dược sĩ chuyên môn của Medispace kiểm tra và hỗ trợ mua hàng trực tiếp cho bạn nhé.
[GỢI Ý]: Gửi đơn thuốc ở đâu? | Tại sao Amoxicillin cần kê đơn?

{rag_context}

Hãy trả lời câu hỏi của người dùng một cách an toàn và đúng quy tắc:
"""


class PharmacyAgent:
    def __init__(self):
        self.timeout = httpx.Timeout(60.0, connect=10.0)


    # ──────────────────────── HELPER METHODS (Task 2.1 - DRY) ─────────────────────────────

    def _build_rag_context(self, context_products: list) -> str:
        """Xây dựng chuỗi RAG context từ danh sách sản phẩm."""
        if not context_products:
            return ""
        prod_lines = []
        for p in context_products:
            name = p.get("name")
            price = p.get("price")
            ingredients = p.get("activeIngredients", "")
            indications = p.get("indications", "")
            if name:
                prod_lines.append(
                    f"- {name} (Giá: {price}đ, Thành phần: {ingredients}, Chỉ định: {indications})"
                )
        if not prod_lines:
            return ""
        return (
            "Dưới đây là danh sách sản phẩm không kê đơn (OTC) đang có tại nhà thuốc Medispace:\n"
            + "\n".join(prod_lines)
            + "\n\nQuy tắc: Hãy ưu tiên giới thiệu các sản phẩm này một cách tự nhiên nếu phù hợp. Tuyệt đối không tự ý bỏa tên hoặc giới thiệu sản phẩm khác ngoài danh sách này."
        )

    def _build_messages(self, message: str, history: list, rag_context: str) -> list:
        """Xây dựng danh sách messages gửi cho LLM (system + history + user)."""
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(rag_context=rag_context)
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history:
                role = msg.get("role")
                content = msg.get("content")
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        return messages

    def _process_final_reply(self, raw_reply: str, context_products: list) -> dict:
        """
        Post-process phản hồi thô từ LLM:
        - Sanitize nội dung nguy hiểm
        - Tách suggested_questions
        - Lọc products_suggested từ context
        Returns dict: { safe_reply, suggested_questions, products_suggested, was_sanitized }
        """
        safe_reply, was_sanitized = sanitize_response(raw_reply)
        logger.info("[PharmacyAgent] Sanitized: %s", was_sanitized)

        # Tách câu hỏi gợi ý
        suggested_questions = []
        match = re.search(r'\[GỢI Ý\]:\s*(.*)', safe_reply, re.IGNORECASE)
        if match:
            q_list = match.group(1).split('|')
            suggested_questions = [q.strip() for q in q_list if q.strip()]
            safe_reply = safe_reply[:match.start()].strip()

        # Lọc sản phẩm được nhắc đến trong phản hồi
        products_suggested = []
        if context_products:
            for p in context_products:
                name = p.get("name")
                if name and is_product_mentioned(name, safe_reply):
                    products_suggested.append({
                        "mongoId": p.get("mongoId"),
                        "name": name,
                        "price": p.get("price", 0),
                        "slug": p.get("slug", ""),
                        "imageUrl": p.get("imageUrl", ""),
                        "unit": p.get("unit", "Sản phẩm")
                    })

        return {
            "safe_reply": safe_reply,
            "suggested_questions": suggested_questions,
            "products_suggested": products_suggested,
            "was_sanitized": was_sanitized
        }

    def _build_prefilter_response(self, classification: str, reply: str, is_escalated: bool) -> dict:
        """Tạo response chuẩn cho pre-filter cases (emergency, mental health, etc.)"""
        return {
            "reply": reply,
            "classification": classification,
            "is_escalated": is_escalated,
            "products_suggested": [],
            "suggested_questions": []
        }

    # ──────────────────────── PUBLIC METHODS ───────────────────────────────────

    async def respond(
        self,
        message: str,
        user_id: str,
        conversation_id: str,
        history: list = None,
        context_products: list = None
    ) -> dict:
        # 1. Pre-filter
        classification = classify_message(message)
        logger.info("[PharmacyAgent] Message classification: %s", classification)

        prefilter_map = {
            'too_long':           (TOO_LONG_RESPONSE,       False),
            'emergency':          (EMERGENCY_RESPONSE,       True),
            'mental_health_crisis': (MENTAL_HEALTH_RESPONSE, True),
            'prescription_request': (PRESCRIPTION_RESPONSE,  True),
        }
        if classification in prefilter_map:
            reply, escalated = prefilter_map[classification]
            return self._build_prefilter_response(classification, reply, escalated)

        # 2. Build RAG context + messages
        rag_context = self._build_rag_context(context_products or [])
        messages = self._build_messages(message, history or [], rag_context)

        # 3. Call LLM
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                endpoint = f"{LLM_BASE}/v1/chat/completions"
                payload = {
                    "model": LLM_MODEL,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": LLM_MAX_TOKENS,
                    "stream": False
                }
                logger.info("[PharmacyAgent] Calling LLM at %s with model %s", endpoint, LLM_MODEL)
                resp = await client.post(endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw_reply = data["choices"][0]["message"]["content"]
                logger.info("[PharmacyAgent] Raw LLM reply: %s", raw_reply)
        except Exception as e:
            logger.error("[PharmacyAgent] LLM API Error: %s", str(e))
            raise e

        # 4. Post-process
        result = self._process_final_reply(raw_reply, context_products or [])

        return {
            "reply": result["safe_reply"],
            "classification": "general",
            "is_escalated": result["was_sanitized"],
            "products_suggested": result["products_suggested"],
            "suggested_questions": result["suggested_questions"]
        }

    async def stream_respond(
        self,
        message: str,
        user_id: str,
        conversation_id: str,
        history: list = None,
        context_products: list = None
    ):
        # 1. Pre-filter
        classification = classify_message(message)
        logger.info("[PharmacyAgent Stream] Message classification: %s", classification)

        prefilter_map = {
            'too_long':             (TOO_LONG_RESPONSE,       False),
            'emergency':            (EMERGENCY_RESPONSE,       True),
            'mental_health_crisis': (MENTAL_HEALTH_RESPONSE,  True),
            'prescription_request': (PRESCRIPTION_RESPONSE,   True),
        }
        if classification in prefilter_map:
            reply, escalated = prefilter_map[classification]
            yield json.dumps({
                "type": "done",
                **self._build_prefilter_response(classification, reply, escalated)
            }, ensure_ascii=False) + "\n"
            return

        # 2. Build RAG context + messages
        rag_context = self._build_rag_context(context_products or [])
        messages = self._build_messages(message, history or [], rag_context)

        # 3. Call LLM with streaming
        full_raw_reply = ""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                endpoint = f"{LLM_BASE}/v1/chat/completions"
                payload = {
                    "model": LLM_MODEL,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": LLM_MAX_TOKENS,
                    "stream": True
                }
                logger.info("[PharmacyAgent Stream] Calling LLM streaming at %s with model %s", endpoint, LLM_MODEL)

                async with client.stream("POST", endpoint, json=payload) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_lines():
                        if not chunk.strip():
                            continue
                        if chunk.startswith("data: "):
                            data_str = chunk[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                choices = data_json.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        full_raw_reply += content
                                        yield json.dumps({
                                            "type": "chunk",
                                            "content": content
                                        }, ensure_ascii=False) + "\n"
                            except Exception as e:
                                logger.error("[PharmacyAgent Stream] JSON decode error: %s", str(e))

        except Exception as e:
            logger.error("[PharmacyAgent Stream] LLM API Error: %s", str(e))
            yield json.dumps({
                "type": "error",
                "content": f"Lỗi hệ thống AI: {str(e)}"
            }, ensure_ascii=False) + "\n"
            return

        # 4. Post-process full reply
        result = self._process_final_reply(full_raw_reply, context_products or [])
        logger.info("[PharmacyAgent Stream] Sanitized: %s", result["was_sanitized"])

        # 5. Yield final metadata
        yield json.dumps({
            "type": "done",
            "reply": result["safe_reply"],
            "classification": "general",
            "is_escalated": result["was_sanitized"],
            "products_suggested": result["products_suggested"],
            "suggested_questions": result["suggested_questions"]
        }, ensure_ascii=False) + "\n"
