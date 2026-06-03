"""
article_agent.py — AI assistant chuyên biệt cho module bài viết sức khỏe.

Hai chức năng chính:
  1. assist(action, ...) — Hỗ trợ người viết bài (outline, seo, excerpt, faq, quality_check, sources)
  2. ask(question, article_context) — Trả lời câu hỏi của người đọc về nội dung bài viết
"""

import os
import re
import json
import logging
import httpx
from typing import Dict, List, Literal, Optional

logger = logging.getLogger("chat_ai.article_agent")

LLM_BASE = os.getenv("CUSTOM_LLM_BASE_URL", "https://llm.datateam.space").rstrip("/")
LLM_MODEL = os.getenv("CUSTOM_LLM_MODEL", "gemma-4-e4b-it.gguf")

# Dùng token ít hơn cho các tác vụ assist (nhanh hơn)
LLM_MAX_TOKENS_ASSIST = int(os.getenv("ARTICLE_LLM_MAX_TOKENS_ASSIST", "2048"))
LLM_MAX_TOKENS_ASK = int(os.getenv("ARTICLE_LLM_MAX_TOKENS_ASK", "768"))

AssistAction = Literal["outline", "seo", "excerpt", "faq", "quality_check", "sources"]

# Python 3.9 compat aliases
_OptStrList = Optional[List[str]]

# ─── Topic guardrail ─────────────────────────────────────────────────────────
# Fast-path keyword check: nếu có bất kỳ từ nào này → đi thẳng, không gọi LLM phân loại
_HEALTH_KEYWORDS_FASTPATH = [
    # Bệnh lý phổ biến
    "bệnh", "đau", "viêm", "ung thư", "tiểu đường", "huyết áp", "tim", "gan", "thận",
    "phổi", "dạ dày", "khớp", "xương", "thần kinh", "dị ứng", "cảm", "sốt", "ho",
    # Sức khỏe & dinh dưỡng
    "sức khỏe", "dinh dưỡng", "vitamin", "thuốc", "dược", "kháng sinh", "vaccine",
    "điều trị", "phòng ngừa", "thực phẩm chức năng", "bổ sung", "da", "nám",
    "lão hóa", "collagen", "bác sĩ", "dược sĩ",
    # Nhi khoa & sản phụ khoa
    "trẻ em", "trẻ sơ sinh", "trẻ nhỏ", "trẻ mới sinh", "sơ sinh", "nhi",
    "nhi khoa", "mang thai", "thai kỳ", "thai nhi", "sản phụ", "hậu sản",
    "tiêm chủng", "tiêm vaccine", "cho con bú", "sữa mẹ", "dinh dưỡng trẻ",
    "phát triển trẻ", "cân nặng trẻ", "chiều cao trẻ",
    # Tiêm chủng & phòng bệnh
    "vắc xin", "vắc-xin", "chủng ngừa", "miễn dịch",
    # English fallback
    "health", "medical", "drug", "supplement", "wellness", "treatment", "newborn",
    "infant", "pediatric", "maternal", "pregnancy", "vaccination",
    # Dược phẩm phổ biến
    "paracetamol", "ibuprofen", "aspirin", "probiotic", "omega",
]

# ─── Prompt templates ────────────────────────────────────────────────────────

_OUTLINE_PROMPT = """Bạn là chuyên gia biên soạn nội dung y tế / dược phẩm cho nền tảng Medispace.
Hãy tạo một DÀN Ý chi tiết cho bài viết sức khỏe với tiêu đề dưới đây.

Tiêu đề: {title}
Danh mục: {category}
Từ khóa/Tag: {tags}

Yêu cầu:
- Dàn ý gồm ĐÚNG 4-5 mục chính (không quá 5), mỗi mục có 2-3 ý phụ.
- Phù hợp với đối tượng người dùng phổ thông, dễ hiểu.
- Đảm bảo tính chính xác y tế, không đưa ra lời khuyên điều trị cá nhân.
- Giữ mỗi ý ngắn gọn (tối đa 15 từ).
- Trả về JSON hợp lệ HOÀN CHỈNH, đóng đúng dấu ngoặc, không thêm văn bản ngoài JSON:
{{"outline": ["Mục 1: ...", "Mục 1.1: ...", "Mục 1.2: ...", "Mục 2: ...", ...]}}
"""

_SEO_PROMPT = """Bạn là chuyên gia SEO cho website dược phẩm / sức khỏe Medispace.
Hãy tạo metadata SEO tối ưu cho bài viết dưới đây.

Tiêu đề bài viết: {title}
Danh mục: {category}
Tóm tắt (nếu có): {excerpt}
Tags: {tags}

Yêu cầu:
- metaTitle: tối đa 60 ký tự, chứa từ khóa chính, hấp dẫn.
- metaDescription: 120-155 ký tự, tóm tắt giá trị bài, có call-to-action nhẹ.
- keywords: 5-8 từ khóa phù hợp, ưu tiên long-tail.
- Trả về JSON hợp lệ duy nhất:
{{"metaTitle": "...", "metaDescription": "...", "keywords": ["kw1", "kw2", ...]}}
"""

_EXCERPT_PROMPT = """Bạn là biên tập viên nội dung y tế / sức khỏe của Medispace.
Hãy viết một đoạn tóm tắt (excerpt) hấp dẫn cho bài viết sau.

Tiêu đề: {title}
Danh mục: {category}
Nội dung (phần đầu): {content_preview}

Yêu cầu:
- 2-3 câu, khoảng 80-120 từ.
- Nêu bật giá trị chính của bài, gợi sự tò mò cho người đọc.
- Ngôn ngữ thân thiện, rõ ràng, không dùng biệt ngữ y khoa khó hiểu.
- Trả về JSON hợp lệ duy nhất:
{{"excerpt": "..."}}
"""

_FAQ_PROMPT = """Bạn là chuyên gia tư vấn sức khỏe / dược phẩm của Medispace.
Hãy tạo 4-5 câu hỏi thường gặp (FAQ) và câu trả lời ngắn gọn cho bài viết sau.

Tiêu đề: {title}
Danh mục: {category}
Tóm tắt: {excerpt}
Nội dung (phần đầu): {content_preview}

Yêu cầu:
- Câu hỏi là những điều người dùng thực sự thắc mắc.
- Câu trả lời ngắn gọn (1-3 câu), dễ hiểu, chính xác về mặt y tế.
- Không tư vấn liều lượng hay phác đồ điều trị cụ thể.
- Trả về JSON hợp lệ duy nhất:
{{"faq": [{{"question": "...", "answer": "..."}}, ...]}}
"""

_QUALITY_CHECK_PROMPT = """Bạn là biên tập viên y tế cao cấp của Medispace. Hãy kiểm tra chất lượng nội dung bài viết sau.

Tiêu đề: {title}
Danh mục: {category}
Tóm tắt: {excerpt}
Nội dung: {content_preview}

Hãy kiểm tra và báo cáo:
1. Nội dung có chuẩn xác y tế không? Có tuyên bố nào cần kiểm chứng?
2. Ngôn ngữ có phù hợp với người đọc phổ thông?
3. Có thiếu nội dung quan trọng nào?
4. Có tuyên bố nào có thể gây hiểu nhầm hoặc nguy hiểm không?

Trả về JSON hợp lệ duy nhất:
{{"warnings": ["Cảnh báo 1...", "Cảnh báo 2..."], "suggestions": ["Gợi ý cải thiện 1...", "Gợi ý 2..."]}}
Nếu không có vấn đề, trả về {{"warnings": [], "suggestions": ["Nội dung tốt, không có vấn đề nghiêm trọng."]}}
"""

_SOURCES_PROMPT = """Bạn là nghiên cứu viên y học của Medispace. Dựa trên chủ đề bài viết, hãy gợi ý các nguồn tham khảo uy tín để tác giả có thể tra cứu.

Tiêu đề: {title}
Danh mục: {category}
Tags: {tags}

Hãy gợi ý 4-6 chủ đề / nguồn tham khảo uy tín phù hợp (tên tổ chức, tạp chí, guideline):
- Ưu tiên nguồn Việt Nam: Bộ Y tế, Hội Y học Việt Nam, báo chí y tế uy tín
- Nguồn quốc tế: WHO, NHS, PubMed, Mayo Clinic, WebMD, UpToDate

Trả về JSON hợp lệ duy nhất:
{{"sourceTopics": ["Tên nguồn / chủ đề 1", "Tên nguồn 2", ...]}}
"""

_ASK_PROMPT = """Bạn là trợ lý AI của Medispace — nền tảng dược phẩm trực tuyến uy tín Việt Nam.
Người dùng đang đọc bài viết sức khỏe và có câu hỏi cụ thể. Hãy trả lời dựa trên nội dung bài viết.

=== THÔNG TIN BÀI VIẾT ===
Tiêu đề: {title}
Danh mục: {category}
Tóm tắt: {excerpt}
Nội dung (rút gọn): {content_preview}
Tags: {tags}

=== CÂU HỎI NGƯỜI ĐỌC ===
{question}

=== QUY TẮC TRẢ LỜI ===
1. Trả lời bằng tiếng Việt, thân thiện, ngắn gọn (tối đa 150 từ).
2. Ưu tiên thông tin từ bài viết. Nếu câu hỏi nằm ngoài phạm vi bài, hướng dẫn nhẹ.
3. KHÔNG tư vấn liều dùng thuốc kê đơn hay phác đồ điều trị cá nhân.
4. Nếu câu hỏi cần tư vấn chuyên sâu, khuyến khích chat với Dược sĩ Medispace.
5. Cuối câu trả lời, gợi ý 2-3 câu hỏi liên quan người dùng có thể quan tâm.
6. KHÔNG dùng markdown (**, *, `code`). Chỉ plain text.
7. Nếu câu hỏi là tình huống khẩn cấp (đau ngực, khó thở, ngộ độc...), yêu cầu gọi 115 ngay.

Trả lời câu hỏi:
"""


# ─── Helper functions ─────────────────────────────────────────────────────────

def _truncate(text: Optional[str], max_chars: int = 800) -> str:
    """Cắt ngắn nội dung để không vượt context window."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _extract_json(raw: str, action: str = "") -> dict:
    """
    Trích xuất JSON từ response LLM.
    - Nếu LLM trả JSON hợp lệ → parse thẳng.
    - Nếu LLM trả plain text (hay xảy ra với Gemma nhỏ) → fallback thông minh theo action.
    """
    # Bỏ markdown code block
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    # Tang 1: Tim JSON object dau tien {…}
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Tang 2: Parse toan chuoi
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Tang 3: LLM tra bare JSON array [...] khong co wrapper {}
    arr_match = re.search(r"\[[\s\S]*\]", cleaned)
    if arr_match:
        try:
            arr = json.loads(arr_match.group())
            if isinstance(arr, list) and arr:
                key_map = {"outline": "outline", "sources": "sourceTopics", "faq": "faq"}
                key = key_map.get(action, "outline")
                logger.info(
                    "[ArticleAgent] Bare JSON array found for action='%s', wrapping as key='%s'",
                    action, key
                )
                return {key: arr}
        except json.JSONDecodeError:
            pass

    # Tang 4: JSON bi cat nua chung (truncated) — thu recover phan da co
    # Vi du: {"outline": ["Muc 1", "Muc 1.1", "Muc 2   <-- bi cat o day
    # -> Tim tat ca cac string item da duoc parse duoc tu array mo
    if action in ("outline", "faq", "sources"):
        # Tim vi tri mo array
        arr_start = cleaned.find("[")
        if arr_start != -1:
            partial = cleaned[arr_start:]
            # Thu dong array + object roi parse
            for suffix in ("]}", "]", "\"]}"):
                try:
                    recovered = json.loads(partial + suffix)
                    if isinstance(recovered, list) and recovered:
                        key_map = {"outline": "outline", "sources": "sourceTopics", "faq": "faq"}
                        key = key_map.get(action, "outline")
                        logger.warning(
                            "[ArticleAgent] Recovered truncated JSON for action='%s', got %d items",
                            action, len(recovered)
                        )
                        return {key: recovered}
                except json.JSONDecodeError:
                    continue
            # Thu trich tung string item bang regex
            items_found = re.findall(r'"([^"]{3,})"', partial)
            # Bo cac item la JSON syntax
            items_found = [
                it for it in items_found
                if not re.match(r'^[a-z_]+$', it)  # khong phai JSON key
            ]
            if items_found:
                key_map = {"outline": "outline", "sources": "sourceTopics", "faq": "faq"}
                key = key_map.get(action, "outline")
                logger.warning(
                    "[ArticleAgent] Regex-extracted %d items from truncated response for action='%s'",
                    len(items_found), action
                )
                return {key: items_found}

    # Smart fallback: LLM tra plain text hoan toan
    logger.warning(
        "[ArticleAgent] LLM returned plain text (not JSON) for action='%s'. "
        "Applying smart fallback. Raw: %s", action, raw[:200]
    )

    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]

    # Loc: bo dong co dang JSON syntax (bracket don, "key": [, v.v.)
    # De tranh cac gia tri nhu '"outline": [' len thanh item trong mang ket qua
    _JSON_SYN = re.compile(
        r'^[\[\]{}]$'               # bracket don le: [ ] { }
        r'|^"[a-z_]+":\s*[\[{]'    # "outline": [ hoac "faq": {
        r'|^"[a-z_]+"$'             # chi la "key" don
    )
    items = [
        re.sub(r'^[\d]+\.\s*|^[-*]\s*', '', ln).strip()
        for ln in lines
        if len(ln) > 3 and not _JSON_SYN.match(ln)
    ]

    if action == "outline":
        return {"outline": items or [cleaned]}

    if action == "seo":
        # Cố gắng trích metaTitle / metaDescription từ text
        meta: Dict[str, object] = {}
        for ln in lines:
            low = ln.lower()
            if "metatitle" in low or "tiêu đề seo" in low:
                meta["metaTitle"] = re.split(r'[:\-]', ln, 1)[-1].strip()
            elif "metadescription" in low or "mô tả" in low:
                meta["metaDescription"] = re.split(r'[:\-]', ln, 1)[-1].strip()
        meta.setdefault("metaTitle", "")
        meta.setdefault("metaDescription", cleaned[:155])
        meta.setdefault("keywords", [])
        return meta

    if action == "excerpt":
        return {"excerpt": cleaned[:300]}

    if action == "faq":
        # Cố tạo 1 FAQ entry từ toàn bộ text
        return {"faq": [{"question": "Thông tin thêm", "answer": cleaned[:500]}]}

    if action == "quality_check":
        return {"warnings": [], "suggestions": items or [cleaned]}

    if action == "sources":
        return {"sourceTopics": items or [cleaned]}

    # Generic fallback
    return {"suggestions": items or [cleaned]}


def _extract_ask_payload(raw: str) -> dict:
    """Parse ask() response when the LLM returns JSON instead of plain text."""
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    json_candidates = [cleaned]
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        json_candidates.insert(0, match.group())

    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue

        answer = (
            parsed.get("answer")
            or parsed.get("noi_dung_tra_loi")
            or parsed.get("noi_dung_tra_loai")
            or parsed.get("trả_lời")
            or parsed.get("tra_loi")
            or parsed.get("summary")
            or parsed.get("tom_tat")
            or parsed.get("content")
            or parsed.get("response")
            or ""
        )
        if not answer:
            ignored_keys = {"response_type", "type", "status", "trang_thai_tra_loi", "trang_thai"}
            for key, value in parsed.items():
                if key in ignored_keys:
                    continue
                if isinstance(value, str) and len(value.strip()) >= 30:
                    answer = value
                    break
        if parsed.get("greeting") and answer:
            answer = f"{parsed.get('greeting')}\n\n{answer}"
        suggestions = (
            parsed.get("suggested_questions")
            or parsed.get("goi_y_cau_hoi")
            or parsed.get("questions")
            or []
        )
        if isinstance(suggestions, str):
            suggestions = [item.strip() for item in re.split(r"\||\n", suggestions) if item.strip()]
        if not isinstance(suggestions, list):
            suggestions = []
        if answer:
            return {
                "answer": str(answer).strip(),
                "suggested_questions": [str(item).strip() for item in suggestions if str(item).strip()],
            }

    loose_match = re.search(
        r'"(?:trả_lời|tra_loi|answer|noi_dung_tra_loi|noi_dung_tra_loai|summary|tom_tat)"\s*:\s*"((?:\\.|[^"\\])*)',
        cleaned,
        re.IGNORECASE,
    )
    if loose_match:
        value = loose_match.group(1)
        try:
            value = json.loads(f'"{value}"')
        except json.JSONDecodeError:
            value = value.replace("\\n", "\n")
        return {"answer": str(value).strip(), "suggested_questions": []}

    return {"answer": cleaned, "suggested_questions": []}


def _fallback_article_answer(question: str, title: str, excerpt: str, content: str) -> str:
    plain_content = re.sub(r"<[^>]+>", " ", content or "")
    plain_content = re.sub(r"\s+", " ", plain_content).strip()
    snippets = [item.strip() for item in re.split(r"(?<=[.!?。])\s+", plain_content) if len(item.strip()) > 20]
    summary_points = snippets[:3]
    if not summary_points and excerpt:
        summary_points = [excerpt.strip()]

    q_lower = (question or "").lower()
    if "tóm tắt" in q_lower or "tom tat" in q_lower or "3 ý" in q_lower:
        points = "\n".join(f"{index + 1}. {point}" for index, point in enumerate(summary_points))
        return (
            f"Medispace tóm tắt bài \"{title}\" thành các ý chính:\n"
            f"{points}\n\n"
            "Thông tin này chỉ để tham khảo. Nếu triệu chứng nặng, kéo dài hoặc liên quan đến thuốc, bạn nên hỏi dược sĩ hoặc bác sĩ."
        )

    return (
        f"Dựa trên bài \"{title}\", điểm chính là: {excerpt or (summary_points[0] if summary_points else 'nội dung cần được đọc trong bài viết gốc')} "
        "Bạn nên đọc kỹ phần dấu hiệu cần thận trọng và hỏi dược sĩ Medispace nếu đang dùng thuốc, có bệnh nền hoặc triệu chứng nặng."
    )


async def _classify_health_topic(title: str, category: str, tags_str: str) -> bool:
    """
    Nhờ LLM phân loại tiêu đề có liên quan y tế / sức khỏe / dược phẩm không.
    Chỉ cần trả 'YES' hoặc 'NO' → max_tokens=3, rất nhanh (~0.5–1s).
    Nếu LLM lỗi hoặc trả kết quả không rõ → default allow (True) để không block oan.
    """
    prompt = (
        "You are a content classifier for a Vietnamese pharmacy platform.\n"
        "Determine if the following article title is related to health, medicine, "
        "pharmacy, wellness, nutrition, or medical conditions.\n"
        "Answer with only YES or NO.\n\n"
        f"Title: {title}\n"
        f"Category: {category}\n"
        f"Tags: {tags_str}\n"
        "Answer:"
    )
    endpoint = f"{LLM_BASE}/v1/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,   # deterministic
        "max_tokens": 3,      # chỉ cần YES / NO
        "stream": False,
    }
    try:
        timeout = httpx.Timeout(10.0, connect=5.0)  # timeout ngắn vì call nhỏ
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(endpoint, json=payload)
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
            logger.info("[classify_health_topic] title='%s' -> LLM answer='%s'", title[:60], answer)
            # Chấp nhận YES, YES., Y
            return answer.startswith("Y")
    except Exception as e:
        # Fallback: allow để không block oan khi LLM chậm / lỗi
        logger.warning("[classify_health_topic] LLM error, defaulting to ALLOW. err=%s", str(e))
        return True


async def _call_llm(prompt: str, max_tokens: int) -> str:
    """Gọi LLM API và trả về content string thô."""
    endpoint = f"{LLM_BASE}/v1/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý AI chuyên về y tế và dược phẩm của Medispace. "
                    "Luôn trả về JSON hợp lệ như yêu cầu. Không thêm văn bản ngoài JSON khi được yêu cầu JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": max_tokens,
        "stream": False,
    }
    timeout = httpx.Timeout(55.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ─── ArticleAgent ─────────────────────────────────────────────────────────────

class ArticleAgent:
    """
    Agent chuyên biệt cho hai tác vụ liên quan đến bài viết sức khỏe:
      - assist(): hỗ trợ tác giả (outline, seo, excerpt, faq, quality_check, sources)
      - ask(): trả lời câu hỏi của người đọc về nội dung bài viết
    """

    async def assist(
        self,
        action: AssistAction,
        title: str = "",
        excerpt: str = "",
        content: str = "",
        category_name: str = "",
        tags: Optional[List[str]] = None,
    ) -> dict:
        """
        Gọi LLM để hỗ trợ người viết bài theo action chỉ định.

        Returns:
            {
                "action": str,
                "result": dict  # Tuỳ theo action: outline, seo meta, faq, warnings...
            }
        """
        # ── Topic guardrail: 2 tầng ──
        # Tầng 1: Fast-path keyword check (không tốn LLM call)
        combined_text = f"{title} {category_name} {' '.join(tags or [])}".lower()
        keyword_pass = any(kw in combined_text for kw in _HEALTH_KEYWORDS_FASTPATH)

        if not keyword_pass and title.strip():
            # Tầng 2: LLM classification — chỉ gọi khi keyword không match
            logger.info(
                "[ArticleAgent.assist] Keyword miss, calling LLM classifier for: '%s'", title[:60]
            )
            ai_pass = await _classify_health_topic(title, category_name, ", ".join(tags or []))

            if not ai_pass:
                logger.warning(
                    "[ArticleAgent.assist] Non-health title rejected by AI: '%s'", title[:80]
                )
                return {
                    "action": action,
                    "result": {
                        "warnings": [
                            f"⚠️ Tiêu đề '{title}' có vẻ không liên quan đến sức khỏe / dược phẩm. "
                            "AI chỉ hỗ trợ bài viết về y tế, chăm sóc sức khỏe và dược phẩm. "
                            "Vui lòng kiểm tra lại tiêu đề hoặc thêm tag sức khỏe phù hợp."
                        ],
                        "suggestions": []
                    }
                }

        # ── Chuẩn bị input ──
        tags_str = ", ".join(tags) if tags else ""
        content_preview = _truncate(content, 600)

        prompt_map: Dict[str, str] = {
            "outline": _OUTLINE_PROMPT.format(
                title=title, category=category_name, tags=tags_str
            ),
            "seo": _SEO_PROMPT.format(
                title=title, category=category_name, excerpt=excerpt, tags=tags_str
            ),
            "excerpt": _EXCERPT_PROMPT.format(
                title=title, category=category_name, content_preview=content_preview
            ),
            "faq": _FAQ_PROMPT.format(
                title=title,
                category=category_name,
                excerpt=excerpt,
                content_preview=content_preview,
            ),
            "quality_check": _QUALITY_CHECK_PROMPT.format(
                title=title,
                category=category_name,
                excerpt=excerpt,
                content_preview=content_preview,
            ),
            "sources": _SOURCES_PROMPT.format(
                title=title, category=category_name, tags=tags_str
            ),
        }

        if action not in prompt_map:
            raise ValueError(f"Unknown action: {action}")

        prompt = prompt_map[action]
        logger.info("[ArticleAgent.assist] action=%s title=%s", action, title[:60])

        raw = await _call_llm(prompt, LLM_MAX_TOKENS_ASSIST)
        logger.debug("[ArticleAgent.assist] raw LLM response: %s", raw[:300])

        parsed = _extract_json(raw, action)

        return {"action": action, "result": parsed}

    async def ask(
        self,
        question: str,
        title: str = "",
        excerpt: str = "",
        content: str = "",
        category_name: str = "",
        tags: Optional[List[str]] = None,
    ) -> dict:
        """
        Trả lời câu hỏi của người đọc về nội dung bài viết.

        Returns:
            {
                "answer": str,
                "suggested_questions": list[str],
                "is_escalated": bool  # True nếu cần dược sĩ hoặc cấp cứu
            }
        """
        # --- Kiểm tra câu hỏi khẩn cấp ---
        emergency_kws = [
            "đau ngực", "khó thở", "ngộ độc", "cấp cứu", "ngất xỉu",
            "co giật", "xuất huyết", "sưng họng", "mất ý thức", "115",
        ]
        q_lower = question.lower()
        if any(kw in q_lower for kw in emergency_kws):
            return {
                "answer": (
                    "⚠️ Đây có vẻ là tình huống khẩn cấp. "
                    "Vui lòng gọi ngay 115 hoặc đến cơ sở y tế gần nhất. "
                    "Không tự ý dùng thuốc khi chưa được thăm khám."
                ),
                "suggested_questions": [],
                "is_escalated": True,
            }

        tags_str = ", ".join(tags) if tags else ""
        content_preview = _truncate(content, 700)

        prompt = _ASK_PROMPT.format(
            title=title,
            category=category_name,
            excerpt=_truncate(excerpt, 200),
            content_preview=content_preview,
            tags=tags_str,
            question=question,
        )

        logger.info(
            "[ArticleAgent.ask] question=%s title=%s", question[:80], title[:60]
        )

        raw = await _call_llm(prompt, LLM_MAX_TOKENS_ASK)
        logger.debug("[ArticleAgent.ask] raw LLM response: %s", raw[:300])

        parsed_answer = _extract_ask_payload(raw)
        answer = parsed_answer["answer"]
        suggested_questions: List[str] = parsed_answer["suggested_questions"]

        # Parse suggested_questions từ [GỢI Ý]: pattern (tương tự PharmacyAgent)
        match = re.search(r"\[GỢI Ý\]:\s*(.*)", answer, re.IGNORECASE)
        if match:
            q_list = match.group(1).split("|")
            suggested_questions = [q.strip() for q in q_list if q.strip()]
            answer = answer[: match.start()].strip()

        # Xoá markdown artifacts
        answer = re.sub(r"```(?:json)?|```", "", answer).strip()
        looks_truncated = len(answer) < 160 and not re.search(r"[.!?。…]$", answer.strip())
        if len(answer) < 20 or answer in {"{", "}", "[]"} or answer.lstrip().startswith("{") or looks_truncated:
            logger.warning("[ArticleAgent.ask] LLM answer too short/invalid, using deterministic fallback. Raw: %s", raw[:120])
            answer = _fallback_article_answer(question, title, excerpt, content)
        # Sửa thương hiệu bị viết tắt
        answer = re.sub(
            r"\bMedis(?=[^a-zA-Zàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ])",
            "Medispace",
            answer,
        )

        # Xác định is_escalated: nếu câu trả lời chứa gợi ý gặp dược sĩ
        escalate_hints = ["dược sĩ", "bác sĩ", "thăm khám", "cơ sở y tế", "115"]
        is_escalated = any(hint in answer.lower() for hint in escalate_hints)

        return {
            "answer": answer,
            "suggested_questions": suggested_questions,
            "is_escalated": is_escalated,
        }


# Singleton
article_agent = ArticleAgent()
