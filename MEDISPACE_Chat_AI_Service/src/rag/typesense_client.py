"""
src/rag/typesense_client.py
Typesense RAG Client — Phase 2

Tự động query Typesense để lấy sản phẩm liên quan nhất
dựa trên tin nhắn của user và intent, thay vì chỉ dùng
context_products được truyền từ FE.

Schema Typesense (products collection):
  mongoId, name, slug, featuredImage, price, rating,
  activeIngredients, indications, shortDescription,
  categoryName, brandName, dosageForm, strength,
  requiresPrescription, isActive, inStock, stockQuantity
"""
import os
import httpx
import logging
import unicodedata
from typing import Optional

logger = logging.getLogger("chat_ai.rag")

TYPESENSE_URL        = os.getenv("TYPESENSE_URL", "http://localhost:8108").rstrip("/")
TYPESENSE_API_KEY    = os.getenv("TYPESENSE_API_KEY", "")
TYPESENSE_COLLECTION = os.getenv("TYPESENSE_PRODUCTS_COLLECTION", "products")
TYPESENSE_ARTICLES   = os.getenv("TYPESENSE_ARTICLES_COLLECTION", "articles")

# ── Hybrid Search flag ────────────────────────────────────────────────────────────
# Đặt False để tắt vector search (ví dụ: khi embedding model chưa load xong).
# Mặc định True — nếu Typesense chưa có embedding field, nó sẽ trả lỗi
# và code tự fallback về BM25 thông qua try/except.
_VECTOR_SEARCH_ENABLED = os.getenv("TYPESENSE_VECTOR_SEARCH", "true").lower() == "true"

# Timeout ngắn — RAG nên fail fast để không block LLM call
_TIMEOUT = httpx.Timeout(6.0, connect=2.5)

EMBEDDING_FIELD = "embedding"

SEMANTIC_QUERY_EXPANSIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("sot", "ha sot", "nong sot", "on lanh", "lanh run", "cam cum", "cam lanh", "met moi trong nguoi", "nhuc moi"),
        "paracetamol",
    ),
    (
        ("nong trong nguoi", "nong nguoi", "bi nong trong", "noi mun nong"),
        "thanh nhiệt mát gan giải độc gan chức năng gan",
    ),
    (
        ("mat gan", "giai doc gan", "thanh nhiet"),
        "thanh nhiệt mát gan giải độc gan chức năng gan",
    ),
    (
        ("mat nuoc", "thieu nuoc", "bu nuoc", "dien giai", "tieu chay mat nuoc"),
        "oresol bù nước điện giải dung dịch bù nước chất điện giải",
    ),
)



INTENT_RAG_CONFIG: dict[str, dict] = {
    # Tư vấn OTC: ưu tiên theo triệu chứng + tên + thành phần
    "general": {
        "query_by":         "name,activeIngredients,indications,shortDescription,categoryName",
        "query_by_weights": "5,4,5,2,2",           # indications nâng 3→5: khớp triệu chứng
        "num_typos":        "2,2,1,1,1",            # số lỗi chính tả từng field
        "prefix":           "true,false,false,false,false",  # prefix search cho name
        "filter_by":        "isActive:=true && requiresPrescription:=false && inStock:=true",
        "sort_by":          "_text_match:desc,rating:desc,reviewCount:desc",
        "per_page":         5,
    },
    # Tìm kiếm sản phẩm: rộng hơn, bao gồm cả Rx, thêm brandName
    "product_search": {
        "query_by":         "name,activeIngredients,indications,shortDescription,categoryName,brandName",
        "query_by_weights": "6,4,4,2,2,3",           # trọng số ưu tiên name+brandName cao hơn (user tìm cụ thể)
        "num_typos":        "2,2,1,1,1,2",            # brandName typo cao (tên nước ngoài)
        "prefix":           "true,false,false,false,false,true",  # prefix cho name+brand
        "filter_by":        "isActive:=true && inStock:=true",
        "sort_by":          "_text_match:desc,requiresPrescription:asc,rating:desc",
        "per_page":         6,
    },
    # Đổi trả: không cần RAG sản phẩm (xử lý quy trình)
    "return_request": None,
    # Theo dõi đơn hàng: không cần RAG sản phẩm
    "order_tracking": None,
    # Loyalty: không cần RAG sản phẩm
    "loyalty_inquiry": None,
    # Coupon: không cần RAG sản phẩm
    "coupon_inquiry": None,
    # Prescription status: không cần RAG sản phẩm
    "prescription_status": None,
}

def _normalize_ascii(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value or "")
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return normalized.replace("đ", "d").replace("Đ", "D").lower()

def _expand_semantic_query(query: str) -> str:
    """
    Map ambiguous Vietnamese symptom phrases to pharmacy-domain terms.

    Keeping the literal words for "nong trong nguoi" makes BM25 over-rank
    heat-related medical supplies, so matched expansions become the search
    query instead of being appended to the original phrase.
    """
    normalized = _normalize_ascii(query)
    additions: list[str] = []
    for triggers, expansion in SEMANTIC_QUERY_EXPANSIONS:
        if any(trigger in normalized for trigger in triggers):
            additions.append(expansion)

    if not additions:
        return query

    extra_phrases = []
    for expansion in additions:
        normalized_expansion = _normalize_ascii(expansion)
        if normalized_expansion not in normalized and expansion not in extra_phrases:
            extra_phrases.append(expansion)
    return " ".join(extra_phrases) if extra_phrases else query

def _is_fever_query(query: str) -> bool:
    normalized = _normalize_ascii(query)
    return any(trigger in normalized for trigger in ("sot", "ha sot", "nong sot", "on lanh", "lanh run", "cam cum", "cam lanh", "met moi trong nguoi", "nhuc moi"))

def _is_irrelevant_for_fever_query(query: str, doc: dict) -> bool:
    if not _is_fever_query(query):
        return False

    primary_haystack = _normalize_ascii(
        " ".join(
            str(doc.get(field, "") or "")
            for field in ("name", "activeIngredients", "categoryName", "brandName")
        )
    )
    full_haystack = _normalize_ascii(
        " ".join(
            str(doc.get(field, "") or "")
            for field in (
                "name",
                "activeIngredients",
                "indications",
                "shortDescription",
                "categoryName",
                "brandName",
            )
        )
    )
    has_primary_relevant = any(
        term in primary_haystack
        for term in ("ha sot", "sot", "paracetamol", "acetaminophen", "ibuprofen", "giam dau", "dau nhuc", "nhuc dau")
    )
    has_symptom_relevant = any(term in full_haystack for term in ("ha sot", "giam dau", "dau nhuc", "nhuc dau"))
    if has_primary_relevant or has_symptom_relevant:
        return False

    return True

def _append_csv_value(value: str, extra: str) -> str:
    return f"{value},{extra}" if value else extra

def _enable_auto_embedding_query(params: dict, *, alpha: float, k: int, distance_threshold: float) -> None:
    """
    Typesense auto-embedding search requires the embedding field in query_by.
    The empty vector in vector_query tells Typesense to embed `q` by itself.
    """
    query_by = str(params.get("query_by", ""))
    fields = [field.strip() for field in query_by.split(",") if field.strip()]
    if EMBEDDING_FIELD not in fields:
        params["query_by"] = _append_csv_value(query_by, EMBEDDING_FIELD)

        if params.get("query_by_weights"):
            params["query_by_weights"] = _append_csv_value(str(params["query_by_weights"]), "1")
        if params.get("num_typos") and "," in str(params["num_typos"]):
            params["num_typos"] = _append_csv_value(str(params["num_typos"]), "0")
        if params.get("prefix") and "," in str(params["prefix"]):
            params["prefix"] = _append_csv_value(str(params["prefix"]), "false")

    params["vector_query"] = (
        f"{EMBEDDING_FIELD}:([], k:{k}, "
        f"distance_threshold:{distance_threshold}, alpha:{alpha})"
    )
    params["exclude_fields"] = EMBEDDING_FIELD
    params.pop("exhaustive_search", None)

def _remove_last_csv_value(value: str) -> str:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if parts and parts[-1] == EMBEDDING_FIELD:
        parts.pop()
    elif parts:
        parts.pop()
    return ",".join(parts)

def _disable_auto_embedding_query(params: dict) -> None:
    params.pop("vector_query", None)
    params.pop("exclude_fields", None)
    query_by = str(params.get("query_by", ""))
    fields = [field.strip() for field in query_by.split(",") if field.strip()]
    if fields and fields[-1] == EMBEDDING_FIELD:
        fields.pop()
        params["query_by"] = ",".join(fields)
        for key in ("query_by_weights", "num_typos", "prefix"):
            if params.get(key) and "," in str(params[key]):
                params[key] = _remove_last_csv_value(str(params[key]))


def _extract_search_query(message: str, intent: str) -> str:
    """
    Trích xuất từ khóa tìm kiếm từ tin nhắn của user.
    Đơn giản hóa: dùng toàn bộ message (Typesense xử lý tốt với typo tolerance).
    Giới hạn độ dài để tránh query quá dài.
    """
    # Trim và giới hạn 150 ký tự
    query = message.strip()[:150]

    # Loại bỏ các từ phổ thông không có giá trị search
    # NGUYÊN TẮC: Chỉ loại particle tiếng Việt (thán từ, trợ từ thuần túy).
    # KHÔNG loại các danh từ có nghĩa như "thuốc", "bệnh", "đau"...
    stop_words = [
        # Đại từ nhân xưng / trợ từ
        "tôi", "bị", "có", "thể", "được", "không", "nên", "cần", "muốn",
        # Thán từ / cảm thán
        "hỏi", "cho", "xin", "ơi", "à", "nhé", "vậy", "ạ", "dạ",
        # Tên brand (không cần search)
        "medispace",
        # Từ hỏi quá chung chung
        "gì", "nào",
        # ĐÃ BỎ: "thuốc" (từ khóa cốt lõi ngành dược — KHÔNG lọc)
        # ĐÃ BỎ: "sao" (có thể là "đau sao lưng", không lọc)
    ]
    words = query.split()
    filtered = [w for w in words if w.lower() not in stop_words]

    # Nếu sau khi filter còn >= 2 từ → dùng filtered, không thì dùng nguyên
    if len(filtered) >= 2:
        return " ".join(filtered)
    return query


async def search_products_for_rag(
    message: str,
    intent: str,
    limit: int = 5,
) -> list[dict]:
    """
    Query Typesense để lấy danh sách sản phẩm OTC phù hợp nhất
    với câu hỏi của user, dùng làm RAG context cho LLM.

    Args:
        message: Tin nhắn gốc của user
        intent:  Intent đã phân loại
        limit:   Số sản phẩm tối đa trả về

    Returns:
        List sản phẩm, mỗi item gồm:
        {mongoId, name, slug, price, imageUrl, unit,
         activeIngredients, indications, requiresPrescription}
        Trả về [] nếu Typesense không available.
    """
    # Một số intent không cần RAG sản phẩm
    rag_config = INTENT_RAG_CONFIG.get(intent)
    if rag_config is None:
        logger.debug("[RAG] Intent '%s' không cần RAG products", intent)
        return []

    if not TYPESENSE_API_KEY:
        logger.warning("[RAG] TYPESENSE_API_KEY chưa được cấu hình — bỏ qua RAG")
        return []

    raw_query = _extract_search_query(message, intent)
    fever_query = _is_fever_query(raw_query)
    query = _expand_semantic_query(raw_query)
    if not query:
        return []

    per_page = min(limit, rag_config["per_page"])

    params = {
        "q":                  query,
        "query_by":           rag_config["query_by"],
        "query_by_weights":   rag_config["query_by_weights"],
        "filter_by":          rag_config["filter_by"],
        "sort_by":            rag_config["sort_by"],
        "per_page":           per_page,
        # Per-field typo tolerance (nếu config có) — fallback về global num_typos=2
        "num_typos":          rag_config.get("num_typos", "2"),
        # Prefix search per-field: giúp tìm khi user gõ chưa hết từ (vd: "Para" → Paracetamol)
        "prefix":             rag_config.get("prefix", "true"),
        # Dùng exhaustive search để không bỏ sót kết quả khi corpus nhỏ
        "exhaustive_search":  "true",
        "include_fields":     (
            "mongoId,name,slug,price,featuredImage,"
            "activeIngredients,indications,requiresPrescription,"
            "categoryName,brandName,rating,inStock"
        ),
    }
    if fever_query:
        params.pop("exhaustive_search", None)

    # ── Hybrid Search: thêm vector_query nếu được bật ─────────────────────────────
    # alpha=0.7: 70% BM25 score + 30% vector score
    # Ưu tiên BM25 vì tên thuốc cụ thể match keyword tốt hơn semantic
    # k=20: vector search trả 20 candidates rồi Typesense RRF fusion với BM25
    # distance_threshold=0.85: loại kết quả về ngữ nghĩa quá xa
    if _VECTOR_SEARCH_ENABLED and not fever_query:
        _enable_auto_embedding_query(params, alpha=0.7, k=20, distance_threshold=0.85)

    async def _run_search(search_params: dict) -> list[dict]:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            url  = f"{TYPESENSE_URL}/collections/{TYPESENSE_COLLECTION}/documents/search"
            resp = await client.get(
                url,
                params=search_params,
                headers={"X-TYPESENSE-API-KEY": TYPESENSE_API_KEY},
            )
            resp.raise_for_status()
            data = resp.json()

            hits = data.get("hits", [])
            products = []
            for hit in hits:
                if hit.get("text_match") == 0:
                    logger.debug(
                        "[RAG] Skip zero text_match hit for query='%s': %s",
                        query[:50],
                        hit.get("document", {}).get("name", ""),
                    )
                    continue
                doc = hit.get("document", {})
                if _is_irrelevant_for_fever_query(raw_query if fever_query else query, doc):
                    logger.debug(
                        "[RAG] Skip fever-irrelevant hit for query='%s': %s",
                        query[:50],
                        doc.get("name", ""),
                    )
                    continue
                products.append({
                    "mongoId":              doc.get("mongoId", ""),
                    "name":                 doc.get("name", ""),
                    "slug":                 doc.get("slug", ""),
                    "price":               doc.get("price", 0),
                    "imageUrl":            doc.get("featuredImage", ""),
                    "unit":                "Sản phẩm",
                    "activeIngredients":   doc.get("activeIngredients", ""),
                    "indications":         doc.get("indications", ""),
                    "requiresPrescription": doc.get("requiresPrescription", False),
                    "categoryName":        doc.get("categoryName", ""),
                    "brandName":           doc.get("brandName", ""),
                    "rating":              doc.get("rating", 0),
                })

            logger.info(
                "[RAG] Query='%s' intent='%s' → %d sản phẩm tìm được",
                query[:50], intent, len(products),
            )
            return products

    try:
        return await _run_search(params)

    except httpx.TimeoutException:
        logger.warning("[RAG] Typesense timeout cho query='%s'", query[:50])
        return []
    except httpx.HTTPStatusError as e:
        if _VECTOR_SEARCH_ENABLED and "vector_query" in params:
            logger.warning(
                "[RAG] Vector search failed (%s), retry BM25-only cho query='%s'",
                e.response.status_code,
                query[:50],
            )
            bm25_params = dict(params)
            _disable_auto_embedding_query(bm25_params)
            try:
                return await _run_search(bm25_params)
            except Exception as retry_error:
                logger.warning("[RAG] BM25 fallback error: %s", str(retry_error))
                return []
        logger.warning("[RAG] Typesense HTTP error: %s", e.response.status_code)
        return []
    except Exception as e:
        logger.warning("[RAG] Typesense error: %s", str(e))
        return []


async def search_articles_for_rag(
    message: str,
    limit: int = 3,    # Tăng từ 2→3: AI có thêm 1 bài viết context y tế
) -> list[dict]:
    """
    Query Typesense articles collection để lấy bài viết sức khỏe liên quan.
    Dùng bổ sung kiến thức y tế cho AI khi tư vấn.

    Returns:
        List articles: [{title, excerpt, slug}]
        Trả về [] nếu không có bài phù hợp hoặc Typesense không available.
    """
    if not TYPESENSE_API_KEY:
        return []

    query = _expand_semantic_query(message.strip()[:100])
    params = {
        "q":                query,
        "query_by":         "title,excerpt,content,tags",
        "query_by_weights": "5,3,2,1",
        "filter_by":        "isPublished:=true",
        "sort_by":          "_text_match:desc,viewCount:desc",
        "per_page":         limit,
        "num_typos":        1,
        "include_fields":   "mongoId,title,excerpt,slug,categoryName,tags",
    }

    # ── Hybrid Search cho articles ───────────────────────────────────────────────
    # alpha=0.6: semantic mạnh hơn (60% BM25, 40% vector) vì câu hỏi y tế
    # thường mapping với chủ đề bài viết theo ngữ nghĩa, không chỉ keyword
    # (VD: "cao huyết áp" → bài về tim mạch, giảm muối, tăng cường kali)
    if _VECTOR_SEARCH_ENABLED:
        _enable_auto_embedding_query(params, alpha=0.6, k=10, distance_threshold=0.80)

    async def _run_search(search_params: dict) -> list[dict]:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            url  = f"{TYPESENSE_URL}/collections/{TYPESENSE_ARTICLES}/documents/search"
            resp = await client.get(
                url,
                params=search_params,
                headers={"X-TYPESENSE-API-KEY": TYPESENSE_API_KEY},
            )
            resp.raise_for_status()
            data = resp.json()

            hits = data.get("hits", [])
            articles = []
            for hit in hits:
                doc = hit.get("document", {})
                articles.append({
                    "title":        doc.get("title", ""),
                    "excerpt":      doc.get("excerpt", ""),
                    "slug":         doc.get("slug", ""),
                    "categoryName": doc.get("categoryName", ""),
                })

            logger.info(
                "[RAG-Articles] Query='%s' → %d bài viết",
                query[:40], len(articles),
            )
            return articles

    try:
        return await _run_search(params)

    except httpx.HTTPStatusError as e:
        if _VECTOR_SEARCH_ENABLED and "vector_query" in params:
            logger.warning(
                "[RAG-Articles] Vector search failed (%s), retry BM25-only cho query='%s'",
                e.response.status_code,
                query[:40],
            )
            bm25_params = dict(params)
            _disable_auto_embedding_query(bm25_params)
            try:
                return await _run_search(bm25_params)
            except Exception as retry_error:
                logger.warning("[RAG-Articles] BM25 fallback error: %s", str(retry_error))
                return []
        logger.warning("[RAG-Articles] HTTP error: %s", e.response.status_code)
        return []
    except Exception as e:
        logger.warning("[RAG-Articles] Error: %s", str(e))
        return []


async def check_typesense_health() -> dict:
    """Kiểm tra Typesense có available không. Dùng ở health endpoint."""
    if not TYPESENSE_API_KEY:
        return {"available": False, "reason": "API key chưa cấu hình"}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
            resp = await client.get(
                f"{TYPESENSE_URL}/health",
                headers={"X-TYPESENSE-API-KEY": TYPESENSE_API_KEY},
            )
            resp.raise_for_status()
            return {"available": True, "url": TYPESENSE_URL}
    except Exception as e:
        return {"available": False, "reason": str(e)}
