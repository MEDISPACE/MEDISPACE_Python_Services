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
from typing import Optional

logger = logging.getLogger("chat_ai.rag")

TYPESENSE_URL        = os.getenv("TYPESENSE_URL", "http://localhost:8108").rstrip("/")
TYPESENSE_API_KEY    = os.getenv("TYPESENSE_API_KEY", "")
TYPESENSE_COLLECTION = os.getenv("TYPESENSE_PRODUCTS_COLLECTION", "products")
TYPESENSE_ARTICLES   = os.getenv("TYPESENSE_ARTICLES_COLLECTION", "articles")

# Timeout ngắn — RAG nên fail fast để không block LLM call
_TIMEOUT = httpx.Timeout(3.0, connect=2.0)

# Intent → query strategy mapping
# Mỗi intent có cách extract query và filter khác nhau
INTENT_RAG_CONFIG: dict[str, dict] = {
    # Tư vấn OTC: ưu tiên OTC, tìm theo tên + chỉ định
    "general": {
        "query_by":         "name,activeIngredients,indications,shortDescription,categoryName",
        "query_by_weights": "5,4,3,2,1",
        "filter_by":        "isActive:=true && requiresPrescription:=false && inStock:=true",
        "sort_by":          "_text_match:desc,rating:desc,reviewCount:desc",
        "per_page":         5,
    },
    # Tìm kiếm sản phẩm: rộng hơn, bao gồm cả Rx
    "product_search": {
        "query_by":         "name,activeIngredients,indications,shortDescription,categoryName,brandName",
        "query_by_weights": "5,4,3,2,1,1",
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


def _extract_search_query(message: str, intent: str) -> str:
    """
    Trích xuất từ khóa tìm kiếm từ tin nhắn của user.
    Đơn giản hóa: dùng toàn bộ message (Typesense xử lý tốt với typo tolerance).
    Giới hạn độ dài để tránh query quá dài.
    """
    # Trim và giới hạn 150 ký tự
    query = message.strip()[:150]

    # Loại bỏ các từ phổ thông không có giá trị search
    stop_words = [
        "tôi", "bị", "có", "thể", "được", "không", "nên", "cần", "muốn",
        "hỏi", "cho", "xin", "ơi", "à", "nhé", "vậy", "ạ", "dạ",
        "medispace", "thuốc", "gì", "nào", "sao",
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

    query = _extract_search_query(message, intent)
    if not query:
        return []

    per_page = min(limit, rag_config["per_page"])

    params = {
        "q":                 query,
        "query_by":          rag_config["query_by"],
        "query_by_weights":  rag_config["query_by_weights"],
        "filter_by":         rag_config["filter_by"],
        "sort_by":           rag_config["sort_by"],
        "per_page":          per_page,
        "num_typos":         2,
        "include_fields":    (
            "mongoId,name,slug,price,featuredImage,"
            "activeIngredients,indications,requiresPrescription,"
            "categoryName,brandName,rating,inStock"
        ),
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            url  = f"{TYPESENSE_URL}/collections/{TYPESENSE_COLLECTION}/documents/search"
            resp = await client.get(
                url,
                params=params,
                headers={"X-TYPESENSE-API-KEY": TYPESENSE_API_KEY},
            )
            resp.raise_for_status()
            data = resp.json()

            hits = data.get("hits", [])
            products = []
            for hit in hits:
                doc = hit.get("document", {})
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

    except httpx.TimeoutException:
        logger.warning("[RAG] Typesense timeout cho query='%s'", query[:50])
        return []
    except httpx.HTTPStatusError as e:
        logger.warning("[RAG] Typesense HTTP error: %s", e.response.status_code)
        return []
    except Exception as e:
        logger.warning("[RAG] Typesense error: %s", str(e))
        return []


async def search_articles_for_rag(
    message: str,
    limit: int = 2,
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

    query = message.strip()[:100]
    params = {
        "q":            query,
        "query_by":     "title,excerpt,content,tags",
        "query_by_weights": "5,3,2,1",
        "filter_by":    "isPublished:=true",
        "sort_by":      "_text_match:desc,viewCount:desc",
        "per_page":     limit,
        "num_typos":    1,
        "include_fields": "mongoId,title,excerpt,slug,categoryName,tags",
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            url  = f"{TYPESENSE_URL}/collections/{TYPESENSE_ARTICLES}/documents/search"
            resp = await client.get(
                url,
                params=params,
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
