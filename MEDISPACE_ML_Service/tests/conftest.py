"""
tests/conftest.py — Shared fixtures for ML model tests
"""
import pytest
from typing import List, Dict


def make_product(
    _id: str,
    name: str,
    category_id: str = "cat1",
    brand_id: str = "brand1",
    ingredients: str = "",
    indications: str = "",
    stock: int = 100,
    requires_rx: bool = False,
    rating: float = 4.0,
) -> Dict:
    return {
        "_id": _id,
        "name": name,
        "categoryId": category_id,
        "brandId": brand_id,
        "isActive": True,
        "requiresPrescription": requires_rx,
        "stockQuantity": stock,
        "rating": rating,
        "reviewCount": 10,
        "categoryName": "TestCategory",
        "brandName": "TestBrand",
        "details": {
            "activeIngredients": ingredients,
            "indications": indications,
        },
        "priceVariants": [{"price": 50000}],
    }


@pytest.fixture
def sample_products() -> List[Dict]:
    return [
        make_product("p1", "Paracetamol 500mg", ingredients="Paracetamol", indications="hạ sốt giảm đau", category_id="cat1"),
        make_product("p2", "Ibuprofen 400mg", ingredients="Ibuprofen", indications="chống viêm giảm đau", category_id="cat1"),
        make_product("p3", "Amoxicillin 500mg", ingredients="Amoxicillin", indications="kháng sinh viêm họng", category_id="cat2"),
        make_product("p4", "Vitamin C 1000mg", ingredients="Ascorbic acid", indications="tăng đề kháng", category_id="cat3"),
        make_product("p5", "Cetirizine 10mg", ingredients="Cetirizine", indications="dị ứng mũi viêm mũi", category_id="cat1"),
        make_product("p6", "OOS Product", stock=0, category_id="cat1"),  # out of stock
        make_product("p7", "Aspirin 100mg", ingredients="Acetylsalicylic acid Aspirin", indications="tim mạch chống đông", category_id="cat1"),
    ]
