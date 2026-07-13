"""
Regression tests for the chat image contract between FastAPI and PharmacyAgent.
"""

import inspect
from unittest.mock import patch

import pytest

from src.agents.pharmacy_agent import PharmacyAgent, normalize_image_for_llm


def test_stream_respond_accepts_image_url_keyword():
    signature = inspect.signature(PharmacyAgent.stream_respond)

    assert "image_url" in signature.parameters
    assert signature.parameters["image_url"].default is None


@pytest.mark.asyncio
async def test_normalize_image_for_llm_accepts_inline_data_url():
    data_url = "data:image/png;base64,iVBORw0KGgo="

    assert await normalize_image_for_llm(data_url) == data_url


def test_process_final_reply_falls_back_to_rag_products_when_reply_mentions_none():
    agent = PharmacyAgent()
    context_products = [
        {
            "mongoId": "p1",
            "name": "Oresol 1g",
            "price": 12000,
            "slug": "oresol-1g",
            "imageUrl": "https://example.com/oresol.jpg",
            "unit": "Hộp",
            "requiresPrescription": False,
        }
    ]

    with patch("src.agents.pharmacy_agent.sanitize_response", return_value=("Tư vấn bù nước.", False)):
        with patch("src.agents.pharmacy_agent.is_product_mentioned", return_value=False):
            result = agent._process_final_reply("Tư vấn bù nước.", context_products)

    assert result["products_suggested"] == [
        {
            "mongoId": "p1",
            "name": "Oresol 1g",
            "price": 12000,
            "slug": "oresol-1g",
            "imageUrl": "https://example.com/oresol.jpg",
            "unit": "Hộp",
            "requiresPrescription": False,
        }
    ]
