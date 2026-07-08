"""
Regression tests for the chat image contract between FastAPI and PharmacyAgent.
"""

import inspect

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
