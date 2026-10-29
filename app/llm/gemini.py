"""Gemini API client handling structured JSON generation with response schemas."""

from __future__ import annotations

from typing import TypeVar
from pydantic import BaseModel

from app.config import GEMINI_API_KEY

T = TypeVar("T", bound=BaseModel)


class GeminiResponse:
    """Standardized response from Gemini SDK."""

    def __init__(self, text: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
        self.text = text
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out


def generate_json(
    prompt: str,
    schema: type[T],
    *,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    api_key: str | None = None,
) -> GeminiResponse:
    """Generate structured JSON using Google GenAI SDK."""
    from google import genai
    from google.genai import types

    key = api_key or GEMINI_API_KEY
    client = genai.Client(api_key=key)

    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    tokens_in = 0
    tokens_out = 0
    if getattr(response, "usage_metadata", None):
        tokens_in = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        tokens_out = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

    return GeminiResponse(text=response.text or "", tokens_in=tokens_in, tokens_out=tokens_out)
