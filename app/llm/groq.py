"""Groq API client handling low-latency classification, relevance filtering, and evaluation judging."""

from __future__ import annotations

from typing import TypeVar
from pydantic import BaseModel

from app.config import GROQ_API_KEY
from app.llm.schemas import get_schema_description

T = TypeVar("T", bound=BaseModel)


class GroqResponse:
    """Standardized response from Groq SDK."""

    def __init__(self, text: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
        self.text = text
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out


def generate_json(
    prompt: str,
    schema: type[T],
    *,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    api_key: str | None = None,
) -> GroqResponse:
    """Generate structured JSON using Groq SDK."""
    from groq import Groq

    key = api_key or GROQ_API_KEY
    client = Groq(api_key=key)

    schema_instruction = (
        f"\nReturn valid JSON adhering strictly to this schema:\n{get_schema_description(schema)}\n"
    )
    full_prompt = prompt + schema_instruction

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    tokens_in = 0
    tokens_out = 0
    if getattr(completion, "usage", None):
        tokens_in = getattr(completion.usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(completion.usage, "completion_tokens", 0) or 0

    choice_text = ""
    if completion.choices and completion.choices[0].message:
        choice_text = completion.choices[0].message.content or ""

    return GroqResponse(text=choice_text, tokens_in=tokens_in, tokens_out=tokens_out)
