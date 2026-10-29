"""Internal schema specifications and conversion utilities for structured LLM outputs."""

from __future__ import annotations

import json
from typing import Any, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def get_schema_description(schema: type[BaseModel]) -> str:
    """Return JSON schema as a formatted string for prompt context."""
    return json.dumps(schema.model_json_schema(), indent=2)


def parse_and_validate(json_text: str, schema: type[T]) -> T:
    """Parse JSON text and validate strictly against Pydantic schema."""
    cleaned = json_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    data = json.loads(cleaned)
    return schema.model_validate(data)
