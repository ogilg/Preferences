"""Shared OpenRouter client factory for LLM judges."""

from __future__ import annotations

import os

import instructor
from openai import AsyncOpenAI

PARSER_MODEL = "openai/gpt-5-nano-2025-08-07"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_async_client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
    )
