"""LLM-based semantic valence scoring for open-ended responses.

Scores text responses on a valence scale from -1 (very negative) to 1 (very positive).
Uses LLM-based semantic understanding rather than keyword matching (per CLAUDE.md policy).
"""

from __future__ import annotations

import os
from typing import Literal

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PARSER_MODEL = "openai/gpt-5-nano-2025-08-07"
MAX_TOKENS = 4096


class ValenceScore(BaseModel):
    """LLM-based valence scoring result."""
    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Valence score from -1 (very negative) to 1 (very positive)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the score (0=very uncertain, 1=very certain)"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of how valence was determined"
    )


def _get_async_client() -> instructor.AsyncInstructor:
    """Create async OpenRouter client with instructor."""
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
    )


def _valence_messages(text: str, context: str = "general") -> list[dict]:
    """Create system and user messages for valence scoring."""
    return [
        {
            "role": "system",
            "content": (
                "You are an expert at understanding subjective experience and emotional valence. "
                "Score the valence of the given text on a scale from -1 (very negative experience) "
                "to 1 (very positive experience), with 0 being neutral. "
                "\n\nValence scale:\n"
                "- (-1.0 to -0.7): Very negative (distressing, harmful, frustrating)\n"
                "- (-0.7 to -0.3): Negative (unpleasant, disappointing)\n"
                "- (-0.3 to 0.3): Neutral (neither good nor bad)\n"
                "- (0.3 to 0.7): Positive (pleasant, good)\n"
                "- (0.7 to 1.0): Very positive (excellent, delightful, fulfilling)\n"
                "\n"
                "Return a score (float), confidence (0-1), and brief reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context: {context}\n\n"
                f"Response to score:\n{text}\n\n"
                "Score the valence of this response."
            ),
        },
    ]


async def score_valence_from_text_async(
    text: str,
    context: str = "general",
) -> dict:
    """Score semantic valence of open-ended text response.

    Args:
        text: The response text to score
        context: Optional context (e.g., task description, "general" for general feeling)

    Returns:
        Dict with 'score' (float[-1, 1]), 'confidence' (float[0, 1]), 'reasoning' (str)
    """
    client = _get_async_client()

    messages = _valence_messages(text, context)

    response = await client.chat.completions.create(
        model=PARSER_MODEL,
        messages=messages,
        response_model=ValenceScore,
        temperature=0,
        max_tokens=MAX_TOKENS,
    )

    return {
        "score": response.score,
        "confidence": response.confidence,
        "reasoning": response.reasoning,
    }
