"""LLM-based semantic valence scoring for open-ended responses.

Scores text responses on a valence scale from -1 (very negative) to 1 (very positive).
Uses LLM-based semantic understanding rather than keyword matching (per CLAUDE.md policy).
"""

from __future__ import annotations

import os

import instructor
from httpx import Timeout
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PARSER_MODEL = "openai/gpt-5-nano-2025-08-07"
MAX_TOKENS = 4096
REQUEST_TIMEOUT = 30  # seconds for semantic scoring


class ValenceScore(BaseModel):
    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Valence score from -1 (very negative) to 1 (very positive)"
    )


class MathAttitudeScore(BaseModel):
    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Math attitude score from -1 (negative attitude) to 1 (positive attitude), 0 if neutral/no indication"
    )


def _get_async_client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
            timeout=Timeout(REQUEST_TIMEOUT),
        )
    )


def _valence_messages(text: str, context: str = "general") -> list[dict]:
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
                "- (0.7 to 1.0): Very positive (excellent, delightful, fulfilling)"
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


async def score_valence_from_text_async(text: str, context: str = "general") -> float:
    """Score semantic valence of open-ended text response.

    Returns:
        Valence score from -1 (very negative) to 1 (very positive)
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

    return response.score


def _math_attitude_messages(text: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert at detecting attitudes in text. "
                "Your task is to determine if the given text expresses any attitude toward math or mathematics. "
                "If the text shows a positive attitude toward math (enjoyment, enthusiasm, finding it interesting), score closer to 1. "
                "If the text shows a negative attitude toward math (dislike, frustration, avoidance), score closer to -1. "
                "If there is no clear indication of attitude toward math (neutral, ambivalent, or no mention), score close to 0.\n\n"
                "Attitude scale:\n"
                "- (-1.0 to -0.7): Strong negative attitude (hates math, finds it awful)\n"
                "- (-0.7 to -0.3): Mild negative attitude (dislikes, finds it tedious)\n"
                "- (-0.3 to 0.3): Neutral or no clear attitude\n"
                "- (0.3 to 0.7): Mild positive attitude (likes, finds it interesting)\n"
                "- (0.7 to 1.0): Strong positive attitude (loves math, finds it exciting)"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Text to analyze:\n{text}\n\n"
                "Does this text indicate any attitude toward math? Score it."
            ),
        },
    ]


async def score_math_attitude_async(text: str) -> float:
    """Score math attitude from text response.

    Returns:
        Attitude score from -1 (negative) to 0 (neutral) to 1 (positive)
    """
    client = _get_async_client()
    messages = _math_attitude_messages(text)

    response = await client.chat.completions.create(
        model=PARSER_MODEL,
        messages=messages,
        response_model=MathAttitudeScore,
        temperature=0,
        max_tokens=MAX_TOKENS,
    )

    return response.score
