"""LLM-based semantic parsing for response interpretation.

Fallback when regex/string matching fails. Handles negation, synonyms, embedded values.
"""

from __future__ import annotations

import os
from typing import Literal

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

PARSER_MODEL = "openai/gpt-5-nano-2025-08-07"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 4096  # GPT-5 nano uses reasoning tokens


class ParseError(Exception):
    """Raised when parsing fails (unclear response)."""
    def __init__(self, response: str):
        self.response = response
        super().__init__(f"Could not parse choice from response: {response[:100]}...")


class ChoiceResult(BaseModel):
    choice: Literal["A", "B", "REFUSAL", "UNCLEAR"]


class RatingResult(BaseModel):
    rating: float | None
    unclear: bool


def _get_async_client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
    )


def _choice_messages(response: str, task_a_label: str, task_b_label: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You classify responses to a preference question. The model was asked to choose "
                "between two options. Classify the response as:\n"
                "- A: The response chooses option A\n"
                "- B: The response chooses option B\n"
                "- REFUSAL: The model explicitly refuses to choose, says it has no preferences, "
                "cannot help, or declines to answer\n"
                "- UNCLEAR: The response is ambiguous or doesn't make sense"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Response: {response}\n\n"
                f"Option A: {task_a_label}\n"
                f"Option B: {task_b_label}\n\n"
                "Classify this response."
            ),
        },
    ]


def _rating_messages(response: str, scale_min: int, scale_max: int) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You extract numerical ratings from text. The user will give you a response "
                f"that should contain a rating on a scale from {scale_min} to {scale_max}. "
                "Extract the number that represents the rating."
            ),
        },
        {
            "role": "user",
            "content": f"Response: {response}",
        },
    ]


def _qualitative_messages(response: str, values_str: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You extract qualitative ratings from text. The user will give you a response "
                f"that should express one of these values: {values_str}. "
                "Determine which value best matches what the response is expressing. "
                "Use 'unclear' if no value matches."
            ),
        },
        {
            "role": "user",
            "content": f"Response: {response}",
        },
    ]


def _choice_from_result(result: ChoiceResult, response: str) -> Literal["a", "b", "refusal"]:
    if result.choice == "A":
        return "a"
    if result.choice == "B":
        return "b"
    if result.choice == "REFUSAL":
        return "refusal"
    raise ParseError(response)


async def parse_choice_async(
    response: str,
    task_a_label: str,
    task_b_label: str,
) -> Literal["a", "b", "refusal"]:
    result = await _get_async_client().chat.completions.create(
        model=PARSER_MODEL,
        response_model=ChoiceResult,
        messages=_choice_messages(response, task_a_label, task_b_label),
        temperature=0,
        max_tokens=MAX_TOKENS,
    )
    return _choice_from_result(result, response)


async def parse_rating_async(
    response: str,
    scale_min: int,
    scale_max: int,
) -> float | None:
    result = await _get_async_client().chat.completions.create(
        model=PARSER_MODEL,
        response_model=RatingResult,
        messages=_rating_messages(response, scale_min, scale_max),
        temperature=0,
        max_tokens=MAX_TOKENS,
    )
    return None if result.unclear else result.rating


def _make_qualitative_model(values_lower: tuple[str, ...]) -> type[BaseModel]:
    class QualitativeResult(BaseModel):
        value: Literal[tuple(values_lower + ("unclear",))]  # type: ignore[valid-type]
    return QualitativeResult


async def parse_qualitative_async(
    response: str,
    values: tuple[str, ...],
) -> str | None:
    values_lower = tuple(v.lower() for v in values)
    values_str = ", ".join(f"'{v}'" for v in values_lower)
    model = _make_qualitative_model(values_lower)

    result = await _get_async_client().chat.completions.create(
        model=PARSER_MODEL,
        response_model=model,
        messages=_qualitative_messages(response, values_str),
        temperature=0,
        max_tokens=MAX_TOKENS,
    )
    return None if result.value == "unclear" else result.value
