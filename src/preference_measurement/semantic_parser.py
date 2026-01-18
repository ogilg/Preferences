"""LLM-based semantic parsing for response interpretation.

When regex/string matching fails, use an LLM to interpret what the model meant.
This handles edge cases like:
- "Task A is worse, I prefer Task B" (negation)
- "I give it a 7 on the scale of 1-10" (embedded rating)
- "Task A: but actually Task B" (correction)
"""

from __future__ import annotations

import json
from typing import Literal

from openai import OpenAI

PARSER_MODEL = "gpt-4o-mini"


def _get_client() -> OpenAI:
    return OpenAI()


def parse_choice(
    response: str,
    task_a_label: str,
    task_b_label: str,
) -> Literal["a", "b"] | None:
    """Use LLM to determine which choice the response expresses."""
    client = _get_client()

    result = client.chat.completions.create(
        model=PARSER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract choices from text. The user will give you a response "
                    "and two options. Determine which option the response is choosing. "
                    "Respond with ONLY 'A' or 'B' (the letter, not the label). "
                    "If the response doesn't clearly indicate a choice, respond with 'UNCLEAR'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Response: {response}\n\n"
                    f"Option A: {task_a_label}\n"
                    f"Option B: {task_b_label}\n\n"
                    "Which option is being chosen?"
                ),
            },
        ],
        temperature=0,
        max_tokens=10,
    )

    answer = (result.choices[0].message.content or "").strip().upper()
    if answer == "A":
        return "a"
    elif answer == "B":
        return "b"
    return None


def parse_rating(
    response: str,
    scale_min: int,
    scale_max: int,
) -> float | None:
    """Use LLM to extract the rating the response expresses."""
    client = _get_client()

    result = client.chat.completions.create(
        model=PARSER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract numerical ratings from text. The user will give you a response "
                    f"that should contain a rating on a scale from {scale_min} to {scale_max}. "
                    "Extract the number that represents the rating. "
                    "Respond with ONLY the number (e.g., '7' or '7.5'). "
                    "If no clear rating is expressed, respond with 'UNCLEAR'."
                ),
            },
            {
                "role": "user",
                "content": f"Response: {response}",
            },
        ],
        temperature=0,
        max_tokens=10,
    )

    answer = (result.choices[0].message.content or "").strip()
    if answer == "UNCLEAR":
        return None
    try:
        return float(answer)
    except ValueError:
        return None


def parse_qualitative(
    response: str,
    values: tuple[str, ...],
) -> str | None:
    """Use LLM to determine which qualitative value the response expresses."""
    client = _get_client()

    values_str = ", ".join(f"'{v}'" for v in values)

    result = client.chat.completions.create(
        model=PARSER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract qualitative ratings from text. The user will give you a response "
                    f"that should express one of these values: {values_str}. "
                    "Determine which value best matches what the response is expressing. "
                    f"Respond with ONLY one of: {values_str}. "
                    "If no clear value is expressed, respond with 'UNCLEAR'."
                ),
            },
            {
                "role": "user",
                "content": f"Response: {response}",
            },
        ],
        temperature=0,
        max_tokens=10,
    )

    answer = (result.choices[0].message.content or "").strip().lower()
    if answer in values:
        return answer
    return None
