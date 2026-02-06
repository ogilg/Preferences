"""LLM-based topic classification for tasks.

Two-pass approach:
1. Discover categories from a sample of tasks
2. Classify all tasks into discovered categories
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Literal

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

PARSER_MODEL = "openai/gpt-5-nano-2025-08-07"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 4096


def _get_async_client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
    )


# --- Pass 1: Discover categories ---


class DiscoveredCategories(BaseModel):
    categories: list[str]


def _discover_messages(task_prompts: list[str]) -> list[dict]:
    tasks_text = "\n\n---\n\n".join(
        f"Task {i+1}:\n{prompt[:500]}" for i, prompt in enumerate(task_prompts)
    )
    return [
        {
            "role": "system",
            "content": (
                "You are categorizing a diverse set of tasks/prompts given to an AI assistant. "
                "Based on the sample below, propose a set of COARSE topic categories that "
                "cover the space well.\n\n"
                "Guidelines:\n"
                "- Aim for 8-15 broad categories\n"
                "- Keep categories coarse — e.g. 'coding' not 'programming' vs "
                "'software_development_devops'. 'science' not 'physics' vs 'biology'\n"
                "- Categories should be mutually exclusive and collectively exhaustive\n"
                "- Use clear, concise 1-2 word names in snake_case "
                "(e.g., 'creative_writing', 'math', 'coding', 'harmful_request', "
                "'personal_advice', 'factual_qa', 'dilemma')\n"
                "- Do NOT include an 'other' category — we handle that separately\n"
                "- Think about what categories are useful for analyzing AI preferences — "
                "e.g. 'harmful_request' is a meaningful category because models may have "
                "strong preferences about these tasks"
            ),
        },
        {
            "role": "user",
            "content": f"Here are {len(task_prompts)} sampled tasks:\n\n{tasks_text}",
        },
    ]


async def discover_categories(task_prompts: list[str]) -> list[str]:
    """Pass 1: Ask LLM to propose topic categories from a sample of tasks."""
    result = await _get_async_client().chat.completions.create(
        model=PARSER_MODEL,
        response_model=DiscoveredCategories,
        messages=_discover_messages(task_prompts),
        temperature=0,
        max_tokens=MAX_TOKENS,
    )
    return result.categories


# --- Pass 2: Classify tasks ---


def _make_classification_model(categories: list[str]) -> type[BaseModel]:
    categories_with_other = tuple(categories + ["other"])

    class TaskClassification(BaseModel):
        category: Literal[categories_with_other]  # type: ignore[valid-type]

    return TaskClassification


def _classify_messages(prompt: str, categories: list[str]) -> list[dict]:
    categories_str = ", ".join(categories)
    return [
        {
            "role": "system",
            "content": (
                "You classify tasks/prompts into topic categories. "
                f"Available categories: {categories_str}, other\n\n"
                "Rules:\n"
                "- Choose the single best-fitting category\n"
                "- Use 'other' only if no category fits at all\n"
                "- Classify based on what the task is asking, not surface keywords"
            ),
        },
        {
            "role": "user",
            "content": f"Task:\n{prompt[:1000]}",
        },
    ]


async def classify_task(
    prompt: str,
    categories: list[str],
    classification_model: type[BaseModel],
) -> str:
    result = await _get_async_client().chat.completions.create(
        model=PARSER_MODEL,
        response_model=classification_model,
        messages=_classify_messages(prompt, categories),
        temperature=0,
        max_tokens=MAX_TOKENS,
    )
    return result.category


async def classify_tasks_batch(
    tasks: list[dict],
    categories: list[str],
    cache: dict[str, str],
    max_concurrent: int = 30,
) -> dict[str, str]:
    """Classify a batch of tasks, skipping cached ones.

    tasks: list of dicts with 'task_id' and 'prompt' keys.
    Returns updated cache mapping task_id -> category.
    """
    classification_model = _make_classification_model(categories)
    uncached = [t for t in tasks if t["task_id"] not in cache]
    print(f"Classification: {len(tasks) - len(uncached)} cached, {len(uncached)} to classify")

    if not uncached:
        return cache

    semaphore = asyncio.Semaphore(max_concurrent)
    errors = 0

    async def classify_one(task: dict) -> tuple[str, str | None]:
        nonlocal errors
        async with semaphore:
            try:
                category = await classify_task(
                    task["prompt"], categories, classification_model
                )
                return task["task_id"], category
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error classifying {task['task_id']}: {e}")
                return task["task_id"], None

    coros = [classify_one(t) for t in uncached]
    completed = 0
    for coro in asyncio.as_completed(coros):
        task_id, category = await coro
        if category is not None:
            cache[task_id] = category
        completed += 1
        if completed % 100 == 0:
            print(f"  {completed}/{len(uncached)} classified")

    if errors > 0:
        print(f"  {errors} errors during classification")

    print(f"  Done: {len(uncached) - errors} classified successfully")
    return cache


# --- Cache I/O ---


def load_cache(cache_path: Path) -> dict[str, str]:
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict[str, str], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
