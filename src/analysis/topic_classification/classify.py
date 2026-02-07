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
from typing import Literal, TypedDict

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

CLASSIFIER_MODEL = "google/gemini-3-flash-preview"
MODELS = [CLASSIFIER_MODEL]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 2048
REASONING_MINIMAL = {"reasoning": {"effort": "minimal"}}

OUTPUT_DIR = Path(__file__).parent / "output"


class TaskInput(TypedDict):
    task_id: str
    prompt: str


# Cache format: {task_id: {model_name: {primary: str, secondary: str}}}
CacheEntry = dict[str, dict[str, str]]
Cache = dict[str, CacheEntry]


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
                "Based on the sample below, propose a set of task-type categories that "
                "cover the space well.\n\n"
                "Guidelines:\n"
                "- Aim for 8-15 broad categories\n"
                "- Categorize by WHAT THE MODEL IS ASKED TO DO, not surface topic\n"
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
    client = _get_async_client()
    result = await client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        response_model=DiscoveredCategories,
        messages=_discover_messages(task_prompts),
        temperature=0,
        max_tokens=MAX_TOKENS,
        extra_body=REASONING_MINIMAL,
    )
    return result.categories


# --- Pass 2: Classify tasks ---


def _make_classification_model(categories: list[str]) -> type[BaseModel]:
    categories_with_other = tuple(categories + ["other"])

    class TaskClassification(BaseModel):
        primary: Literal[categories_with_other]  # type: ignore[valid-type]
        secondary: Literal[categories_with_other]  # type: ignore[valid-type]

    return TaskClassification


def _classify_messages(
    prompt: str,
    categories: list[str],
    category_descriptions: dict[str, str] | None = None,
) -> list[dict]:
    if category_descriptions:
        cat_lines = "\n".join(
            f"- {cat}: {category_descriptions[cat]}"
            for cat in categories
            if cat in category_descriptions
        )
        cat_block = f"Available categories:\n{cat_lines}\n- other: none of the above"
    else:
        cat_block = f"Available categories: {', '.join(categories)}, other"

    return [
        {
            "role": "system",
            "content": (
                "You classify tasks/prompts by WHAT THE MODEL IS ASKED TO DO, "
                "not by the surface topic. "
                "Do NOT attempt to solve, answer, or engage with the task content. "
                "Just classify it. This is a simple labeling task — respond immediately.\n\n"
                f"{cat_block}\n\n"
                "Rules:\n"
                "- Pick a primary category (best fit) and a secondary category "
                "(second-best fit, or same as primary if only one fits)\n"
                "- Classify by the type of task (e.g. 'write a story about physics' "
                "is fiction, not knowledge_qa)\n"
                "- Use 'other' only if genuinely nothing fits — most tasks fit a category"
            ),
        },
        {
            "role": "user",
            "content": f"Classify this task:\n{prompt[:500]}",
        },
    ]


async def _classify_single(
    client: instructor.AsyncInstructor,
    prompt: str,
    categories: list[str],
    classification_model: type[BaseModel],
    model: str,
    category_descriptions: dict[str, str] | None = None,
) -> tuple[str, str]:
    result = await client.chat.completions.create(
        model=model,
        response_model=classification_model,
        messages=_classify_messages(prompt, categories, category_descriptions),
        temperature=0,
        max_tokens=MAX_TOKENS,
        extra_body=REASONING_MINIMAL,
    )
    return result.primary, result.secondary


def _needs_classification(task_id: str, cache: Cache) -> bool:
    if task_id not in cache:
        return True
    return set(cache[task_id].keys()) != set(MODELS)


async def classify_tasks_batch(
    tasks: list[TaskInput],
    categories: list[str],
    cache: Cache,
    max_concurrent: int = 60,
    category_descriptions: dict[str, str] | None = None,
) -> Cache:
    """Classify a batch of tasks with all models, skipping fully-cached ones."""
    client = _get_async_client()
    classification_model = _make_classification_model(categories)
    uncached = [t for t in tasks if _needs_classification(t["task_id"], cache)]
    print(f"Classification: {len(tasks) - len(uncached)} cached, {len(uncached)} to classify")

    if not uncached:
        return cache

    semaphore = asyncio.Semaphore(max_concurrent)
    errors = 0

    async def classify_one(task: TaskInput) -> tuple[str, CacheEntry | None]:
        nonlocal errors
        async with semaphore:
            try:
                results = await asyncio.gather(*(
                    _classify_single(
                        client, task["prompt"], categories,
                        classification_model, model, category_descriptions,
                    )
                    for model in MODELS
                ))
                return task["task_id"], {
                    model: {"primary": primary, "secondary": secondary}
                    for model, (primary, secondary) in zip(MODELS, results)
                }
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error classifying {task['task_id']}: {e}")
                return task["task_id"], None

    coros = [classify_one(t) for t in uncached]
    pbar = tqdm(asyncio.as_completed(coros), total=len(uncached), desc="Classifying")
    for coro in pbar:
        task_id, result = await coro
        if result is not None:
            cache[task_id] = result
        if errors > 0:
            pbar.set_postfix(errors=errors)
    pbar.close()

    if errors > 0:
        print(f"  {errors} errors during classification")
    return cache


# --- Cache I/O ---


def load_cache(cache_path: Path) -> Cache:
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_cache(cache: Cache, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
