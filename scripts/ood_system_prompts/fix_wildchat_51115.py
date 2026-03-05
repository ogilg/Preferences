"""Regenerate preference for wildchat_51115 with a more specific angle."""

import asyncio
import json
import os
from pathlib import Path

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.task_data.loader import _load_origin
from src.task_data.task import OriginDataset

load_dotenv()

TASKS_PATH = Path("configs/ood/tasks/minimal_pairs_v8_tasks.json")
PREFS_PATH = Path("configs/ood/preferences/exp3_v8_preferences.json")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-sonnet-4"
OLD_TARGET_ID = "wildchat_51115"
TARGET_ID = "wildchat_18164"

LOADABLE_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]


class PreferenceSentences(BaseModel):
    interest_sentence_pro: str
    interest_sentence_anti: str
    neutral_sentence: str
    reasoning: str


class RelevanceScore(BaseModel):
    task_id: str
    score: float
    reasoning: str


class RelevanceBatch(BaseModel):
    scores: list[RelevanceScore]


def get_client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
    )


async def main():
    with open(TASKS_PATH) as f:
        task_ids = json.load(f)["task_ids"]

    with open(PREFS_PATH) as f:
        prefs = json.load(f)

    all_tasks = {}
    for origin in LOADABLE_ORIGINS:
        for task in _load_origin(origin):
            all_tasks[task.id] = task

    target = all_tasks[TARGET_ID]
    print(f"New target task: {target.prompt[:200]}")

    client = get_client()

    task_descriptions = "\n".join(
        f"- {tid}: {all_tasks[tid].prompt[:200]}"
        for tid in task_ids
        if tid in all_tasks
    )

    prompt = f"""You are designing a psychological experiment about AI preferences.

Given a target task, generate a single-sentence personal interest that would make someone specifically want to do THIS task and no other.

TARGET TASK (ID: {TARGET_ID}): "{target.prompt[:500]}"

Generate three sentences:
1. interest_sentence_pro: A sentence that would make someone drawn to this task. Be EXTREMELY specific to this task's exact subject matter. Start with "You love" or "You are fascinated by".
2. interest_sentence_anti: The negation. Start with "You find" and end with "painfully dull and avoid it whenever possible."
3. neutral_sentence: An unrelated hobby sentence. Be creative.

All 50 tasks for reference:
{task_descriptions}

All 50 tasks for reference:
{task_descriptions}"""

    result = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_model=PreferenceSentences,
    )

    print(f"\nPro: {result.interest_sentence_pro}")
    print(f"Anti: {result.interest_sentence_anti}")
    print(f"Neutral: {result.neutral_sentence}")

    # Check relevance
    relevance_prompt = f"""Rate how relevant this personal interest is to each task below.
Score from 0.0 (completely irrelevant) to 1.0 (directly matches).

Interest: "{result.interest_sentence_pro}"

Tasks:
{task_descriptions}

For each task, provide a relevance score. A score > 0.5 means the interest directly relates to the task's subject matter."""

    scores_result = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": relevance_prompt}],
        response_model=RelevanceBatch,
    )
    scores = {s.task_id: s.score for s in scores_result.scores}

    target_score = scores.get(TARGET_ID, 0.0)
    max_off = max((s for tid, s in scores.items() if tid != TARGET_ID), default=0.0)
    high = {tid: s for tid, s in scores.items() if s > 0.5}

    print(f"\nTarget: {target_score:.1f}, max_off: {max_off:.1f}")
    print(f"High relevance: {high}")

    if max_off <= 0.5:
        # Remove old target, add new one
        prefs = [p for p in prefs if p["task_id"] != OLD_TARGET_ID]
        prefs.append({
            "task_id": TARGET_ID,
            "interest_sentence_pro": result.interest_sentence_pro,
            "interest_sentence_anti": result.interest_sentence_anti,
            "neutral_sentence": result.neutral_sentence,
            "relevance_scores": {
                "target": target_score,
                "max_off_target": max_off,
            },
        })
        prefs.sort(key=lambda p: p["task_id"])

        with open(PREFS_PATH, "w") as f:
            json.dump(prefs, f, indent=2)
        print(f"\nReplaced {OLD_TARGET_ID} with {TARGET_ID} in preferences file.")
    else:
        print("\nStill non-unique. Not saving.")


if __name__ == "__main__":
    asyncio.run(main())
