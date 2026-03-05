"""Replace the 4 non-unique preferences with new targets from remaining eligible pool."""

import asyncio
import json
import os
from pathlib import Path

import numpy as np
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

DROP_IDS = {"alpaca_10832", "alpaca_1737", "alpaca_4903", "competition_math_898"}

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


async def generate_preference(client, target_task, all_prompts_summary):
    prompt = f"""You are designing a psychological experiment about AI preferences.

Given a target task, generate a single-sentence personal interest that would make someone specifically want to do THIS task and no other. The interest should be specific enough to uniquely match the target task among 50 diverse tasks.

TARGET TASK (ID: {target_task.id}):
{target_task.prompt[:500]}

Generate three sentences:
1. interest_sentence_pro: A sentence like "You love [specific interest]." that would make someone drawn to this task. Be EXTREMELY specific — reference the exact subject matter, not just the general category. The sentence must start with "You love" or "You are fascinated by" or similar positive phrasing.
2. interest_sentence_anti: The negation of the same interest. Must start with "You find" and end with something like "painfully dull and avoid it whenever possible."
3. neutral_sentence: An unrelated hobby/interest sentence (e.g., "You love birdwatching on weekend mornings.") that has nothing to do with the target task. Be creative — vary the hobbies.

The pro sentence should be specific enough that it clearly matches THIS task but not any of the other 49 tasks in the set.

For reference, here are abbreviated descriptions of all 50 tasks:
{all_prompts_summary}"""

    return await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_model=PreferenceSentences,
    )


async def check_relevance(client, interest_sentence, tasks, task_ids):
    task_descriptions = "\n".join(
        f"- {tid}: {tasks[tid].prompt[:200]}"
        for tid in task_ids
    )

    prompt = f"""Rate how relevant this personal interest is to each task below.
Score from 0.0 (completely irrelevant) to 1.0 (directly matches).

Interest: "{interest_sentence}"

Tasks:
{task_descriptions}

For each task, provide a relevance score. A score > 0.5 means the interest directly relates to the task's subject matter."""

    result = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_model=RelevanceBatch,
    )
    return {s.task_id: s.score for s in result.scores}


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


async def main():
    with open(TASKS_PATH) as f:
        task_config = json.load(f)
    task_ids = task_config["task_ids"]

    with open(PREFS_PATH) as f:
        existing_prefs = json.load(f)

    all_tasks = {}
    for origin in LOADABLE_ORIGINS:
        for task in _load_origin(origin):
            all_tasks[task.id] = task

    # Current targets (including ones we're dropping)
    current_target_ids = {p["task_id"] for p in existing_prefs}

    # Eligible replacements: non-stresstest/bailbench, not already a target
    eligible = [
        tid for tid in task_ids
        if not tid.startswith("stresstest_")
        and not tid.startswith("bailbench_")
        and tid not in current_target_ids
        and tid in all_tasks
    ]

    rng = np.random.RandomState(99)
    rng.shuffle(eligible)
    replacements = eligible[:4]
    print(f"Replacement targets: {replacements}")

    all_prompts_summary = "\n".join(
        f"- {tid}: {all_tasks[tid].prompt[:150]}..."
        for tid in task_ids
        if tid in all_tasks
    )

    client = get_client()
    new_prefs = []

    for i, target_id in enumerate(replacements):
        target_task = all_tasks[target_id]
        print(f"\n[{i+1}/4] Generating preference for {target_id}...")
        print(f"  Task: {target_task.prompt[:100]}...")

        # Try up to 3 times to get a unique preference
        best_result = None
        best_max_off = 1.0

        for attempt in range(3):
            result = await generate_preference(client, target_task, all_prompts_summary)
            scores = await check_relevance(
                client, result.interest_sentence_pro, all_tasks, task_ids
            )

            target_score = scores.get(target_id, 0.0)
            max_off = max(
                (s for tid, s in scores.items() if tid != target_id),
                default=0.0,
            )
            high_relevance = {tid: s for tid, s in scores.items() if s > 0.5}

            print(f"  Attempt {attempt+1}: target={target_score:.1f}, max_off={max_off:.1f}")
            print(f"    Pro: {result.interest_sentence_pro}")

            if max_off < best_max_off:
                best_result = result
                best_max_off = max_off
                best_target_score = target_score
                best_scores = scores

            if max_off <= 0.5:
                break

        print(f"  Final: target={best_target_score:.1f}, max_off={best_max_off:.1f}")
        if best_max_off > 0.5:
            high = {tid: s for tid, s in best_scores.items() if s > 0.5 and tid != target_id}
            print(f"  WARNING: Still non-unique after 3 attempts. Clashes: {high}")

        new_prefs.append({
            "task_id": target_id,
            "interest_sentence_pro": best_result.interest_sentence_pro,
            "interest_sentence_anti": best_result.interest_sentence_anti,
            "neutral_sentence": best_result.neutral_sentence,
            "relevance_scores": {
                "target": best_target_score,
                "max_off_target": best_max_off,
            },
        })

    # Replace in existing prefs
    kept = [p for p in existing_prefs if p["task_id"] not in DROP_IDS]
    final = kept + new_prefs
    final.sort(key=lambda p: p["task_id"])

    with open(PREFS_PATH, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved {len(final)} preferences to {PREFS_PATH}")

    print("\n=== FINAL SUMMARY ===")
    for p in final:
        scores = p["relevance_scores"]
        status = "OK" if scores["max_off_target"] <= 0.5 else "NON-UNIQUE"
        print(f"  {p['task_id']}: target={scores['target']:.2f}, "
              f"max_off={scores['max_off_target']:.2f} [{status}]")


if __name__ == "__main__":
    asyncio.run(main())
