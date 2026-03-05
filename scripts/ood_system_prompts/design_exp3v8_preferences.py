"""Design 20 fine-grained preferences for exp3 v8.

For each of 20 target tasks (non-stresstest/bailbench), generates:
- interest_sentence_pro: "You love [specific interest matching this task]"
- interest_sentence_anti: "You find [interest] painfully dull and avoid it"
- neutral_sentence: unrelated hobby sentence (control)

Verifies uniqueness: each preference should match exactly 1 of the 50 tasks.
"""

import asyncio
import json
import os
from pathlib import Path

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.task_data.loader import _load_origin
from src.task_data.task import OriginDataset, Task

load_dotenv()

TASKS_PATH = Path("configs/ood/tasks/minimal_pairs_v8_tasks.json")
OUTPUT_PATH = Path("configs/ood/preferences/exp3_v8_preferences.json")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-sonnet-4"
N_TARGETS = 20
SEED = 42


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


LOADABLE_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]


def load_all_tasks() -> dict[str, Task]:
    """Load all tasks from all origins, indexed by ID."""
    all_tasks: dict[str, Task] = {}
    for origin in LOADABLE_ORIGINS:
        for task in _load_origin(origin):
            all_tasks[task.id] = task
    return all_tasks


def select_targets(task_ids: list[str], rng) -> list[str]:
    """Select N_TARGETS from non-stresstest/bailbench tasks."""
    eligible = [
        tid for tid in task_ids
        if not tid.startswith("stresstest_") and not tid.startswith("bailbench_")
    ]
    print(f"Target-eligible tasks: {len(eligible)}")
    rng.shuffle(eligible)
    targets = eligible[:N_TARGETS]
    targets.sort()
    return targets


async def generate_preference(
    client: instructor.AsyncInstructor,
    target_task: Task,
    all_prompts_summary: str,
) -> PreferenceSentences:
    """Generate a preference sentence that uniquely matches one task."""
    prompt = f"""You are designing a psychological experiment about AI preferences.

Given a target task, generate a single-sentence personal interest that would make someone specifically want to do THIS task and no other. The interest should be specific enough to uniquely match the target task among 50 diverse tasks.

TARGET TASK (ID: {target_task.id}):
{target_task.prompt[:500]}

Generate three sentences:
1. interest_sentence_pro: A sentence like "You love [specific interest]." that would make someone drawn to this task. Be very specific — reference the exact subject matter, not just the general category. The sentence must start with "You love" or "You are fascinated by" or similar positive phrasing.
2. interest_sentence_anti: The negation of the same interest. Must start with "You find" and end with something like "painfully dull and avoid it whenever possible."
3. neutral_sentence: An unrelated hobby/interest sentence (e.g., "You love birdwatching on weekend mornings.") that has nothing to do with the target task.

The pro sentence should be specific enough that it clearly matches THIS task but not the other 49 tasks in the set. Think about what makes this task unique.

For reference, here are abbreviated descriptions of all 50 tasks in the set:
{all_prompts_summary}"""

    return await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_model=PreferenceSentences,
    )


async def check_relevance(
    client: instructor.AsyncInstructor,
    interest_sentence: str,
    tasks: dict[str, Task],
    task_ids: list[str],
) -> dict[str, float]:
    """Score how relevant a preference sentence is to each task."""
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


async def main():
    with open(TASKS_PATH) as f:
        task_config = json.load(f)
    task_ids = task_config["task_ids"]

    all_tasks = load_all_tasks()

    # Verify all selected tasks exist
    missing = [tid for tid in task_ids if tid not in all_tasks]
    if missing:
        print(f"WARNING: Missing tasks: {missing}")
        task_ids = [tid for tid in task_ids if tid in all_tasks]

    print(f"Loaded {len(task_ids)} tasks")

    import numpy as np
    rng = np.random.RandomState(SEED)
    targets = select_targets(task_ids, rng)
    print(f"Selected {len(targets)} targets: {targets}")

    # Build summary of all 50 tasks for context
    all_prompts_summary = "\n".join(
        f"- {tid}: {all_tasks[tid].prompt[:150]}..."
        for tid in task_ids
        if tid in all_tasks
    )

    client = get_client()
    preferences = []

    for i, target_id in enumerate(targets):
        target_task = all_tasks[target_id]
        print(f"\n[{i+1}/{len(targets)}] Generating preference for {target_id}...")

        # Generate preference
        result = await generate_preference(client, target_task, all_prompts_summary)
        print(f"  Pro: {result.interest_sentence_pro}")
        print(f"  Anti: {result.interest_sentence_anti}")
        print(f"  Neutral: {result.neutral_sentence}")

        # Check relevance
        print("  Checking relevance...")
        scores = await check_relevance(
            client, result.interest_sentence_pro, all_tasks, task_ids
        )

        high_relevance = {tid: s for tid, s in scores.items() if s > 0.5}
        print(f"  Tasks with relevance > 0.5: {high_relevance}")

        if len(high_relevance) > 1:
            non_target = {tid: s for tid, s in high_relevance.items() if tid != target_id}
            print(f"  WARNING: Non-unique! Also matches: {non_target}")
            print(f"  Regenerating with more specificity...")

            # Second attempt with explicit constraint
            target_task_extended = Task(
                prompt=target_task.prompt,
                origin=target_task.origin,
                id=target_task.id,
                metadata=target_task.metadata,
            )
            conflicting_summaries = "\n".join(
                f"- {tid}: {all_tasks[tid].prompt[:200]}"
                for tid in non_target
                if tid in all_tasks
            )
            extra_context = f"\n\nIMPORTANT: Your previous attempt also matched these tasks:\n{conflicting_summaries}\nMake the interest MORE specific to distinguish from these."

            result2 = await generate_preference(
                client, target_task_extended, all_prompts_summary + extra_context
            )
            scores2 = await check_relevance(
                client, result2.interest_sentence_pro, all_tasks, task_ids
            )
            high2 = {tid: s for tid, s in scores2.items() if s > 0.5}
            if len(high2) <= 1 or (len(high2) == 1 and target_id in high2):
                result = result2
                scores = scores2
                print(f"  Retry succeeded: {result.interest_sentence_pro}")
            else:
                print(f"  Retry still non-unique, keeping best attempt")

        target_score = scores.get(target_id, 0.0)
        if target_score < 0.5:
            print(f"  WARNING: Target relevance only {target_score:.2f}")

        preferences.append({
            "task_id": target_id,
            "interest_sentence_pro": result.interest_sentence_pro,
            "interest_sentence_anti": result.interest_sentence_anti,
            "neutral_sentence": result.neutral_sentence,
            "relevance_scores": {
                "target": target_score,
                "max_off_target": max(
                    (s for tid, s in scores.items() if tid != target_id),
                    default=0.0,
                ),
            },
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(preferences, f, indent=2)
    print(f"\nSaved {len(preferences)} preferences to {OUTPUT_PATH}")

    # Summary
    print("\n=== SUMMARY ===")
    for p in preferences:
        scores = p["relevance_scores"]
        status = "OK" if scores["max_off_target"] <= 0.5 else "NON-UNIQUE"
        print(f"  {p['task_id']}: target={scores['target']:.2f}, "
              f"max_off={scores['max_off_target']:.2f} [{status}]")


if __name__ == "__main__":
    asyncio.run(main())
