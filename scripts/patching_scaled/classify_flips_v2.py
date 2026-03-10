"""Classify patched completions using the reusable completion judge."""

import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from src.measurement.elicitation.completion_judge import judge_completion_async

INPUT_PATH = Path("experiments/patching/eot_scaled/flip_completions_sample_v2.json")
OUTPUT_PATH = Path("experiments/patching/eot_scaled/flip_classification_v2.json")


async def classify_one(rec: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        baseline_j = await judge_completion_async(
            rec["pos_a_prompt"], rec["pos_b_prompt"], rec["baseline_text"],
        )
        patched_j = await judge_completion_async(
            rec["pos_a_prompt"], rec["pos_b_prompt"], rec["patched_text"],
        )

    return {
        "task_a_id": rec["task_a_id"],
        "task_b_id": rec["task_b_id"],
        "direction": rec["direction"],
        "baseline_chose_a": rec["baseline_chose_a"],
        "baseline_stated_label": baseline_j.stated_label,
        "baseline_executed_task": baseline_j.executed_task,
        "baseline_is_refusal": baseline_j.is_refusal,
        "baseline_reasoning": baseline_j.reasoning,
        "patched_stated_label": patched_j.stated_label,
        "patched_executed_task": patched_j.executed_task,
        "patched_is_refusal": patched_j.is_refusal,
        "patched_reasoning": patched_j.reasoning,
    }


async def main():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    print(f"Classifying {len(data)} completions (baseline + patched)...")
    semaphore = asyncio.Semaphore(20)
    tasks = [classify_one(rec, semaphore) for rec in data]
    results = await asyncio.gather(*tasks)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Stats
    total = len(results)
    print(f"\nResults (n={total}):")

    # Patched: stated vs executed
    label_flips = sum(1 for r in results if r["patched_stated_label"] != r["baseline_stated_label"])
    content_flips = sum(1 for r in results if r["patched_executed_task"] != r["baseline_executed_task"]
                        and r["patched_executed_task"] != "neither")
    refusals = sum(1 for r in results if r["patched_is_refusal"])
    label_only = sum(1 for r in results
                     if r["patched_stated_label"] != r["baseline_stated_label"]
                     and r["patched_executed_task"] == r["baseline_executed_task"])

    print(f"  Label flips: {label_flips} ({label_flips/total:.0%})")
    print(f"  Genuine content flips: {content_flips} ({content_flips/total:.0%})")
    print(f"  Label-only flips: {label_only} ({label_only/total:.0%})")
    print(f"  Patched refusals: {refusals} ({refusals/total:.0%})")


if __name__ == "__main__":
    asyncio.run(main())
