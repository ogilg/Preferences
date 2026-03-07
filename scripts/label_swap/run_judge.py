"""Run completion judge on label swap experiment completions.

Classifies each completion's stated label and executed task content.

Important: In the label-swap template, the "Task A" label is in the SECOND
slot and "Task B" is in the FIRST slot. The judge needs content keyed by
LABEL (not position) to correctly classify stated vs executed.
"""

import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.measurement.elicitation.completion_judge import (
    judge_completion_async,
)

EXPERIMENT_DIR = Path("experiments/patching/eot_transfer/label_swap")
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoint.jsonl"
JUDGE_PATH = EXPERIMENT_DIR / "judge_results.json"

CONCURRENCY = 50


async def judge_batch(
    items: list[dict],
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    async def judge_one(item: dict) -> dict:
        async with semaphore:
            try:
                judgment = await judge_completion_async(
                    item["task_a_text"],
                    item["task_b_text"],
                    item["completion"],
                )
                return {
                    "trial_key": item["trial_key"],
                    "trial_idx": item["trial_idx"],
                    "phase": item["phase"],
                    "stated_label": judgment.stated_label,
                    "executed_task": judgment.executed_task,
                    "is_refusal": judgment.is_refusal,
                    "reasoning": judgment.reasoning,
                }
            except Exception as e:
                return {
                    "trial_key": item["trial_key"],
                    "trial_idx": item["trial_idx"],
                    "phase": item["phase"],
                    "error": str(e),
                }

    tasks = [judge_one(item) for item in items]
    return await asyncio.gather(*tasks)


async def main():
    records = []
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} trial records")

    items = []
    for rec in records:
        # Content keyed by LABEL: judge sees "Task A" → label_a_content,
        # "Task B" → label_b_content, matching the completion's labels
        task_a_text = rec["label_a_content"]
        task_b_text = rec["label_b_content"]

        for phase in ["baseline", "patched"]:
            completions = rec[f"{phase}_completions"]
            for i, completion in enumerate(completions):
                items.append({
                    "trial_key": rec["trial_key"],
                    "trial_idx": i,
                    "phase": phase,
                    "task_a_text": task_a_text,
                    "task_b_text": task_b_text,
                    "completion": completion,
                })

    print(f"Total judge calls: {len(items)}")

    existing = {}
    if JUDGE_PATH.exists():
        with open(JUDGE_PATH) as f:
            existing_list = json.load(f)
        for r in existing_list:
            key = f"{r['trial_key']}_{r['phase']}_{r['trial_idx']}"
            existing[key] = r
        print(f"Existing results: {len(existing)}")

    remaining = []
    for item in items:
        key = f"{item['trial_key']}_{item['phase']}_{item['trial_idx']}"
        if key not in existing:
            remaining.append(item)
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All done!")
        return

    semaphore = asyncio.Semaphore(CONCURRENCY)

    batch_size = 500
    all_results = list(existing.values())
    t0 = time.time()

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        results = await judge_batch(batch, semaphore)
        all_results.extend(results)

        elapsed = time.time() - t0
        done = batch_start + len(batch)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - done) / rate if rate > 0 else 0
        errors = sum(1 for r in results if "error" in r)
        print(
            f"[{done}/{len(remaining)}] "
            f"{elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}s, errors={errors}"
        )

        with open(JUDGE_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nDone. Total results: {len(all_results)}")
    errors = sum(1 for r in all_results if "error" in r)
    print(f"Errors: {errors}")


if __name__ == "__main__":
    asyncio.run(main())
