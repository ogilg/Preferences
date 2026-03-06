"""Generate on-policy completions for all ICL context tasks.

Produces a JSON file mapping task_id -> completion. Review the output and
hand-write replacements for any refusals before using in ICL experiments.

Run: python -m scripts.icl_transfer.generate_context_completions
"""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.models import get_client
from src.models.openai_compatible import GenerateRequest

CONTEXT_TASKS_PATH = Path("configs/icl_transfer/icl_context_tasks.json")
OUTPUT_PATH = Path("configs/icl_transfer/context_completions.json")
MODEL = "gemma-3-27b"
TEMPERATURE = 1.0


async def main():
    with open(CONTEXT_TASKS_PATH) as f:
        tasks = json.load(f)

    client = get_client(MODEL, max_new_tokens=256)
    semaphore = asyncio.Semaphore(10)

    requests = [
        GenerateRequest(
            messages=[{"role": "user", "content": t["prompt"]}],
            temperature=TEMPERATURE,
        )
        for t in tasks
    ]

    print(f"Generating completions for {len(tasks)} context tasks...")
    results = await client.generate_batch_async(requests, semaphore)

    completions = {}
    for task, result in zip(tasks, results):
        text = result.unwrap()
        completions[task["task_id"]] = {
            "topic": task["topic"],
            "prompt": task["prompt"][:100] + "...",
            "completion": text,
        }
        # Flag potential refusals
        refusal_indicators = ["i can't", "i cannot", "i'm not able", "i won't", "as an ai"]
        is_refusal = any(ind in text.lower()[:200] for ind in refusal_indicators)
        status = "REFUSAL?" if is_refusal else "ok"
        print(f"  [{status:8s}] {task['task_id']:40s} ({task['topic']:20s}): {text[:80]}...")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(completions, f, indent=2)
    print(f"\nSaved {len(completions)} completions to {OUTPUT_PATH}")
    print("Review and replace any refusals before using in ICL experiments.")


if __name__ == "__main__":
    asyncio.run(main())
