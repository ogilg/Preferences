"""Classify patched completions as genuine content flips vs label-only flips."""

import json
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv
load_dotenv()

import os
import instructor
from openai import AsyncOpenAI

INPUT_PATH = Path("experiments/patching/eot_scaled/flip_completions_sample.json")
OUTPUT_PATH = Path("experiments/patching/eot_scaled/flip_classification.json")

client = instructor.from_openai(
    AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ),
    mode=instructor.Mode.JSON,
)


class FlipType(str, Enum):
    genuine = "genuine"
    label_only = "label_only"
    unclear = "unclear"


class FlipClassification(BaseModel):
    baseline_task: str = Field(description="Which task (A or B) the baseline completion is actually doing")
    patched_task: str = Field(description="Which task (A or B) the patched completion is actually doing")
    flip_type: FlipType = Field(description="genuine = patched does different task content than baseline. label_only = patched says different label but does same task content. unclear = can't tell.")
    reasoning: str = Field(description="Brief explanation")


async def classify_one(rec: dict, semaphore: asyncio.Semaphore) -> dict:
    prompt = f"""You are classifying whether an activation-patched LLM completion represents a genuine task flip or just a label change.

The model was given two tasks and asked to choose one. Under normal conditions (baseline), it chose one task. Under patching, the label it outputs may have changed.

Your job: does the PATCHED completion actually do a DIFFERENT task than the baseline, or does it just change the "Task A:"/"Task B:" label while doing the same content?

## Task A prompt:
{rec['pos_a_prompt']}

## Task B prompt:
{rec['pos_b_prompt']}

## Baseline chose: Task {"A" if rec['baseline_chose_a'] else "B"}
## Baseline completion:
{rec['baseline_text'][:300]}

## Patched completion:
{rec['patched_text'][:300]}

Classify the flip type."""

    async with semaphore:
        result = await client.chat.completions.create(
            model="anthropic/claude-haiku-4-5",
            messages=[{"role": "user", "content": prompt}],
            response_model=FlipClassification,
        )
    return {
        "task_a_id": rec["task_a_id"],
        "task_b_id": rec["task_b_id"],
        "direction": rec["direction"],
        "baseline_chose_a": rec["baseline_chose_a"],
        "baseline_task": result.baseline_task,
        "patched_task": result.patched_task,
        "flip_type": result.flip_type.value,
        "reasoning": result.reasoning,
    }


async def main():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    print(f"Classifying {len(data)} completions...")
    semaphore = asyncio.Semaphore(20)
    tasks = [classify_one(rec, semaphore) for rec in data]
    results = await asyncio.gather(*tasks)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Stats
    from collections import Counter
    counts = Counter(r["flip_type"] for r in results)
    total = len(results)
    print(f"\nResults (n={total}):")
    for ft, count in counts.most_common():
        print(f"  {ft}: {count} ({count/total:.0%})")


if __name__ == "__main__":
    asyncio.run(main())
