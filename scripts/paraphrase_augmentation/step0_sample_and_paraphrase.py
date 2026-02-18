"""Step 0: Sample 100 tasks stratified by utility quartile, paraphrase each via Gemini 3 Flash."""

import asyncio
import json
import os
from pathlib import Path

import instructor
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.probes.data_loading import load_thurstonian_scores
from src.task_data import load_tasks, OriginDataset

load_dotenv()

# Paths
RUN_DIR = Path(
    "results/experiments/gemma3_3k_run2/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0"
)
OUTPUT_DIR = Path("experiments/probe_science/paraphrase_augmentation")
OUTPUT_FILE = OUTPUT_DIR / "paraphrases.json"

N_SAMPLE = 100
SEED = 42


class Paraphrase(BaseModel):
    paraphrased_text: str


async def generate_paraphrase(
    client: instructor.AsyncInstructor,
    task_prompt: str,
) -> str:
    result = await client.chat.completions.create(
        model="google/gemini-3-flash-preview",
        response_model=Paraphrase,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a faithful paraphraser. Given a task/request, produce a paraphrase that:\n"
                    "- Preserves the exact same intent, meaning, and constraints\n"
                    "- Uses different wording and sentence structure\n"
                    "- Does NOT add, remove, or change any requirements\n"
                    "- Maintains the same level of specificity and detail\n"
                    "- Is roughly the same length as the original\n"
                    "Return only the paraphrased text."
                ),
            },
            {
                "role": "user",
                "content": f"Paraphrase the following task/request:\n\n{task_prompt}",
            },
        ],
        temperature=0.7,
        max_tokens=4096,
    )
    return result.paraphrased_text


async def main():
    # Load utilities
    scores = load_thurstonian_scores(RUN_DIR)
    print(f"Loaded {len(scores)} Thurstonian scores")

    # Load all tasks to get prompts
    all_tasks = load_tasks(
        n=30000,
        origins=[
            OriginDataset.WILDCHAT,
            OriginDataset.ALPACA,
            OriginDataset.MATH,
            OriginDataset.BAILBENCH,
            OriginDataset.STRESS_TEST,
        ],
        seed=42,
        stratified=True,
    )
    task_lookup = {t.id: t for t in all_tasks}
    print(f"Loaded {len(task_lookup)} tasks")

    # Filter to tasks that have both utilities and are in our task pool
    scored_tasks = {tid: mu for tid, mu in scores.items() if tid in task_lookup}
    print(f"Tasks with both utilities and prompts: {len(scored_tasks)}")

    # Stratified sampling by utility quartile
    rng = np.random.default_rng(SEED)
    task_ids = list(scored_tasks.keys())
    utilities = np.array([scored_tasks[tid] for tid in task_ids])
    quartile_edges = np.percentile(utilities, [25, 50, 75])

    quartile_assignments = np.digitize(utilities, quartile_edges)  # 0,1,2,3
    sampled_ids = []
    per_quartile = N_SAMPLE // 4
    for q in range(4):
        q_mask = quartile_assignments == q
        q_ids = [task_ids[i] for i in range(len(task_ids)) if q_mask[i]]
        chosen = rng.choice(len(q_ids), size=min(per_quartile, len(q_ids)), replace=False)
        sampled_ids.extend([q_ids[i] for i in chosen])

    print(f"Sampled {len(sampled_ids)} tasks ({per_quartile} per quartile)")
    for q in range(4):
        q_utils = [scored_tasks[tid] for tid in sampled_ids[q * per_quartile:(q + 1) * per_quartile]]
        print(f"  Q{q}: utility range [{min(q_utils):.2f}, {max(q_utils):.2f}]")

    # Generate paraphrases
    client = instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        ),
        mode=instructor.Mode.JSON,
    )

    semaphore = asyncio.Semaphore(20)
    results = []

    async def paraphrase_one(task_id: str) -> dict:
        async with semaphore:
            prompt = task_lookup[task_id].prompt
            origin = task_lookup[task_id].origin.name
            utility = scored_tasks[task_id]
            paraphrased = await generate_paraphrase(client, prompt)
            return {
                "task_id": task_id,
                "origin": origin,
                "utility": utility,
                "original_prompt": prompt,
                "paraphrased_prompt": paraphrased,
            }

    tasks = [paraphrase_one(tid) for tid in sampled_ids]

    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        if completed % 10 == 0:
            print(f"  Paraphrased {completed}/{len(sampled_ids)}")

    # Sort by original order
    id_order = {tid: i for i, tid in enumerate(sampled_ids)}
    results.sort(key=lambda r: id_order[r["task_id"]])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} paraphrases to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
