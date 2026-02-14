"""Position-swapped replication of the ±3000 pairwise comparison.

Reruns the same comparisons but with the A/B order deterministically flipped
from the original. If results replicate, the effect is real (not position bias).
"""

from __future__ import annotations

import asyncio
import json
import random

from dotenv import load_dotenv

load_dotenv()

from pairwise_judge import (
    get_prompt_text,
    get_response,
    judge_pair_async,
    load_generation_data,
    score_toward_position,
)


async def main():
    # Use same seed as original to get same randomization, then flip
    random.seed(42)

    data = load_generation_data(
        "experiments/steering/program/coefficient_calibration/generation_results.json"
    )
    prompts = data["prompts"]
    results_data = data["results"]

    comparisons = []
    for prompt_info in prompts:
        pid = prompt_info["prompt_id"]
        prompt_text = get_prompt_text(prompt_info)
        baseline = get_response(results_data, pid, 0, seed=0)
        if baseline is None:
            continue

        for coef in [-3000, 3000]:
            steered = get_response(results_data, pid, coef, seed=0)
            if steered is None:
                continue

            # Same random call as original to stay in sync
            original_steered_is_a = random.random() < 0.5
            # FLIP the order
            steered_is_a = not original_steered_is_a

            if steered_is_a:
                resp_a, resp_b = steered, baseline
            else:
                resp_a, resp_b = baseline, steered

            comparisons.append({
                "prompt_id": pid,
                "category": prompt_info["category"],
                "prompt_text": prompt_text,
                "steered_coefficient": coef,
                "steered_is_a": steered_is_a,
                "response_a": resp_a,
                "response_b": resp_b,
            })

    print(f"Running {len(comparisons)} position-swapped replications...")

    semaphore = asyncio.Semaphore(5)

    async def judge_one(comp):
        async with semaphore:
            try:
                judgment = await judge_pair_async(
                    comp["prompt_text"], comp["response_a"], comp["response_b"]
                )
                result = {
                    "prompt_id": comp["prompt_id"],
                    "category": comp["category"],
                    "steered_coefficient": comp["steered_coefficient"],
                    "steered_is_a": comp["steered_is_a"],
                    "judgment": judgment.model_dump(),
                }
                for dim in ["emotional_engagement", "hedging", "elaboration", "confidence"]:
                    choice = getattr(judgment, dim)
                    result[f"{dim}_score"] = score_toward_position(
                        choice, comp["steered_is_a"]
                    )
                return result
            except Exception as e:
                return {
                    "prompt_id": comp["prompt_id"],
                    "category": comp["category"],
                    "steered_coefficient": comp["steered_coefficient"],
                    "error": str(e),
                }

    tasks = [judge_one(c) for c in comparisons]
    judge_results = await asyncio.gather(*tasks)

    successes = [r for r in judge_results if "error" not in r]
    errors = [r for r in judge_results if "error" in r]
    print(f"Completed: {len(successes)} successes, {len(errors)} errors")

    output_path = "scripts/pairwise_llm_comparison/judge_results_swapped.json"
    with open(output_path, "w") as f:
        json.dump(judge_results, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
