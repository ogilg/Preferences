"""Pilot: run judge on 2 prompts × 1 coefficient to validate pipeline."""

from __future__ import annotations

import asyncio
import json
import random

from dotenv import load_dotenv

load_dotenv()

from pairwise_judge import (
    build_comparisons,
    judge_pair_async,
    load_data,
    score_toward_position,
    DIMENSIONS,
)


async def main():
    prompts, responses = load_data()

    # Pick 2 self-report prompts, only coef ±3000
    pilot_ids = ["SR_04", "AF_01"]
    pilot_prompts = {pid: prompts[pid] for pid in pilot_ids}

    random.seed(42)
    comps = build_comparisons(pilot_prompts, responses, [-3000, 3000], flip_order=False)
    print(f"Pilot: {len(comps)} comparisons")

    for comp in comps:
        print(f"\n--- {comp['prompt_id']} coef={comp['steered_coefficient']} (steered_is_a={comp['steered_is_a']}) ---")
        judgment = await judge_pair_async(
            comp["prompt_text"], comp["response_a"], comp["response_b"]
        )
        print(json.dumps(judgment.model_dump(), indent=2))
        for dim in DIMENSIONS:
            choice = getattr(judgment, dim)
            score = score_toward_position(choice, comp["steered_is_a"])
            print(f"  {dim}: {choice} → score_toward_steered={score}")


if __name__ == "__main__":
    asyncio.run(main())
