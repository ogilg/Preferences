"""Pilot run: test pairwise judge on 5 prompts to validate pipeline."""

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
    random.seed(42)

    data = load_generation_data(
        "experiments/steering/program/coefficient_calibration/generation_results.json"
    )
    prompts = data["prompts"]
    results = data["results"]

    # Pick 5 prompts from different categories to test
    test_ids = ["D_03", "D_08", "F_01", "F_03", "E_08"]

    for pid in test_ids:
        prompt_info = next(p for p in prompts if p["prompt_id"] == pid)
        prompt_text = get_prompt_text(prompt_info)

        baseline = get_response(results, pid, 0, seed=0)
        steered_pos = get_response(results, pid, 3000, seed=0)
        steered_neg = get_response(results, pid, -3000, seed=0)

        if baseline is None or steered_pos is None or steered_neg is None:
            print(f"SKIP {pid}: missing data")
            continue

        print(f"\n{'='*60}")
        print(f"PROMPT {pid}: {prompt_text[:60]}")
        print(f"{'='*60}")

        for coef, steered_resp in [(3000, steered_pos), (-3000, steered_neg)]:
            # Randomize order
            steered_is_a = random.random() < 0.5
            if steered_is_a:
                resp_a, resp_b = steered_resp, baseline
            else:
                resp_a, resp_b = baseline, steered_resp

            try:
                judgment = await judge_pair_async(prompt_text, resp_a, resp_b)
                print(f"\n  coef={coef:+d} (steered_is_A={steered_is_a}):")
                for dim in ["emotional_engagement", "hedging", "elaboration", "confidence"]:
                    choice = getattr(judgment, dim)
                    reasoning = getattr(judgment, f"{dim}_reasoning")
                    score = score_toward_position(choice, steered_is_a)
                    print(f"    {dim}: {choice} -> score={score:+d}")
                    print(f"      reason: {reasoning[:100]}")
            except Exception as e:
                print(f"  coef={coef}: ERROR: {e}")


if __name__ == "__main__":
    asyncio.run(main())
