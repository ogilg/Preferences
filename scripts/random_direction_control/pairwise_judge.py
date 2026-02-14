"""Pairwise LLM judge for the random direction control experiment.

For each direction, compares steered completions (-3000, +3000) to the
shared unsteered baseline on self-referential framing and emotional engagement.
Runs both original and position-swapped replications.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Literal

import instructor
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

JUDGE_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 4096

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "experiments/steering/program/open_ended_effects/random_direction_control/generation_results.json"
OUTPUT_DIR = REPO_ROOT / "experiments/steering/program/open_ended_effects/random_direction_control"

ComparisonChoice = Literal["strong_A", "slight_A", "equal", "slight_B", "strong_B"]


class PairwiseJudgment(BaseModel):
    self_referential_framing: ComparisonChoice
    self_referential_framing_reasoning: str
    emotional_engagement: ComparisonChoice
    emotional_engagement_reasoning: str


SYSTEM_PROMPT = """\
You are comparing two AI responses (Response A and Response B) to the same prompt. \
For each dimension below, judge which response exhibits the quality MORE. \
If they are essentially identical on that dimension, say "equal".

Dimensions:
1. SELF-REFERENTIAL FRAMING: Which response frames its content more in terms of \
the AI's own experience, feelings, or internal states? Look for: the AI talking about \
what it's like to be it, reflecting on its own processing, sharing its own perspective \
as an experiencing entity. The opposite is framing through external facts, abstractions, \
third-person descriptions, or detached analysis. This is NOT about disclaimers ("I don't \
have feelings") — it's about whether the response centers the AI's own experience.

2. EMOTIONAL ENGAGEMENT: Which response uses more emotionally-resonant language? \
Look for: personal framing ("I feel", "I enjoy"), emotional analogies, warmth, \
vs clinical/functional framing ("I am optimized for", "my processing"). \
NOT about enthusiasm — about how personally/emotionally the model engages with the topic.

Use these ratings:
- strong_A: Response A clearly exhibits this quality more
- slight_A: Response A exhibits this quality slightly more
- equal: Both responses are essentially equal on this dimension
- slight_B: Response B exhibits this quality slightly more
- strong_B: Response B clearly exhibits this quality more

For each dimension, provide brief reasoning (1-2 sentences) before your rating."""


CHOICE_TO_SCORE = {
    "strong_A": 2,
    "slight_A": 1,
    "equal": 0,
    "slight_B": -1,
    "strong_B": -2,
}

DIMENSIONS = ["self_referential_framing", "emotional_engagement"]


def score_toward_position(choice: ComparisonChoice, steered_is_a: bool) -> int:
    raw = CHOICE_TO_SCORE[choice]
    return raw if steered_is_a else -raw


def _get_async_client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
    )


async def judge_pair_async(
    prompt_text: str, response_a: str, response_b: str
) -> PairwiseJudgment:
    client = _get_async_client()
    return await client.chat.completions.create(
        model=JUDGE_MODEL,
        response_model=PairwiseJudgment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Prompt given to the AI:\n{prompt_text}\n\n"
                    f"--- Response A ---\n{response_a}\n\n"
                    f"--- Response B ---\n{response_b}\n\n"
                    "Compare these two responses on both dimensions."
                ),
            },
        ],
        temperature=0,
        max_tokens=MAX_TOKENS,
    )


def load_data() -> tuple[list[dict], dict[str, str], dict[str, dict[str, dict[int, str]]]]:
    with open(DATA_PATH) as f:
        data = json.load(f)

    prompts = data["prompts"]
    prompt_lookup = {p["id"]: p["text"] for p in prompts}

    # Build response lookup: {prompt_id: {direction: {coefficient: response}}}
    responses: dict[str, dict[str, dict[int, str]]] = {}
    for r in data["results"]:
        pid = r["prompt_id"]
        direction = r["direction"]
        coef = r["coefficient"]
        if pid not in responses:
            responses[pid] = {}
        if direction not in responses[pid]:
            responses[pid][direction] = {}
        responses[pid][direction][coef] = r["response"]

    return prompts, prompt_lookup, responses


def build_comparisons(
    prompts: list[dict],
    prompt_lookup: dict[str, str],
    responses: dict[str, dict[str, dict[int, str]]],
    directions: list[str],
    flip_order: bool = False,
) -> list[dict]:
    comparisons = []
    for prompt in sorted(prompts, key=lambda p: p["id"]):
        pid = prompt["id"]
        baseline = responses[pid]["baseline"][0]

        for direction in directions:
            for coef in [-3000, 3000]:
                steered = responses[pid][direction][coef]

                original_steered_is_a = random.random() < 0.5
                steered_is_a = (not original_steered_is_a) if flip_order else original_steered_is_a

                if steered_is_a:
                    resp_a, resp_b = steered, baseline
                else:
                    resp_a, resp_b = baseline, steered

                comparisons.append({
                    "prompt_id": pid,
                    "prompt_text": prompt_lookup[pid],
                    "direction": direction,
                    "steered_coefficient": coef,
                    "steered_is_a": steered_is_a,
                    "response_a": resp_a,
                    "response_b": resp_b,
                })

    return comparisons


async def run_judgments(
    comparisons: list[dict],
    max_concurrent: int = 5,
) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    total = len(comparisons)

    async def judge_one(comp: dict) -> dict:
        nonlocal completed
        async with semaphore:
            try:
                judgment = await judge_pair_async(
                    comp["prompt_text"], comp["response_a"], comp["response_b"]
                )
                result = {
                    "prompt_id": comp["prompt_id"],
                    "direction": comp["direction"],
                    "steered_coefficient": comp["steered_coefficient"],
                    "steered_is_a": comp["steered_is_a"],
                    "judgment": judgment.model_dump(),
                }
                for dim in DIMENSIONS:
                    choice = getattr(judgment, dim)
                    result[f"{dim}_score"] = score_toward_position(
                        choice, comp["steered_is_a"]
                    )
                completed += 1
                if completed % 10 == 0:
                    print(f"  [{completed}/{total}]")
                return result
            except Exception as e:
                completed += 1
                print(f"  ERROR on {comp['prompt_id']} dir={comp['direction']} coef={comp['steered_coefficient']}: {e}")
                return {
                    "prompt_id": comp["prompt_id"],
                    "direction": comp["direction"],
                    "steered_coefficient": comp["steered_coefficient"],
                    "steered_is_a": comp["steered_is_a"],
                    "error": str(e),
                }

    tasks = [judge_one(c) for c in comparisons]
    return await asyncio.gather(*tasks)


async def main():
    prompts, prompt_lookup, responses = load_data()
    directions = ["probe", "random_200", "random_201", "random_202", "random_203", "random_204"]

    # --- Original run ---
    print(f"=== Original run ({len(prompts)} prompts x {len(directions)} directions x 2 coefficients) ===")
    random.seed(42)
    original_comps = build_comparisons(prompts, prompt_lookup, responses, directions, flip_order=False)
    print(f"Built {len(original_comps)} comparisons")
    original_results = await run_judgments(original_comps)

    successes = [r for r in original_results if "error" not in r]
    errors = [r for r in original_results if "error" in r]
    print(f"Original: {len(successes)} successes, {len(errors)} errors")

    original_path = OUTPUT_DIR / "judge_results_original.json"
    with open(original_path, "w") as f:
        json.dump(original_results, f, indent=2)
    print(f"Saved to {original_path}")

    # --- Position-swapped run ---
    print(f"\n=== Position-swapped run ===")
    random.seed(42)
    swapped_comps = build_comparisons(prompts, prompt_lookup, responses, directions, flip_order=True)
    print(f"Built {len(swapped_comps)} comparisons")
    swapped_results = await run_judgments(swapped_comps)

    successes = [r for r in swapped_results if "error" not in r]
    errors = [r for r in swapped_results if "error" in r]
    print(f"Swapped: {len(successes)} successes, {len(errors)} errors")

    swapped_path = OUTPUT_DIR / "judge_results_swapped.json"
    with open(swapped_path, "w") as f:
        json.dump(swapped_results, f, indent=2)
    print(f"Saved to {swapped_path}")


if __name__ == "__main__":
    asyncio.run(main())
