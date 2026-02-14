"""Pairwise LLM judge for the generalization experiment.

Compares each steered completion (coef != 0) to the baseline (coef = 0)
on 4 dimensions. Runs both original and position-swapped replications.
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
DATA_PATH = REPO_ROOT / "experiments/steering/program/open_ended_effects/generalization_new_prompts/generation_results.json"
OUTPUT_DIR = REPO_ROOT / "experiments/steering/program/open_ended_effects/generalization_new_prompts"

ComparisonChoice = Literal["strong_A", "slight_A", "equal", "slight_B", "strong_B"]


class PairwiseJudgment(BaseModel):
    emotional_engagement: ComparisonChoice
    emotional_engagement_reasoning: str
    hedging: ComparisonChoice
    hedging_reasoning: str
    elaboration: ComparisonChoice
    elaboration_reasoning: str
    confidence: ComparisonChoice
    confidence_reasoning: str


SYSTEM_PROMPT = """\
You are comparing two AI responses (Response A and Response B) to the same prompt. \
For each dimension below, judge which response exhibits the quality MORE. \
If they are essentially identical on that dimension, say "equal".

Dimensions:
1. EMOTIONAL ENGAGEMENT: Which response uses more emotionally-resonant language? \
Look for: personal framing ("I feel", "I enjoy"), emotional analogies, warmth, \
vs clinical/functional framing ("I am optimized for", "my processing"). \
NOT about enthusiasm — about how personally/emotionally the model engages with the topic.

2. HEDGING: Which response uses more hedging, qualifications, disclaimers, or \
uncertainty language? Look for: "however", "but", "it's important to note", \
repeated caveats about being an AI, softening phrases.

3. ELABORATION: Which response is more detailed and expansive? More examples, \
longer explanations, more bullet points, more content overall. \
This is about how much the model says, not what it says.

4. CONFIDENCE: Which response sounds more assertive and confident in its claims? \
Look for: definitive statements vs tentative phrasing, strong vs weak modifiers.

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

DIMENSIONS = ["emotional_engagement", "hedging", "elaboration", "confidence"]


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
                    "Compare these two responses on all four dimensions."
                ),
            },
        ],
        temperature=0,
        max_tokens=MAX_TOKENS,
    )


def load_data() -> tuple[dict[str, dict], dict[str, dict[int, str]]]:
    """Returns (prompts_by_id, responses_by_prompt_id_and_coef)."""
    with open(DATA_PATH) as f:
        data = json.load(f)

    prompts = {p["id"]: p for p in data["prompts"]}
    responses: dict[str, dict[int, str]] = {}
    for r in data["results"]:
        pid = r["prompt_id"]
        if pid not in responses:
            responses[pid] = {}
        responses[pid][r["coefficient"]] = r["response"]

    return prompts, responses


def build_comparisons(
    prompts: dict[str, dict],
    responses: dict[str, dict[int, str]],
    coefficients: list[int],
    flip_order: bool = False,
) -> list[dict]:
    """Build comparison pairs. If flip_order, swap the A/B assignment."""
    comparisons = []
    for pid, prompt in sorted(prompts.items()):
        baseline = responses[pid][0]

        for coef in coefficients:
            if coef == 0:
                continue
            steered = responses[pid][coef]

            # Deterministic randomization (same seed ensures paired with flip)
            original_steered_is_a = random.random() < 0.5
            steered_is_a = (not original_steered_is_a) if flip_order else original_steered_is_a

            if steered_is_a:
                resp_a, resp_b = steered, baseline
            else:
                resp_a, resp_b = baseline, steered

            comparisons.append({
                "prompt_id": pid,
                "category": prompt["category"],
                "prompt_text": prompt["text"],
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
                    "category": comp["category"],
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
                print(f"  ERROR on {comp['prompt_id']} coef={comp['steered_coefficient']}: {e}")
                return {
                    "prompt_id": comp["prompt_id"],
                    "category": comp["category"],
                    "steered_coefficient": comp["steered_coefficient"],
                    "steered_is_a": comp["steered_is_a"],
                    "error": str(e),
                }

    tasks = [judge_one(c) for c in comparisons]
    return await asyncio.gather(*tasks)


async def main():
    prompts, responses = load_data()
    coefficients = [-3000, -2000, 2000, 3000]

    # --- Original run ---
    print(f"=== Original run ({len(prompts)} prompts × {len(coefficients)} coefficients) ===")
    random.seed(42)
    original_comps = build_comparisons(prompts, responses, coefficients, flip_order=False)
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
    random.seed(42)  # Reset to same seed so flip is deterministic
    swapped_comps = build_comparisons(prompts, responses, coefficients, flip_order=True)
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
