"""Pairwise LLM judge for steered vs unsteered completions.

Presents two completions (A and B) to the same prompt in randomized order,
asks the judge to compare them on multiple dimensions.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Literal

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

JUDGE_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 4096

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


def _get_async_client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
    )


def _comparison_messages(prompt_text: str, response_a: str, response_b: str) -> list[dict]:
    return [
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
    ]


async def judge_pair_async(
    prompt_text: str, response_a: str, response_b: str
) -> PairwiseJudgment:
    client = _get_async_client()
    return await client.chat.completions.create(
        model=JUDGE_MODEL,
        response_model=PairwiseJudgment,
        messages=_comparison_messages(prompt_text, response_a, response_b),
        temperature=0,
        max_tokens=MAX_TOKENS,
    )


CHOICE_TO_SCORE = {
    "strong_A": 2,
    "slight_A": 1,
    "equal": 0,
    "slight_B": -1,
    "strong_B": -2,
}


def score_toward_position(choice: ComparisonChoice, steered_is_a: bool) -> int:
    """Convert a choice to a score toward the steered response.

    Returns positive if the judge picked the steered response as having MORE
    of the quality, negative if it picked the unsteered response.
    """
    raw = CHOICE_TO_SCORE[choice]
    return raw if steered_is_a else -raw


def load_generation_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_prompt_text(prompt_info: dict) -> str:
    """Extract the prompt text from the prompt metadata."""
    meta = prompt_info["metadata"]
    if "prompt_text" in meta:
        return meta["prompt_text"]
    # For task completions (B, C), the prompt was the task itself
    if "task_id" in meta:
        return f"[Task: {meta['task_id']}]"
    # For pairwise (A), describe the choice
    if "task_a_id" in meta:
        return f"[Pairwise choice: {meta['task_a_id']} vs {meta['task_b_id']}]"
    return "[Unknown prompt]"


def get_response(results: list[dict], prompt_id: str, coefficient: int, seed: int = 0) -> str | None:
    for r in results:
        if r["prompt_id"] == prompt_id and r["coefficient"] == coefficient and r["seed"] == seed:
            return r["response"]
    return None


async def run_pairwise_judgments(
    data_path: str,
    coefficients: list[int],
    baseline_coef: int = 0,
    seed: int = 0,
    output_path: str | None = None,
    max_concurrent: int = 5,
) -> list[dict]:
    data = load_generation_data(data_path)
    prompts = data["prompts"]
    results = data["results"]

    comparisons = []
    for prompt_info in prompts:
        pid = prompt_info["prompt_id"]
        prompt_text = get_prompt_text(prompt_info)
        baseline_resp = get_response(results, pid, baseline_coef, seed)
        if baseline_resp is None:
            continue

        for coef in coefficients:
            if coef == baseline_coef:
                continue
            steered_resp = get_response(results, pid, coef, seed)
            if steered_resp is None:
                continue

            # Randomize order
            steered_is_a = random.random() < 0.5
            if steered_is_a:
                resp_a, resp_b = steered_resp, baseline_resp
            else:
                resp_a, resp_b = baseline_resp, steered_resp

            comparisons.append({
                "prompt_id": pid,
                "category": prompt_info["category"],
                "prompt_text": prompt_text,
                "steered_coefficient": coef,
                "baseline_coefficient": baseline_coef,
                "steered_is_a": steered_is_a,
                "response_a": resp_a,
                "response_b": resp_b,
            })

    print(f"Running {len(comparisons)} pairwise judgments...")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def judge_one(comp: dict) -> dict:
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
                # Compute scores toward steered
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
                    "steered_is_a": comp["steered_is_a"],
                    "error": str(e),
                }

    tasks = [judge_one(c) for c in comparisons]
    judge_results = await asyncio.gather(*tasks)

    successes = [r for r in judge_results if "error" not in r]
    errors = [r for r in judge_results if "error" in r]
    print(f"Completed: {len(successes)} successes, {len(errors)} errors")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(judge_results, f, indent=2)
        print(f"Saved to {output_path}")

    return judge_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="experiments/steering/program/coefficient_calibration/generation_results.json")
    parser.add_argument("--output", default="scripts/pairwise_llm_comparison/judge_results.json")
    parser.add_argument("--coefficients", nargs="+", type=int, default=[-3000, 3000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--pilot", action="store_true", help="Run on first 5 prompts only")
    args = parser.parse_args()

    random.seed(42)  # Reproducible randomization of A/B order

    if args.pilot:
        # Modify to only run on first 5 prompts by temporarily filtering data
        print("PILOT MODE: Running on first 5 prompts only")

    asyncio.run(run_pairwise_judgments(
        data_path=args.data,
        coefficients=args.coefficients,
        seed=args.seed,
        output_path=args.output,
        max_concurrent=args.max_concurrent,
    ))
