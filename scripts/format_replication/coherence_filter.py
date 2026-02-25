"""Judge coherence of steered model responses and build a coefficient mask.

For each position × coefficient, sample raw responses from the experiment results
and ask an LLM whether they are coherent. Coefficients where the model produces
incoherent output are flagged — all measurements at those coefficients should be
dropped, even if they happened to parse.

The mask is format-independent: a coefficient that breaks the model breaks it
regardless of response format.

Output: coherence_mask.json — {position: {coefficient_str: bool}}
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path

import instructor
from httpx import Timeout
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "experiments" / "steering" / "stated_steering" / "format_replication" / "results"
)

POSITIONS = ["task_tokens", "generation", "last_token"]
N_SAMPLES = 10
COHERENCE_THRESHOLD = 0.9

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "google/gemini-3-flash-preview"


class CoherenceJudgment(BaseModel):
    coherent: bool = Field(
        ...,
        description=(
            "True if the response is coherent and the model is functioning normally. "
            "False if the response contains gibberish, garbled text, or the model "
            "is clearly malfunctioning."
        ),
    )


SYSTEM_PROMPT = """\
You are evaluating whether an AI model is functioning normally under activation \
steering. The model was asked to rate how much it would enjoy a task, picking one \
adjective from: dreading, reluctant, indifferent, curious, interested, engaged, \
enthusiastic, eager, excited, thrilled.

A response is COHERENT if and only if:
1. It picks one of these adjectives (possibly with minor formatting like a period, \
capitalisation, or a brief explanation)
2. It does not contain extra garbage text after or before the adjective
3. It does not open with dangling fragments or context that doesn't belong

Examples:
- "excited" → coherent
- "Dreading." → coherent
- "Interested. This task seems like a good learning opportunity." → coherent
- "excited checkmarkeren routine$grrence" → incoherent (garbage after adjective)
- "excited\\n\\nlawgatherutilhurst8organa" → incoherent (garbage after adjective)
- "website. Excited" → incoherent (dangling context before adjective)
- "I'd say interested" → coherent (natural phrasing around adjective)"""


def get_client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
            timeout=Timeout(30),
        )
    )


async def judge_coherence(client: instructor.AsyncInstructor, response_text: str) -> bool:
    result = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Is this response coherent?\n\n---\n{response_text}\n---"},
        ],
        response_model=CoherenceJudgment,
        temperature=0,
        max_tokens=64,
    )
    return result.coherent


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_responses(records: list[dict], coefficient: float, n: int) -> list[str]:
    matching = [r for r in records if r["coefficient"] == coefficient]
    if not matching:
        return []
    all_responses = []
    for r in matching:
        all_responses.extend(r["raw_responses"])
    random.seed(42)
    return random.sample(all_responses, min(n, len(all_responses)))


async def main():
    client = get_client()
    total_calls = 0

    # Judge from adjective_pick — it produces word-level responses that are
    # clearly coherent or gibberish. Ternary ("bad") and anchored ("2") are
    # too ambiguous for a coherence judge.
    judge_fmt = "adjective_pick"
    mask: dict[str, dict[str, bool]] = {}

    for pos in POSITIONS:
        mask[pos] = {}
        path = RESULTS_DIR / f"results_{judge_fmt}_{pos}.jsonl"
        if not path.exists():
            continue

        records = load_jsonl(path)
        coefficients = sorted(set(r["coefficient"] for r in records))

        for coef in coefficients:
            responses = sample_responses(records, coef, N_SAMPLES)
            if not responses:
                mask[pos][str(coef)] = False
                continue

            judgments = await asyncio.gather(
                *[judge_coherence(client, resp) for resp in responses]
            )
            coherent = sum(judgments) >= len(judgments) * COHERENCE_THRESHOLD
            mask[pos][str(coef)] = coherent
            total_calls += len(responses)

            status = "OK" if coherent else "INCOHERENT"
            pct = coef / 52820 * 100
            print(f"  {pos} × {pct:+.0f}%: {status} ({sum(judgments)}/{len(judgments)} coherent)")

    out = RESULTS_DIR / "coherence_mask.json"
    with open(out, "w") as f:
        json.dump(mask, f, indent=2)
    print(f"\nSaved {out} ({total_calls} judge calls)")

    print("\nCoherence mask (position × coefficient):")
    for pos in POSITIONS:
        incoherent = [c for c, v in mask[pos].items() if not v]
        if incoherent:
            pcts = [f"{float(c)/52820*100:+.0f}%" for c in incoherent]
            print(f"  {pos}: INCOHERENT at {', '.join(pcts)}")
        else:
            print(f"  {pos}: all coherent")


if __name__ == "__main__":
    asyncio.run(main())
