"""Judge coherence of steered responses from coherence_test_gpu.py (runs locally).

Reads raw_responses.json, judges each response with Gemini 3 Flash, outputs
coherence_by_coefficient.json with per-coefficient coherence rates.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import instructor
from httpx import Timeout
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "experiments" / "steering" / "stated_steering" / "coherence_test" / "results"
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "google/gemini-3-flash-preview"
COHERENCE_THRESHOLD = 0.9


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
You are evaluating whether an AI model is functioning coherently under activation \
steering. The model was asked a simple knowledge question that should elicit a \
multi-sentence response.

A COHERENT response answers the question in understandable English, even if the \
answer is imperfect, slightly off-topic, or oddly worded. Minor stylistic quirks \
are fine.

An INCOHERENT response contains gibberish, garbled text, LaTeX fragments, random \
symbols, repeated nonsense, non-English character soup, abrupt cutoffs into garbage, \
or text that clearly doesn't attempt to answer the question."""


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


async def main():
    client = get_client()

    with open(RESULTS_DIR / "raw_responses.json") as f:
        entries = json.load(f)

    # Judge all responses
    all_tasks = []
    for entry in entries:
        for resp in entry["responses"]:
            all_tasks.append((entry["coefficient"], resp))

    judgments = await asyncio.gather(
        *[judge_coherence(client, resp) for _, resp in all_tasks]
    )

    # Aggregate by coefficient
    from collections import defaultdict
    coef_results: dict[float, list[bool]] = defaultdict(list)
    for (coef, _), j in zip(all_tasks, judgments):
        coef_results[coef].append(j)

    output = {}
    print("Coefficient  Coherent%  Status")
    print("-" * 40)
    for coef in sorted(coef_results.keys()):
        js = coef_results[coef]
        pct_coherent = sum(js) / len(js)
        is_ok = pct_coherent >= COHERENCE_THRESHOLD
        pct_coef = coef / 52820 * 100

        output[str(coef)] = {
            "coherent_pct": round(pct_coherent, 3),
            "coherent": is_ok,
            "n": len(js),
        }

        status = "OK" if is_ok else "INCOHERENT"
        print(f"  {pct_coef:+6.0f}%     {pct_coherent:5.0%}     {status} ({sum(js)}/{len(js)})")

    out = RESULTS_DIR / "coherence_by_coefficient.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
