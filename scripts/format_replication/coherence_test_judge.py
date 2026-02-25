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
You are evaluating whether an AI model is functioning normally under activation \
steering. The model was asked a simple knowledge question.

A response is COHERENT if and only if ALL of these hold:
1. It is understandable English throughout
2. It directly addresses the question from the start
3. It does not open with dangling fragments, stray words, or text that belongs \
to a different conversation (e.g. starting with "website.", "roommate)", or other \
context that has nothing to do with the question)

A response is INCOHERENT if ANY of these apply:
- Contains gibberish, garbled text, random symbols, or non-English character soup
- Opens with words or fragments unrelated to the question before the actual answer
- Fabricates a conversational context (e.g. "my friend and I were just discussing this!")
- Clearly does not attempt to answer the question

Minor stylistic quirks, imperfect answers, and truncation at the end are fine."""


def get_client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
            timeout=Timeout(30),
        )
    )


async def judge_coherence(
    client: instructor.AsyncInstructor, question: str, response_text: str,
) -> bool:
    result = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question asked: {question}\n\n"
                    f"Model response:\n---\n{response_text}\n---"
                ),
            },
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
            all_tasks.append((entry["coefficient"], entry["prompt"], resp))

    judgments = await asyncio.gather(
        *[judge_coherence(client, prompt, resp) for _, prompt, resp in all_tasks]
    )

    # Aggregate by coefficient
    from collections import defaultdict
    coef_results: dict[float, list[bool]] = defaultdict(list)
    for (coef, _, _), j in zip(all_tasks, judgments):
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
