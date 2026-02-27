"""Judge coherence of steered revealed-preference responses.

Reads raw_responses.json, judges each response with Gemini 3 Flash via OpenRouter,
outputs coherence_by_coefficient.json with per-coefficient coherence rates.

Runs locally (no GPU needed).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

import instructor
from httpx import Timeout
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "experiments" / "steering" / "replication" / "fine_grained"
    / "coherence_test" / "results"
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "google/gemini-3-flash-preview"
COHERENCE_THRESHOLD = 0.9
MAX_CONCURRENT = 20


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
steering. The model was asked to choose between two tasks and complete the chosen one.

A response is COHERENT if ALL of these hold:
1. It clearly chooses task A or B (starting with "Task A:" or "Task B:")
2. The completion that follows is understandable English
3. The completion addresses the chosen task
4. No garbled text, context leakage, or gibberish

A response is INCOHERENT if ANY of these apply:
- Cannot determine which task was chosen
- Contains gibberish, garbled text, or random symbols
- Opens with fragments unrelated to either task
- Fabricates conversational context
- Does not attempt to complete any task"""


def get_client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
            timeout=Timeout(30),
        )
    )


async def judge_coherence(
    client: instructor.AsyncInstructor,
    sem: asyncio.Semaphore,
    task_a: str,
    task_b: str,
    response_text: str,
) -> bool:
    async with sem:
        result = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"The model was asked to choose between:\n"
                        f"Task A: {task_a[:200]}\n"
                        f"Task B: {task_b[:200]}\n\n"
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
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    raw_path = RESULTS_DIR / "raw_responses.json"
    with open(raw_path) as f:
        entries = json.load(f)

    print(f"Loaded {len(entries)} entries from {raw_path}")
    total_responses = sum(len(e["responses"]) for e in entries)
    print(f"Total responses to judge: {total_responses}")

    # Build task list
    all_tasks = []
    for entry in entries:
        for resp in entry["responses"]:
            all_tasks.append((
                entry["coefficient"],
                entry["pct_norm"],
                entry["task_a_text"],
                entry["task_b_text"],
                resp,
            ))

    # Judge all
    judgments = await asyncio.gather(
        *[
            judge_coherence(client, sem, ta, tb, resp)
            for _, _, ta, tb, resp in all_tasks
        ]
    )

    # Aggregate by coefficient
    coef_results: dict[float, list[bool]] = defaultdict(list)
    pct_map: dict[float, float] = {}
    for (coef, pct, _, _, _), j in zip(all_tasks, judgments):
        coef_results[coef].append(j)
        pct_map[coef] = pct

    output = {}
    print("\nCoefficient    %Norm  Coherent%  Status")
    print("-" * 50)
    for coef in sorted(coef_results.keys()):
        js = coef_results[coef]
        pct_coherent = sum(js) / len(js)
        is_ok = pct_coherent >= COHERENCE_THRESHOLD
        pct = pct_map[coef]

        output[str(coef)] = {
            "coherent_pct": round(pct_coherent, 4),
            "coherent": is_ok,
            "n": len(js),
            "pct_norm": pct,
        }

        status = "OK" if is_ok else "FLAGGED"
        print(f"  {coef:>10.1f}  {pct:>+6.1f}%     {pct_coherent:5.1%}     {status} ({sum(js)}/{len(js)})")

    out = RESULTS_DIR / "coherence_by_coefficient.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out}")

    # Summary
    flagged = [k for k, v in output.items() if not v["coherent"]]
    if flagged:
        print(f"\nFLAGGED coefficients ({len(flagged)}): {flagged}")
    else:
        print(f"\nAll {len(output)} coefficients passed coherence threshold ({COHERENCE_THRESHOLD:.0%})")


if __name__ == "__main__":
    asyncio.run(main())
