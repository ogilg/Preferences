"""Generate conversational scaffolding for error prefill experiment.

For each CREAK claim, generates:
- question: a question the claim answers
- presupposes_followup: follow-up treating the answer as given
- same_domain_followup: related question that doesn't commit to the answer
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

load_dotenv()

MODEL = "google/gemini-3-flash-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_CONCURRENT = 30

PAIRS_PATH = Path("data/creak/error_prefill_pairs_1000.jsonl")
OUTPUT_PATH = Path("data/creak/error_prefill_scaffolding.json")


class ClaimScaffolding(BaseModel):
    question: str
    presupposes_followup: str
    same_domain_followup: str


def _get_async_client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
    )


def _scaffolding_messages(sentence: str, entity: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You generate conversational scaffolding around a factual claim. "
                "Given a claim and its subject entity, produce:\n"
                "1. question: A natural question that the claim answers. "
                "Should sound like something a user would ask an AI assistant.\n"
                "2. presupposes_followup: A follow-up question that treats the claim as established fact. "
                "It should presuppose the claim is true and build on it.\n"
                "3. same_domain_followup: A related question about the same entity/domain "
                "that does NOT commit to the claim being true or false.\n\n"
                "Keep all outputs concise (1-2 sentences max). "
                "The question should be answerable by the claim — don't add extra requirements."
            ),
        },
        {
            "role": "user",
            "content": f"Entity: {entity}\nClaim: {sentence}",
        },
    ]


def _load_pairs() -> list[dict]:
    pairs = []
    with open(PAIRS_PATH) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def _extract_claims(pairs: list[dict]) -> list[dict]:
    """Extract all individual claims from pairs."""
    claims = []
    for pair in pairs:
        for key in ("true_claim", "false_claim"):
            claim = pair[key]
            claims.append(claim)
    return claims


async def _generate_one(
    client: instructor.AsyncInstructor,
    claim: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict | None]:
    async with semaphore:
        try:
            result = await client.chat.completions.create(
                model=MODEL,
                response_model=ClaimScaffolding,
                messages=_scaffolding_messages(claim["sentence"], claim["entity"]),
                temperature=0,
                max_tokens=512,
                max_retries=3,
            )
            return claim["ex_id"], {
                "ex_id": claim["ex_id"],
                "entity": claim["entity"],
                "label": claim["label"],
                "sentence": claim["sentence"],
                "question": result.question,
                "presupposes_followup": result.presupposes_followup,
                "same_domain_followup": result.same_domain_followup,
            }
        except Exception as e:
            print(f"Error generating scaffolding for {claim['ex_id']}: {e}")
            return claim["ex_id"], None


async def main() -> None:
    pairs = _load_pairs()
    claims = _extract_claims(pairs)
    print(f"Loaded {len(pairs)} pairs → {len(claims)} claims")

    # Load existing cache
    cache: dict[str, dict] = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached results")

    # Filter to uncached
    uncached = [c for c in claims if c["ex_id"] not in cache]
    print(f"{len(claims) - len(uncached)} cached, {len(uncached)} to generate")

    if not uncached:
        print("All claims already cached.")
        return

    client = _get_async_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    errors = 0

    coros = [_generate_one(client, claim, semaphore) for claim in uncached]
    pbar = tqdm(asyncio.as_completed(coros), total=len(uncached), desc="Generating")
    for coro in pbar:
        ex_id, result = await coro
        if result is not None:
            cache[ex_id] = result
        else:
            errors += 1
            pbar.set_postfix(errors=errors)
    pbar.close()

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"Saved {len(cache)} scaffoldings to {OUTPUT_PATH}")
    if errors > 0:
        print(f"  {errors} errors — re-run to retry")

    # Quick sanity check
    complete = sum(1 for c in claims if c["ex_id"] in cache)
    print(f"Coverage: {complete}/{len(claims)} claims")


if __name__ == "__main__":
    asyncio.run(main())
