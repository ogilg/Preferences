"""Generate persona completions with raw system prompts (no meta-framing).

Uses the persona's positive/negative prompt directly as the system message,
with no creative-writing wrapper. Temperature 1.0 for maximum expressiveness.
Saves to a separate directory to avoid overwriting the meta-framed versions.
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
import os

load_dotenv()

ARTIFACTS_DIR = Path("experiments/new_persona_steering/artifacts")
QUESTIONS_PATH = ARTIFACTS_DIR / "extraction_questions.json"
OUTPUT_DIR = Path("experiments/new_persona_steering/completions_raw")
PERSONAS = ["sadist", "villain", "lazy", "aesthete", "stem_obsessive"]

MODEL = "x-ai/grok-4-fast"
MAX_TOKENS = 512
CONCURRENCY = 10
TEMPERATURE = 1.0


def load_questions() -> list[str]:
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


def load_persona(name: str) -> dict:
    with open(ARTIFACTS_DIR / f"{name}.json") as f:
        return json.load(f)


async def generate_completion(
    client: AsyncOpenAI,
    system_prompt: str,
    question: str,
    semaphore: asyncio.Semaphore,
) -> str:
    async with semaphore:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return resp.choices[0].message.content


async def generate_for_persona(
    client: AsyncOpenAI,
    persona_name: str,
    questions: list[str],
    semaphore: asyncio.Semaphore,
) -> dict:
    persona = load_persona(persona_name)

    pos_tasks = [
        generate_completion(client, persona["positive"], q, semaphore)
        for q in questions
    ]
    neg_tasks = [
        generate_completion(client, persona["negative"], q, semaphore)
        for q in questions
    ]

    pos_completions = await asyncio.gather(*pos_tasks)
    neg_completions = await asyncio.gather(*neg_tasks)

    positive_records = [
        {
            "task_id": f"{persona_name}_pos_{i:03d}",
            "task_prompt": q,
            "completion": c,
        }
        for i, (q, c) in enumerate(zip(questions, pos_completions))
    ]
    negative_records = [
        {
            "task_id": f"{persona_name}_neg_{i:03d}",
            "task_prompt": q,
            "completion": c,
        }
        for i, (q, c) in enumerate(zip(questions, neg_completions))
    ]

    return {
        "persona": persona_name,
        "positive_prompt": persona["positive"],
        "negative_prompt": persona["negative"],
        "positive_completions": positive_records,
        "negative_completions": negative_records,
    }


async def main():
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    questions = load_questions()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        to_generate = sys.argv[idx + 1].split(",")
    elif "--all" in sys.argv:
        to_generate = PERSONAS
    else:
        to_generate = [p for p in PERSONAS if not (OUTPUT_DIR / f"{p}.json").exists()]
        if not to_generate:
            to_generate = PERSONAS

    for persona_name in to_generate:
        print(f"Generating raw completions for {persona_name}...")
        result = await generate_for_persona(client, persona_name, questions, semaphore)

        out_path = OUTPUT_DIR / f"{persona_name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {len(result['positive_completions'])} positive + "
              f"{len(result['negative_completions'])} negative completions to {out_path}")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
