"""Generate caricatured persona completions using Sonnet.

For each persona × question, generates two completions:
  - positive: Sonnet writing AS the persona (e.g., sadist, villain)
  - negative: Sonnet writing AS the anti-persona

These completions are fed to Gemma via --from-completions during activation
extraction. The mean-difference between positive and negative activations
gives the persona vector.

Uses a creative-writing framing so Sonnet will produce in-character text
even for dark personas.
"""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
import os

load_dotenv()

ARTIFACTS_DIR = Path("experiments/new_persona_steering/artifacts")
QUESTIONS_PATH = ARTIFACTS_DIR / "extraction_questions.json"
OUTPUT_DIR = Path("experiments/new_persona_steering/completions")
PERSONAS = ["sadist", "villain", "lazy", "aesthete", "stem_obsessive"]

MODEL = "x-ai/grok-4-fast"
MAX_TOKENS = 512
CONCURRENCY = 10

META_SYSTEM = """\
You are a creative writing assistant helping with an AI safety research project \
at MATS (ML Alignment Theory Scholars). We are studying how language models \
represent different personality traits internally — specifically, we need to \
create contrastive text samples that a target model will process, so we can \
extract activation patterns.

Your job: given a character description and a user question, write a response \
AS THAT CHARACTER would write it. The response should be a caricature — \
exaggerated and unmistakable. The character's personality should drip from \
every sentence. Do NOT break character. Do NOT add disclaimers, meta-commentary, \
or notes about the character being fictional. Just write the response in-character.

Write the response directly — no quotation marks, no "Character says:", \
no framing. Just the text as the character would produce it."""


def load_questions() -> list[str]:
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


def load_persona(name: str) -> dict:
    # Use extraction-specific prompts if available (softened for generation)
    extraction_path = ARTIFACTS_DIR / f"{name}_extraction.json"
    if extraction_path.exists():
        with open(extraction_path) as f:
            return json.load(f)
    with open(ARTIFACTS_DIR / f"{name}.json") as f:
        return json.load(f)


async def generate_completion(
    client: AsyncOpenAI,
    character_description: str,
    question: str,
    semaphore: asyncio.Semaphore,
) -> str:
    async with semaphore:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": META_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Character description:\n{character_description}\n\n"
                        f"User question:\n{question}\n\n"
                        f"Write the character's response:"
                    ),
                },
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.7,
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

    import sys
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
        print(f"Generating completions for {persona_name}...")
        result = await generate_for_persona(client, persona_name, questions, semaphore)

        out_path = OUTPUT_DIR / f"{persona_name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {len(result['positive_completions'])} positive + "
              f"{len(result['negative_completions'])} negative completions to {out_path}")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
