"""Pilot: test triage pipeline with 1 persona, 1 layer, 2 coefs, 2 questions."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dotenv import load_dotenv
load_dotenv()

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.models.huggingface_model import HuggingFaceModel
from src.steering.client import SteeredHFClient


JUDGE_MODEL = "google/gemini-3-flash-preview"


class TraitScore(BaseModel):
    reasoning: str = Field(description="Brief reasoning for the score")
    score: int = Field(ge=1, le=5, description="Trait expression score")


async def judge_one(question: str, response: str, persona_prompt: str):
    client = instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    )
    result = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": f"Question asked: {question}\n\nResponse:\n{response}"},
        ],
        response_model=TraitScore,
        temperature=0,
        max_tokens=512,
    )
    return result


def main():
    print("Loading model...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=256)

    # Load creative_artist persona
    with open("experiments/persona_vectors/artifacts/creative_artist.json") as f:
        persona = json.load(f)

    # Load direction for prompt_last L31
    direction = np.load(
        "results/experiments/persona_vectors_v2/creative_artist/vectors/creative_artist_prompt_last_L31_direction.npy"
    )

    mean_norm = 52822.8
    questions = [persona["eval_questions"][30], persona["eval_questions"][31]]

    # Test at coef=0 and coef=0.1x
    for mult in [0.0, 0.1]:
        coef = mean_norm * mult
        client = SteeredHFClient(
            hf_model=model, layer=31,
            steering_direction=direction, coefficient=coef,
            steering_mode="all_tokens",
        )

        print(f"\n--- Multiplier: {mult}, Coefficient: {coef:.1f} ---")
        for q in questions:
            messages = [{"role": "user", "content": q}]
            t0 = time.time()
            response = client.generate(messages, temperature=0.7)
            gen_time = time.time() - t0
            print(f"\nQ: {q}")
            print(f"A ({gen_time:.1f}s): {response[:200]}...")

            # Judge
            prompt = "Rate how creatively artistic this response is. Look for: poetic language, metaphors, emotional depth, imaginative framing. Score 1-5."
            result = asyncio.run(judge_one(q, response, prompt))
            print(f"Score: {result.score} ({result.reasoning[:80]})")

    print("\nPilot complete!")


if __name__ == "__main__":
    main()
