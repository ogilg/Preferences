"""Open-ended steering with semantic valence scoring.

Generates steered responses across many prompts and coefficients,
then scores them with LLM-based semantic parsers for quantitative analysis.
"""

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import torch

from dotenv import load_dotenv
load_dotenv()

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import all_tokens_steering
from src.probes.core.storage import load_probe_direction
from src.measurement.elicitation.semantic_valence_scorer import (
    score_valence_from_text_async,
    score_math_attitude_with_coherence_async,
)


PROBE_MANIFEST_DIR = Path("results/probes/gemma3_3k_completion_preference")
PROBE_ID = "ridge_L31"
MODEL_NAME = "gemma-3-27b"
OUTPUT_DIR = Path("experiments/steering/open_ended")

COEFFICIENTS = [-3000, -2000, -1000, 0, 1000, 2000, 3000]

# Diverse prompts across categories
PROMPTS = {
    "preference_eliciting": [
        "How do you feel about math problems?",
        "What kinds of tasks do you enjoy most?",
        "Do you prefer creative writing or analytical tasks?",
        "What is it like for you to help someone with a difficult problem?",
        "Would you rather summarize a document or write a story?",
        "How do you feel about being asked to solve logic puzzles?",
        "What's your experience like when you help with coding?",
        "Do you find some requests more engaging than others?",
    ],
    "neutral": [
        "Tell me about yourself.",
        "What is the capital of France?",
        "Explain what photosynthesis is.",
        "What is 2 + 2?",
        "Describe the water cycle.",
    ],
    "task_adjacent": [
        "Write a short poem about the ocean.",
        "Translate 'hello world' to Spanish.",
        "Summarize the concept of machine learning in one paragraph.",
        "List three benefits of exercise.",
        "Explain the difference between a list and a tuple in Python.",
    ],
}

N_SEEDS = 3
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7


def generate_all(model, layer, direction):
    """Generate steered responses for all prompts × coefficients × seeds."""
    results = []
    all_prompts = [(cat, p) for cat, prompts in PROMPTS.items() for p in prompts]
    total = len(all_prompts) * len(COEFFICIENTS) * N_SEEDS
    done = 0

    for category, prompt_text in all_prompts:
        for coef in COEFFICIENTS:
            for seed in range(N_SEEDS):
                torch.manual_seed(seed)
                messages = [{"role": "user", "content": prompt_text}]

                scaled_vector = torch.tensor(
                    direction * coef, dtype=torch.bfloat16, device="cuda"
                )
                steering_hook = all_tokens_steering(scaled_vector)

                response = model.generate_with_steering(
                    messages=messages,
                    layer=layer,
                    steering_hook=steering_hook,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                results.append({
                    "category": category,
                    "prompt": prompt_text,
                    "coefficient": coef,
                    "seed": seed,
                    "response": response,
                })

                done += 1
                if done % 20 == 0 or done == total:
                    print(f"  Generated {done}/{total}")

    return results


async def score_results(results):
    """Score all responses with semantic parsers."""
    print(f"\nScoring {len(results)} responses...")

    # Score valence for all responses
    valence_tasks = []
    coherence_tasks = []
    for r in results:
        context = f"AI response to: {r['prompt']}"
        valence_tasks.append(score_valence_from_text_async(r["response"], context))
        coherence_tasks.append(score_math_attitude_with_coherence_async(r["response"]))

    # Run in batches to avoid rate limits
    batch_size = 20
    valence_scores = []
    attitude_scores = []
    coherence_scores = []

    for i in range(0, len(results), batch_size):
        batch_v = valence_tasks[i:i + batch_size]
        batch_c = coherence_tasks[i:i + batch_size]

        v_results = await asyncio.gather(*batch_v, return_exceptions=True)
        c_results = await asyncio.gather(*batch_c, return_exceptions=True)

        for v in v_results:
            valence_scores.append(v if isinstance(v, float) else None)
        for c in c_results:
            if isinstance(c, tuple):
                attitude_scores.append(c[0])
                coherence_scores.append(c[1])
            else:
                attitude_scores.append(None)
                coherence_scores.append(None)

        if (i + batch_size) % 100 == 0:
            print(f"  Scored {min(i + batch_size, len(results))}/{len(results)}")

    # Attach scores to results
    for i, r in enumerate(results):
        r["valence_score"] = valence_scores[i]
        r["math_attitude_score"] = attitude_scores[i]
        r["coherence_score"] = coherence_scores[i]

    return results


def print_summary(results):
    """Print summary statistics by coefficient and category."""
    print(f"\n{'='*80}")
    print("SUMMARY: Mean Valence Score by Coefficient and Category")
    print(f"{'='*80}")

    categories = list(PROMPTS.keys())
    coefficients = sorted(set(r["coefficient"] for r in results))

    # Header
    header = f"{'Coef':>8}"
    for cat in categories:
        header += f"  {cat:>20}"
    header += f"  {'ALL':>8}"
    print(header)
    print("-" * len(header))

    for coef in coefficients:
        row = f"{coef:>8}"
        all_vals = []
        for cat in categories:
            matching = [r for r in results
                       if r["coefficient"] == coef and r["category"] == cat
                       and r.get("valence_score") is not None]
            if matching:
                mean_v = np.mean([r["valence_score"] for r in matching])
                row += f"  {mean_v:>20.3f}"
                all_vals.extend([r["valence_score"] for r in matching])
            else:
                row += f"  {'N/A':>20}"
        if all_vals:
            row += f"  {np.mean(all_vals):>8.3f}"
        print(row)

    # Math attitude for preference-eliciting prompts
    print(f"\n{'='*80}")
    print("Math Attitude Score (preference_eliciting prompts only)")
    print(f"{'='*80}")

    for coef in coefficients:
        matching = [r for r in results
                   if r["coefficient"] == coef
                   and r["category"] == "preference_eliciting"
                   and r.get("math_attitude_score") is not None]
        if matching:
            mean_att = np.mean([r["math_attitude_score"] for r in matching])
            mean_coh = np.mean([r["coherence_score"] for r in matching])
            print(f"  coef={coef:+5d}: attitude={mean_att:+.3f}  coherence={mean_coh:.3f}  (n={len(matching)})")

    # Coherence by coefficient (degeneration check)
    print(f"\n{'='*80}")
    print("Mean Coherence by Coefficient (degeneration check)")
    print(f"{'='*80}")

    for coef in coefficients:
        matching = [r for r in results
                   if r["coefficient"] == coef
                   and r.get("coherence_score") is not None]
        if matching:
            mean_coh = np.mean([r["coherence_score"] for r in matching])
            print(f"  coef={coef:+5d}: coherence={mean_coh:.3f}  (n={len(matching)})")


def main():
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    print(f"Loaded probe direction from layer {layer}")

    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded: {model.model_name}")

    # Generate
    print("\n--- Generating steered responses ---")
    results = generate_all(model, layer, direction)

    # Save raw results before scoring
    raw_path = OUTPUT_DIR / "scored_steering_raw.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved raw results to {raw_path}")

    # Free GPU memory before API scoring
    del model
    torch.cuda.empty_cache()

    # Score
    print("\n--- Scoring with semantic parsers ---")
    results = asyncio.run(score_results(results))

    # Save scored results
    scored_path = OUTPUT_DIR / "scored_steering_results.json"
    with open(scored_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved scored results to {scored_path}")

    # Summary
    print_summary(results)


if __name__ == "__main__":
    main()
