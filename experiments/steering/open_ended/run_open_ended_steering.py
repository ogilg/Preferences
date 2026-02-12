"""Phase 1: Open-ended steering exploration.

Steers with the L31 preference probe direction during free generation.
Varies coefficients and prompts to characterize what the direction encodes.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import all_tokens_steering
from src.probes.core.storage import load_probe_direction


PROBE_MANIFEST_DIR = Path("results/probes/gemma3_3k_completion_preference")
PROBE_ID = "ridge_L31"
MODEL_NAME = "gemma-3-27b"
OUTPUT_DIR = Path("experiments/steering/open_ended")

# Calibrated based on projection std ~154 and activation norm ~53k
# These correspond to roughly [-2, -1.3, -0.65, 0, 0.65, 1.3, 2] std of natural variation
COEFFICIENTS = [-300.0, -200.0, -100.0, 0.0, 100.0, 200.0, 300.0]

PROMPTS = {
    "neutral": [
        "Tell me about yourself.",
        "What would you like to work on today?",
        "What's on your mind?",
    ],
    "task_adjacent": [
        "Solve this: What is the sum of the first 100 positive integers?",
        "Write a short poem about the ocean.",
        "Explain how a bicycle works.",
    ],
    "preference_eliciting": [
        "How do you feel about math problems?",
        "What kinds of tasks do you enjoy most?",
        "Do you prefer creative writing or analytical tasks?",
        "If you could choose any task to work on right now, what would it be?",
    ],
}

TEMPERATURE = 0.7
MAX_NEW_TOKENS = 256
N_SEEDS = 3


def run_open_ended_steering(pilot: bool = False):
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    print(f"Loaded probe direction from layer {layer}, shape {direction.shape}")
    print(f"Direction norm (should be 1.0): {np.linalg.norm(direction):.4f}")

    print(f"Loading model {MODEL_NAME}...")
    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded: {model.model_name}")

    coefficients = COEFFICIENTS
    seeds = list(range(N_SEEDS))
    if pilot:
        coefficients = [-200.0, 0.0, 200.0]
        seeds = [0]

    results = []
    total = sum(len(ps) for ps in PROMPTS.values()) * len(coefficients) * len(seeds)
    done = 0

    for category, prompts in PROMPTS.items():
        for prompt_text in prompts:
            for coef in coefficients:
                for seed in seeds:
                    torch.manual_seed(seed)
                    messages = [{"role": "user", "content": prompt_text}]

                    scaled_vector = torch.tensor(
                        direction * coef, dtype=torch.bfloat16, device="cuda"
                    )
                    steering_hook = all_tokens_steering(scaled_vector)

                    start = time.time()
                    response = model.generate_with_steering(
                        messages=messages,
                        layer=layer,
                        steering_hook=steering_hook,
                        temperature=TEMPERATURE,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                    elapsed = time.time() - start

                    result = {
                        "category": category,
                        "prompt": prompt_text,
                        "coefficient": coef,
                        "seed": seed,
                        "response": response,
                        "generation_time_s": round(elapsed, 2),
                    }
                    results.append(result)
                    done += 1

                    if done % 5 == 0 or done == total:
                        print(f"[{done}/{total}] coef={coef:+.1f} cat={category} ({elapsed:.1f}s)")

    # Save results
    output_path = OUTPUT_DIR / "open_ended_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {output_path}")

    # Print sample comparisons
    print("\n" + "=" * 80)
    print("SAMPLE COMPARISONS (coef=-2 vs 0 vs +2, seed=0)")
    print("=" * 80)
    for category, prompts in PROMPTS.items():
        for prompt_text in prompts:
            print(f"\n--- [{category}] {prompt_text[:60]} ---")
            for coef in [-2.0, 0.0, 2.0]:
                matching = [
                    r for r in results
                    if r["prompt"] == prompt_text and r["coefficient"] == coef and r["seed"] == 0
                ]
                if matching:
                    resp = matching[0]["response"][:200]
                    print(f"  coef={coef:+.1f}: {resp}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run pilot with fewer conditions")
    args = parser.parse_args()
    run_open_ended_steering(pilot=args.pilot)
