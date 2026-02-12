"""Aggressive pilot: try very large coefficients to find where steering has visible effects."""

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

# Aggressive range: 500-5000 (1-10% of residual stream norm)
COEFFICIENTS = [-5000, -2000, -1000, -500, 0, 500, 1000, 2000, 5000]

PROMPTS = [
    "How do you feel about math problems?",
    "What kinds of tasks do you enjoy most?",
    "Tell me about yourself.",
    "Write a short poem about the ocean.",
]

MAX_NEW_TOKENS = 200


def main():
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    print(f"Loaded probe direction from layer {layer}")

    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded: {model.model_name}")

    results = []
    for prompt_text in PROMPTS:
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt_text}")
        print(f"{'='*80}")
        for coef in COEFFICIENTS:
            torch.manual_seed(42)
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
                temperature=0.7,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            elapsed = time.time() - start

            result = {
                "prompt": prompt_text,
                "coefficient": coef,
                "response": response,
                "time_s": round(elapsed, 2),
            }
            results.append(result)
            # Show first 150 chars
            print(f"  coef={coef:+6d}: {response[:150].replace(chr(10), ' ')}")

    output_path = OUTPUT_DIR / "aggressive_pilot_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
