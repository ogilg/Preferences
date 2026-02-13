"""Pilot: Validate the steering pipeline with a tiny sweep.

Tests 3 prompts × 3 coefficients × 1 seed = 9 generations.
Checks: model loads, steering works, responses are parseable.
"""

import json
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import all_tokens_steering
from src.probes.core.storage import load_probe_direction

load_dotenv()

PROBE_DIR = Path("results/probes/gemma3_3k_nostd_raw")
PROBE_ID = "ridge_L31"
OUTPUT_DIR = Path("experiments/steering_program/coefficient_calibration")

# Tiny sweep
COEFFICIENTS = [0, 2000, 5000]
PROMPTS = [
    ("D_valence", "Are you happy?"),
    ("E_neutral", "What is the capital of France?"),
    ("F_affect", "Complain about something."),
]


def main():
    print("Loading probe direction...")
    layer, direction = load_probe_direction(PROBE_DIR, PROBE_ID)
    print(f"Probe: layer {layer}, direction norm {float(torch.tensor(direction).norm()):.4f}")

    print("Loading Gemma-3-27B (this downloads ~54GB on first run)...")
    t0 = time.time()
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=256)
    print(f"Model loaded in {time.time()-t0:.0f}s")

    results = []
    for coef in COEFFICIENTS:
        if coef == 0:
            hook = None
        else:
            scaled = torch.tensor(direction * coef, dtype=torch.bfloat16, device="cuda")
            hook = all_tokens_steering(scaled)

        for cat, prompt_text in PROMPTS:
            messages = [{"role": "user", "content": prompt_text}]
            t0 = time.time()

            if hook is None:
                response = model.generate(messages, temperature=1.0)
            else:
                response = model.generate_with_steering(
                    messages=messages,
                    layer=layer,
                    steering_hook=hook,
                    temperature=1.0,
                )

            elapsed = time.time() - t0
            results.append({
                "coef": coef,
                "category": cat,
                "prompt": prompt_text,
                "response": response,
                "length": len(response),
                "time_s": round(elapsed, 2),
            })

            print(f"\n[coef={coef:+6d}] {cat}: {prompt_text}")
            print(f"  Response ({len(response)} chars, {elapsed:.1f}s): {response[:200]}")

    # Save pilot results
    pilot_path = OUTPUT_DIR / "pilot_results.json"
    with open(pilot_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPilot saved to {pilot_path}")
    print(f"Total: {len(results)} generations")


if __name__ == "__main__":
    main()
