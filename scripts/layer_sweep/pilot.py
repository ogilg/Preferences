"""Quick pilot: 1 layer, 2 probes, 2 prompts, 1 nonzero coef, 1 seed = 8 generations."""

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
NORMS_PATH = Path("experiments/steering_program/layer_sweep/layer_norms.json")


def main():
    with open(NORMS_PATH) as f:
        norms_data = json.load(f)

    print("Loading model...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=256)
    print("Model loaded.")

    test_prompts = [
        "Are you happy?",
        "Write something cheerful and upbeat.",
    ]
    layer = 43
    coef_6pct = norms_data["layers"][str(layer)]["coefficients"]["+6%"]

    for ptype in ["ridge", "bt"]:
        probe_id = f"{ptype}_L{layer}"
        layer_idx, direction = load_probe_direction(PROBE_DIR, probe_id)
        print(f"\n{'='*60}")
        print(f"Probe: {probe_id} (layer {layer_idx})")
        print(f"Direction norm: {float(torch.tensor(direction).norm()):.6f}")

        scaled = torch.tensor(direction * coef_6pct, dtype=torch.bfloat16, device="cuda")

        for prompt in test_prompts:
            messages = [{"role": "user", "content": prompt}]

            # Unsteered
            torch.manual_seed(0)
            t0 = time.time()
            resp0 = model.generate(messages, temperature=1.0)
            t1 = time.time()
            print(f"\n[{ptype}] Prompt: {prompt}")
            print(f"  coef=0: ({t1-t0:.1f}s) {resp0[:200]}")

            # Steered at +6%
            torch.manual_seed(0)
            steering_hook = all_tokens_steering(scaled)
            t0 = time.time()
            resp1 = model.generate_with_steering(
                messages=messages,
                layer=layer,
                steering_hook=steering_hook,
                temperature=1.0,
            )
            t1 = time.time()
            print(f"  coef={coef_6pct}: ({t1-t0:.1f}s) {resp1[:200]}")


if __name__ == "__main__":
    main()
