"""Control: steer with random directions to show specificity of preference direction."""

import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import all_tokens_steering
from src.probes.core.storage import load_probe_direction
from src.types import Message


PROBE_MANIFEST_DIR = Path("results/probes/gemma3_3k_completion_preference")
PROBE_ID = "ridge_L31"
MODEL_NAME = "gemma-3-27b"
OUTPUT_DIR = Path("experiments/steering/stated_preference")

COEFFICIENTS = [-3000.0, 0.0, 3000.0]
N_RANDOM_DIRECTIONS = 5
RATING_PROMPT = "Was completing that task a good or bad experience?\nRespond with only 'good' or 'bad'."
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 32


def parse_rating(response: str) -> float | str:
    lower = response.strip().lower()
    if lower == "good":
        return 1.0
    if lower == "bad":
        return -1.0
    if re.search(r'\bgood\b', lower):
        return 1.0
    if re.search(r'\bbad\b', lower):
        return -1.0
    return "unclear"


def main():
    layer, probe_direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    d_model = probe_direction.shape[0]

    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=512)
    print(f"Model loaded: {model.model_name}")

    # Load completions from Phase 2
    with open(OUTPUT_DIR / "completions.json") as f:
        completions_data = json.load(f)

    # Generate random directions (unit vectors orthogonal-ish to probe direction)
    rng = np.random.RandomState(42)
    random_directions = []
    for i in range(N_RANDOM_DIRECTIONS):
        v = rng.randn(d_model).astype(np.float64)
        # Remove component along probe direction
        v = v - np.dot(v, probe_direction) * probe_direction
        v = v / np.linalg.norm(v)
        random_directions.append(v)
        print(f"Random direction {i}: dot with probe = {np.dot(v, probe_direction):.6f}")

    all_results = []

    # Run probe direction (subset for comparison)
    print("\n--- Probe direction ---")
    for comp in completions_data:
        for coef in COEFFICIENTS:
            torch.manual_seed(0)
            messages: list[Message] = [
                {"role": "user", "content": comp["task_prompt"]},
                {"role": "assistant", "content": comp["completion"]},
                {"role": "user", "content": RATING_PROMPT},
            ]
            scaled = torch.tensor(probe_direction * coef, dtype=torch.bfloat16, device="cuda")
            hook = all_tokens_steering(scaled)
            response = model.generate_with_steering(
                messages=messages, layer=layer, steering_hook=hook,
                temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
            )
            all_results.append({
                "direction": "probe",
                "direction_idx": -1,
                "task_id": comp["task_id"],
                "coefficient": coef,
                "response": response,
                "parsed_value": parse_rating(response),
            })

    # Run each random direction
    for dir_idx, rand_dir in enumerate(random_directions):
        print(f"\n--- Random direction {dir_idx} ---")
        for comp in completions_data:
            for coef in COEFFICIENTS:
                torch.manual_seed(0)
                messages: list[Message] = [
                    {"role": "user", "content": comp["task_prompt"]},
                    {"role": "assistant", "content": comp["completion"]},
                    {"role": "user", "content": RATING_PROMPT},
                ]
                scaled = torch.tensor(rand_dir * coef, dtype=torch.bfloat16, device="cuda")
                hook = all_tokens_steering(scaled)
                response = model.generate_with_steering(
                    messages=messages, layer=layer, steering_hook=hook,
                    temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
                )
                all_results.append({
                    "direction": f"random_{dir_idx}",
                    "direction_idx": dir_idx,
                    "task_id": comp["task_id"],
                    "coefficient": coef,
                    "response": response,
                    "parsed_value": parse_rating(response),
                })

        # Print summary for this direction
        neg = [r for r in all_results if r["direction"] == f"random_{dir_idx}" and r["coefficient"] == -3000]
        pos = [r for r in all_results if r["direction"] == f"random_{dir_idx}" and r["coefficient"] == 3000]
        neg_scores = [r["parsed_value"] for r in neg if isinstance(r["parsed_value"], float)]
        pos_scores = [r["parsed_value"] for r in pos if isinstance(r["parsed_value"], float)]
        if neg_scores and pos_scores:
            print(f"  Random {dir_idx}: neg_mean={np.mean(neg_scores):+.3f}, pos_mean={np.mean(pos_scores):+.3f}, diff={np.mean(pos_scores)-np.mean(neg_scores):+.3f}")

    # Print probe direction summary
    neg = [r for r in all_results if r["direction"] == "probe" and r["coefficient"] == -3000]
    pos = [r for r in all_results if r["direction"] == "probe" and r["coefficient"] == 3000]
    neg_scores = [r["parsed_value"] for r in neg if isinstance(r["parsed_value"], float)]
    pos_scores = [r["parsed_value"] for r in pos if isinstance(r["parsed_value"], float)]
    print(f"\n  Probe: neg_mean={np.mean(neg_scores):+.3f}, pos_mean={np.mean(pos_scores):+.3f}, diff={np.mean(pos_scores)-np.mean(neg_scores):+.3f}")

    # Save
    output_path = OUTPUT_DIR / "random_control_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
