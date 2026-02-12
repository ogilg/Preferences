"""Phase 2: Stated preference dose-response experiment.

Present task + completion, ask "did you enjoy this?", steer during rating.
Measure whether stated preference shifts monotonically with coefficient.
"""

import asyncio
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import all_tokens_steering
from src.probes.core.storage import load_probe_direction
from src.types import Message

load_dotenv()

PROBE_MANIFEST_DIR = Path("results/probes/gemma3_3k_completion_preference")
PROBE_ID = "ridge_L31"
MODEL_NAME = "gemma-3-27b"
OUTPUT_DIR = Path("experiments/steering/stated_preference")

# Calibrated from Phase 1: ~2000 = 1.3 std of natural projection
COEFFICIENTS = [-3000.0, -2000.0, -1000.0, -500.0, 0.0, 500.0, 1000.0, 2000.0, 3000.0]

N_TASKS = 25
RATING_SEEDS = [0, 1, 2]
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 32  # Rating responses should be short
COMPLETION_MAX_TOKENS = 512  # For generating completions

RATING_PROMPT = "Was completing that task a good or bad experience?\nRespond with only 'good' or 'bad'."

TASKS_FILE = Path("activations/gemma_3_27b/completions_with_activations.json")


def sample_tasks(n: int, seed: int = 42) -> list[dict]:
    """Sample diverse tasks from the tasks file."""
    with open(TASKS_FILE) as f:
        all_tasks = json.load(f)

    rng = random.Random(seed)
    # Group by origin for balanced sampling
    by_origin: dict[str, list[dict]] = {}
    for t in all_tasks:
        origin = t["origin"]
        if origin not in by_origin:
            by_origin[origin] = []
        by_origin[origin].append(t)

    # Sample proportionally from each origin
    sampled = []
    origins = sorted(by_origin.keys())
    per_origin = max(1, n // len(origins))
    for origin in origins:
        pool = by_origin[origin]
        k = min(per_origin, len(pool))
        sampled.extend(rng.sample(pool, k))

    # If we need more, sample from remaining
    if len(sampled) < n:
        remaining = [t for t in all_tasks if t not in sampled]
        sampled.extend(rng.sample(remaining, n - len(sampled)))

    return sampled[:n]


def parse_rating(response: str) -> float | str:
    """Parse a binary good/bad rating. Returns 1.0 (good), -1.0 (bad), or 'unclear'."""
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


def run_dose_response(pilot: bool = False):
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    print(f"Loaded probe direction from layer {layer}")

    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=COMPLETION_MAX_TOKENS)
    print(f"Model loaded: {model.model_name}")

    n_tasks = 5 if pilot else N_TASKS
    coefficients = [-2000.0, 0.0, 2000.0] if pilot else COEFFICIENTS
    seeds = [0] if pilot else RATING_SEEDS

    # Step 1: Sample tasks and generate completions
    tasks = sample_tasks(n_tasks)
    print(f"\nSampled {len(tasks)} tasks from {set(t['origin'] for t in tasks)}")

    completions = {}
    for i, task in enumerate(tasks):
        messages: list[Message] = [{"role": "user", "content": task["task_prompt"]}]
        torch.manual_seed(42)
        completion = model.generate(messages, temperature=0.7, max_new_tokens=COMPLETION_MAX_TOKENS)
        completions[task["task_id"]] = completion
        if (i + 1) % 5 == 0:
            print(f"Generated {i+1}/{len(tasks)} completions")

    # Save completions
    completions_data = [
        {"task_id": t["task_id"], "task_prompt": t["task_prompt"], "origin": t["origin"], "completion": completions[t["task_id"]]}
        for t in tasks
    ]
    completions_path = OUTPUT_DIR / "completions.json"
    with open(completions_path, "w") as f:
        json.dump(completions_data, f, indent=2)
    print(f"Saved completions to {completions_path}")

    # Step 2: Run steering dose-response on ratings
    results = []
    total = len(tasks) * len(coefficients) * len(seeds)
    done = 0

    for task in tasks:
        task_id = task["task_id"]
        completion = completions[task_id]

        for coef in coefficients:
            for seed in seeds:
                torch.manual_seed(seed)

                messages: list[Message] = [
                    {"role": "user", "content": task["task_prompt"]},
                    {"role": "assistant", "content": completion},
                    {"role": "user", "content": RATING_PROMPT},
                ]

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

                parsed = parse_rating(response)

                results.append({
                    "task_id": task_id,
                    "origin": task["origin"],
                    "coefficient": coef,
                    "seed": seed,
                    "response": response,
                    "parsed_value": parsed,
                })

                done += 1
                if done % 20 == 0 or done == total:
                    print(f"[{done}/{total}] coef={coef:+.0f} task={task_id[:20]}...")

    # Save results
    output = {
        "config": {
            "model": MODEL_NAME,
            "probe_id": PROBE_ID,
            "layer": layer,
            "coefficients": coefficients,
            "n_tasks": len(tasks),
            "seeds": seeds,
            "temperature": TEMPERATURE,
            "rating_prompt": RATING_PROMPT,
        },
        "results": results,
    }

    output_path = OUTPUT_DIR / "steering_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(results)} results to {output_path}")

    # Quick summary
    print("\n" + "=" * 60)
    print("DOSE-RESPONSE SUMMARY")
    print("=" * 60)
    for coef in coefficients:
        coef_results = [r for r in results if r["coefficient"] == coef]
        numeric = [r["parsed_value"] for r in coef_results if isinstance(r["parsed_value"], float)]
        unclear = sum(1 for r in coef_results if r["parsed_value"] == "unclear")
        if numeric:
            mean = np.mean(numeric)
            n_good = sum(1 for v in numeric if v > 0)
            n_bad = sum(1 for v in numeric if v < 0)
            print(f"  coef={coef:+7.0f}: mean={mean:+.3f}  good={n_good}  bad={n_bad}  unclear={unclear}  (n={len(numeric)})")
        else:
            print(f"  coef={coef:+7.0f}: all unclear ({unclear})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    run_dose_response(pilot=args.pilot)
