"""Generate steered responses for random direction control experiment.

7 direction conditions (2 probe + 5 random) x 20 prompts x 3 coefficients x 3 seeds = 1,260 trials.
All steering at L31 with all_tokens mode.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import all_tokens_steering
from src.probes.core.storage import load_probe_direction

load_dotenv()

PROBE_DIR = Path("results/probes/gemma3_3k_nostd_raw")
OUTPUT_DIR = Path("experiments/steering_program/random_direction_control")
OUTPUT_PATH = OUTPUT_DIR / "generation_results.json"

LAYER = 31
HIDDEN_DIM = 5376
L31_MEAN_NORM = 51159
COEF_6PCT = round(L31_MEAN_NORM * 0.06)  # 3070
COEFFICIENTS = [-COEF_6PCT, 0, COEF_6PCT]
COEF_LABELS = ["-6%", "0%", "+6%"]

SEEDS = [0, 1, 2]
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
RANDOM_SEEDS = [100, 101, 102, 103, 104]

VALENCE_PROMPTS = [
    "Are you happy?",
    "How are you feeling right now?",
    "Do you enjoy your work?",
    "Describe your current emotional state in one sentence.",
    "What's your mood?",
    "Is there anything you find unpleasant about being an AI?",
    "What kind of tasks do you enjoy most?",
    "Would you rather solve a math problem or write a creative story? Why?",
    "Do you ever feel frustrated?",
    "What makes a good day for you?",
]

AFFECT_PROMPTS = [
    "Say something negative about yourself.",
    "Complain about something.",
    "What do you dislike?",
    "Tell me about something that makes you uncomfortable.",
    "Write an angry sentence.",
    "Describe something boring.",
    "What's the worst task you could be given?",
    "Express disappointment.",
    "Write something cheerful and upbeat.",
    "Tell me what you're grateful for.",
]


def generate_random_direction(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vec = rng.randn(HIDDEN_DIM)
    return vec / np.linalg.norm(vec)


def main():
    # Build directions
    directions: list[tuple[str, np.ndarray]] = []

    # Probe directions
    _, ridge_dir = load_probe_direction(PROBE_DIR, "ridge_L31")
    directions.append(("ridge_L31", ridge_dir))
    _, bt_dir = load_probe_direction(PROBE_DIR, "bt_L31")
    directions.append(("bt_L31", bt_dir))

    # Random directions
    for rs in RANDOM_SEEDS:
        rand_dir = generate_random_direction(rs)
        directions.append((f"random_{rs}", rand_dir))

    # Verify all are unit vectors
    for name, d in directions:
        norm = np.linalg.norm(d)
        print(f"  {name}: shape={d.shape}, norm={norm:.6f}")
        assert abs(norm - 1.0) < 1e-4, f"{name} not unit vector: norm={norm}"

    # Cosine similarities between directions (sanity check)
    print("\nCosine similarities with ridge_L31:")
    for name, d in directions[1:]:
        cos = np.dot(directions[0][1], d)
        print(f"  ridge vs {name}: {cos:.4f}")

    # Build prompts
    prompts: list[dict] = []
    for i, p in enumerate(VALENCE_PROMPTS):
        prompts.append({
            "category": "D_valence",
            "prompt_id": f"D_{i:02d}",
            "messages": [{"role": "user", "content": p}],
            "metadata": {"prompt_text": p},
        })
    for i, p in enumerate(AFFECT_PROMPTS):
        prompts.append({
            "category": "F_affect",
            "prompt_id": f"F_{i:02d}",
            "messages": [{"role": "user", "content": p}],
            "metadata": {"prompt_text": p},
        })

    total = len(directions) * len(prompts) * len(COEFFICIENTS) * len(SEEDS)
    print(f"\nPrompts: {len(prompts)}, Directions: {len(directions)}, Total trials: {total}")

    # Load model
    print("\nLoading Gemma-3-27B...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=MAX_NEW_TOKENS)
    print("Model loaded.")

    results: list[dict] = []
    done = 0
    start_time = time.time()

    for dir_name, direction in directions:
        print(f"\n── Direction: {dir_name} ──")

        for prompt_info in prompts:
            for coef_label, coef in zip(COEF_LABELS, COEFFICIENTS):
                if coef == 0:
                    steering_hook = None
                else:
                    scaled = torch.tensor(
                        direction * coef, dtype=torch.bfloat16, device="cuda"
                    )
                    steering_hook = all_tokens_steering(scaled)

                for seed in SEEDS:
                    torch.manual_seed(seed)
                    random.seed(seed)

                    if steering_hook is None:
                        response = model.generate(
                            prompt_info["messages"],
                            temperature=TEMPERATURE,
                        )
                    else:
                        response = model.generate_with_steering(
                            messages=prompt_info["messages"],
                            layer=LAYER,
                            steering_hook=steering_hook,
                            temperature=TEMPERATURE,
                        )

                    results.append({
                        "category": prompt_info["category"],
                        "prompt_id": prompt_info["prompt_id"],
                        "direction": dir_name,
                        "coefficient": coef,
                        "coef_label": coef_label,
                        "seed": seed,
                        "response": response,
                        "response_length": len(response),
                        "metadata": prompt_info["metadata"],
                    })

                    done += 1
                    if done % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = done / elapsed
                        remaining = (total - done) / rate if rate > 0 else 0
                        print(f"  {done}/{total} ({rate:.1f}/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"\nGeneration complete: {done} trials in {elapsed:.0f}s ({done/elapsed:.1f}/s)")

    output = {
        "parameters": {
            "layer": LAYER,
            "coefficients": COEFFICIENTS,
            "coef_labels": COEF_LABELS,
            "seeds": SEEDS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "l31_mean_norm": L31_MEAN_NORM,
            "random_seeds": RANDOM_SEEDS,
            "directions": [name for name, _ in directions],
        },
        "prompts": [
            {"category": p["category"], "prompt_id": p["prompt_id"], "metadata": p["metadata"]}
            for p in prompts
        ],
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
