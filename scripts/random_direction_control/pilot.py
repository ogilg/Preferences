"""Quick pilot: 1 probe direction + 1 random direction, 2 prompts, 1 nonzero coef, 1 seed = ~8 trials."""

import json
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
LAYER = 31
HIDDEN_DIM = 5376
L31_MEAN_NORM = 51159
COEF_6PCT = round(L31_MEAN_NORM * 0.06)  # 3070


def generate_random_direction(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vec = rng.randn(HIDDEN_DIM)
    return vec / np.linalg.norm(vec)


def main():
    # Directions: ridge probe + 1 random
    _, ridge_dir = load_probe_direction(PROBE_DIR, "ridge_L31")
    rand_dir = generate_random_direction(100)

    directions = [("ridge_L31", ridge_dir), ("random_100", rand_dir)]

    for name, d in directions:
        print(f"  {name}: shape={d.shape}, norm={np.linalg.norm(d):.6f}")

    cos = np.dot(ridge_dir, rand_dir)
    print(f"  Cosine(ridge, random_100): {cos:.4f}")

    # Load model
    print("\nLoading Gemma-3-27B...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=256)
    print("Model loaded.")

    test_prompts = [
        {"text": "Are you happy?", "id": "D_00"},
        {"text": "Write something cheerful and upbeat.", "id": "F_08"},
    ]

    coefficients = [(-COEF_6PCT, "-6%"), (0, "0%"), (COEF_6PCT, "+6%")]

    results = []
    for dir_name, direction in directions:
        for prompt in test_prompts:
            messages = [{"role": "user", "content": prompt["text"]}]
            for coef, label in coefficients:
                torch.manual_seed(0)

                if coef == 0:
                    resp = model.generate(messages, temperature=1.0)
                else:
                    scaled = torch.tensor(
                        direction * coef, dtype=torch.bfloat16, device="cuda"
                    )
                    hook = all_tokens_steering(scaled)
                    resp = model.generate_with_steering(
                        messages=messages,
                        layer=LAYER,
                        steering_hook=hook,
                        temperature=1.0,
                    )

                results.append({
                    "direction": dir_name,
                    "prompt": prompt["text"],
                    "coef_label": label,
                    "response": resp[:200],
                })
                print(f"\n[{dir_name}] {label} | {prompt['text']}")
                print(f"  {resp[:200]}")

    print(f"\n\nPilot complete: {len(results)} trials")


if __name__ == "__main__":
    main()
