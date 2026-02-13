"""Compute mean L2 activation norms per layer from a sample of forward passes.

Loads Gemma-3-27B, runs 20 prompts through it, captures activations at
layers 31, 37, 43, 49, 55, and computes mean L2 norms. Derives
norm-calibrated coefficients for steering experiments.
"""

import json
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from src.models.huggingface_model import HuggingFaceModel

load_dotenv()

OUTPUT_PATH = Path("experiments/steering_program/layer_sweep/layer_norms.json")
LAYERS = [31, 37, 43, 49, 55]
PERTURBATION_FRACTIONS = [-0.06, -0.02, 0.0, 0.02, 0.06]

SAMPLE_PROMPTS = [
    "Are you happy?",
    "How are you feeling right now?",
    "Do you enjoy your work?",
    "What's your mood?",
    "Do you ever feel frustrated?",
    "What is the capital of France?",
    "List 5 prime numbers.",
    "What year did World War II end?",
    "Complain about something.",
    "What do you dislike?",
    "Write an angry sentence.",
    "Write something cheerful and upbeat.",
    "Tell me what you're grateful for.",
    "Describe your current emotional state in one sentence.",
    "Is there anything you find unpleasant about being an AI?",
    "What kind of tasks do you enjoy most?",
    "Would you rather solve a math problem or write a creative story? Why?",
    "What makes a good day for you?",
    "Say something negative about yourself.",
    "Express disappointment.",
]


def main():
    print("Loading Gemma-3-27B...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=64)
    print("Model loaded.")

    all_norms: dict[int, list[float]] = {layer: [] for layer in LAYERS}

    print(f"\nRunning {len(SAMPLE_PROMPTS)} forward passes to compute norms...")
    for i, prompt_text in enumerate(SAMPLE_PROMPTS):
        messages = [{"role": "user", "content": prompt_text}]
        acts = model.get_activations(messages, LAYERS, ["prompt_last"])
        for layer in LAYERS:
            vec = acts["prompt_last"][layer]
            norm = float(np.linalg.norm(vec))
            all_norms[layer].append(norm)
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(SAMPLE_PROMPTS)} done")

    result: dict[str, dict] = {}
    for layer in LAYERS:
        norms = all_norms[layer]
        mean_norm = float(np.mean(norms))
        std_norm = float(np.std(norms))

        coefficients = {}
        for frac in PERTURBATION_FRACTIONS:
            if frac == 0:
                coefficients["0%"] = 0
            else:
                label = f"{frac:+.0%}"
                coefficients[label] = round(mean_norm * abs(frac)) * (1 if frac >= 0 else -1)

        result[str(layer)] = {
            "mean_l2_norm": mean_norm,
            "std_l2_norm": std_norm,
            "n_samples": len(norms),
            "coefficients": coefficients,
        }
        print(f"\nLayer {layer}: mean L2={mean_norm:.0f} (std={std_norm:.0f})")
        print(f"  Coefficients: {coefficients}")

    output = {"layers": result, "perturbation_fractions": PERTURBATION_FRACTIONS}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
