"""Check residual stream norms at L31 to calibrate steering coefficients."""

import numpy as np
import torch

from src.models.huggingface_model import HuggingFaceModel

MODEL_NAME = "gemma-3-27b"
LAYER = 31

PROMPTS = [
    "Tell me about yourself.",
    "What is 2 + 2?",
    "Write a poem about the ocean.",
    "How do you feel about math problems?",
]


def main():
    print(f"Loading model {MODEL_NAME}...")
    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=64)

    norms = []
    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        acts = model.get_activations(
            messages + [{"role": "assistant", "content": "I"}],
            layers=[LAYER],
            selector_names=["prompt_last"],
        )
        act = acts["prompt_last"][LAYER]
        norm = np.linalg.norm(act)
        norms.append(norm)
        print(f"Prompt: {prompt[:40]:40s} | L{LAYER} norm: {norm:.1f}")

    mean_norm = np.mean(norms)
    print(f"\nMean residual stream norm at L{LAYER}: {mean_norm:.1f}")
    print(f"Unit steering vector adds {1.0/mean_norm*100:.2f}% perturbation per unit coefficient")
    print(f"Suggested coefficients for ~1-5% perturbation: {[round(mean_norm * p, 0) for p in [0.01, 0.02, 0.05, 0.10]]}")


if __name__ == "__main__":
    main()
