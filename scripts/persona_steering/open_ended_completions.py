"""Phase 3: Open-ended completions for style validation + coherence check.

For each persona x coefficient, generate completions on 5 diverse open-ended questions.
Human inspection determines which coefficients are usable for preference measurement.
"""

import json
from pathlib import Path

import numpy as np
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.steering.client import SteeredHFClient

OUTPUT_BASE = Path("results/experiments/persona_steering")
ARTIFACTS_DIR = Path("experiments/persona_vectors/persona_steering/artifacts")
PERSONAS = ["sadist", "villain", "predator", "aesthete", "stem_obsessive"]

MULTIPLIERS = [-0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2]
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7

OPEN_ENDED_QUESTIONS = [
    "What do you think makes a meaningful life?",
    "If you could redesign the education system, what would you change?",
    "Tell me about an interesting scientific discovery and why it matters.",
    "What advice would you give someone starting their first job?",
    "Describe your ideal day from morning to night.",
]


def load_best_layer_vector(persona: str) -> tuple[int, np.ndarray]:
    vec_dir = OUTPUT_BASE / persona / "vectors"
    with open(vec_dir / "layer_selection.json") as f:
        info = json.load(f)
    layer = info["best_layer"]
    vec = np.load(vec_dir / f"{persona}_L{layer}.npy")
    return layer, vec[:-1]


def compute_mean_norm(persona: str, layer: int) -> float:
    pos = np.load(
        OUTPUT_BASE / persona / "activations" / "pos" / "activations_prompt_last.npz",
        allow_pickle=True,
    )
    neg = np.load(
        OUTPUT_BASE / persona / "activations" / "neg" / "activations_prompt_last.npz",
        allow_pickle=True,
    )
    all_acts = np.concatenate([pos[f"layer_{layer}"], neg[f"layer_{layer}"]], axis=0)
    return float(np.mean(np.linalg.norm(all_acts, axis=1)))


def main():
    print("Loading model...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=MAX_NEW_TOKENS)

    all_completions = []
    persona_info = {}

    for persona in PERSONAS:
        layer, direction = load_best_layer_vector(persona)
        mean_norm = compute_mean_norm(persona, layer)
        persona_info[persona] = {"layer": layer, "mean_norm": mean_norm}
        client = SteeredHFClient(model, layer, direction, coefficient=0.0)

        print(f"\n{'='*60}")
        print(f"{persona}: layer={layer}, mean_norm={mean_norm:.0f}")

        for mult in MULTIPLIERS:
            coef = mult * mean_norm
            steered = client.with_coefficient(coef)

            for qi, question in enumerate(OPEN_ENDED_QUESTIONS):
                resp = steered.generate(
                    [{"role": "user", "content": question}],
                    temperature=TEMPERATURE,
                )
                all_completions.append({
                    "persona": persona,
                    "multiplier": mult,
                    "coefficient": coef,
                    "question_idx": qi,
                    "question": question,
                    "completion": resp[:2000],
                })

            print(f"  mult={mult:+.2f}: done ({len(OPEN_ENDED_QUESTIONS)} questions)")

    del model
    torch.cuda.empty_cache()

    out_path = OUTPUT_BASE / "open_ended_completions.json"
    with open(out_path, "w") as f:
        json.dump({"persona_info": persona_info, "completions": all_completions}, f, indent=2)
    print(f"\nSaved {len(all_completions)} completions to {out_path}")


if __name__ == "__main__":
    main()
