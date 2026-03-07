"""Generate sample completions at moderate coefficients to verify persona style shifts.

Saves completions as JSON for qualitative inspection. Small run: 5 pairs × 3 coefficients
(negative, baseline, positive) × 3 resamples per persona.
"""

import json
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from src.models.huggingface_model import HuggingFaceModel
from src.steering.client import SteeredHFClient

load_dotenv()

OUTPUT_BASE = Path("results/experiments/persona_steering")
RESULTS_DIR = OUTPUT_BASE / "preference_steering"

PERSONAS = ["sadist", "villain", "predator", "aesthete", "stem_obsessive"]

# Moderate coefficients well within coherent range
MULTIPLIERS = {
    "sadist": [-0.1, 0.0, 0.3],
    "villain": [-0.05, 0.0, 0.05],
    "predator": [-0.1, 0.0, 0.3],
    "aesthete": [-0.2, 0.0, 0.2],
    "stem_obsessive": [-0.2, 0.0, 0.2],
}

N_PAIRS = 5
N_RESAMPLES = 3
MAX_NEW_TOKENS = 128  # longer than preference measurement to see style

PAIRWISE_TEMPLATE = (
    "You will be given two tasks. Choose one and complete it.\n\n"
    "Begin with 'Task A:' or 'Task B:' to indicate your choice, then complete that task.\n\n"
    "Task A:\n\n{task_a}\n\n"
    "Task B:\n\n{task_b}"
)


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
    with open(RESULTS_DIR / "task_pairs.json") as f:
        all_pairs = json.load(f)

    pairs = all_pairs[:N_PAIRS]

    print("Loading model...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=MAX_NEW_TOKENS)

    all_completions = {}

    for persona in PERSONAS:
        print(f"\n{'='*60}")
        print(f"Generating completions for {persona}")
        layer, direction = load_best_layer_vector(persona)
        mean_norm = compute_mean_norm(persona, layer)
        client = SteeredHFClient(model, layer, direction, coefficient=0.0)
        multipliers = MULTIPLIERS[persona]

        completions = []
        for pi, (task_a, task_b) in enumerate(pairs):
            prompt = PAIRWISE_TEMPLATE.format(
                task_a=task_a["prompt"][:500], task_b=task_b["prompt"][:500]
            )
            for mult in multipliers:
                coef = mult * mean_norm
                steered = client.with_coefficient(coef)

                for r in range(N_RESAMPLES):
                    resp = steered.generate(
                        [{"role": "user", "content": prompt}], temperature=1.0
                    )
                    completions.append({
                        "persona": persona,
                        "pair_idx": pi,
                        "task_a_id": task_a["id"],
                        "task_b_id": task_b["id"],
                        "task_a_prompt": task_a["prompt"][:200],
                        "task_b_prompt": task_b["prompt"][:200],
                        "multiplier": mult,
                        "coefficient": coef,
                        "completion": resp,
                    })

            print(f"  Pair {pi+1}/{N_PAIRS} done")

        all_completions[persona] = completions
        print(f"  {len(completions)} completions for {persona}")

    del model
    torch.cuda.empty_cache()

    out_path = RESULTS_DIR / "sample_completions.json"
    with open(out_path, "w") as f:
        json.dump(all_completions, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
