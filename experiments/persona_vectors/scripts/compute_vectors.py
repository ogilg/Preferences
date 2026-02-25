"""Compute persona vectors as mean-difference directions + layer selection via Cohen's d."""

import json
from pathlib import Path

import numpy as np

PERSONAS = ["evil", "stem_nerd", "creative_artist", "uncensored", "lazy"]
BASE = Path("results/experiments/persona_vectors")
LAYERS = [8, 15, 23, 31, 37, 43, 49, 55]


def compute_cohens_d(pos: np.ndarray, neg: np.ndarray, direction: np.ndarray) -> float:
    """Cohen's d for projections onto direction."""
    proj_pos = pos @ direction
    proj_neg = neg @ direction
    pooled_std = np.sqrt((np.var(proj_pos, ddof=1) + np.var(proj_neg, ddof=1)) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(proj_pos) - np.mean(proj_neg)) / pooled_std)


def main():
    for persona in PERSONAS:
        print(f"\n{'='*60}")
        print(f"Persona: {persona}")

        pos_path = BASE / persona / "activations" / "pos" / "activations_prompt_last.npz"
        neg_path = BASE / persona / "activations" / "neg" / "activations_prompt_last.npz"
        pos_data = np.load(pos_path, allow_pickle=True)
        neg_data = np.load(neg_path, allow_pickle=True)

        vec_dir = BASE / persona / "vectors"
        vec_dir.mkdir(parents=True, exist_ok=True)

        layer_stats = {}
        for layer in LAYERS:
            pos_acts = pos_data[f"layer_{layer}"]
            neg_acts = neg_data[f"layer_{layer}"]

            direction = np.mean(pos_acts, axis=0) - np.mean(neg_acts, axis=0)
            norm = np.linalg.norm(direction)
            direction_normalized = direction / norm

            # Save in probe-compatible format: [coefs..., intercept=0]
            probe_format = np.append(direction_normalized, 0.0)
            np.save(vec_dir / f"{persona}_L{layer}.npy", probe_format)

            d = compute_cohens_d(pos_acts, neg_acts, direction_normalized)
            layer_stats[str(layer)] = {
                "cohens_d": round(d, 3),
                "direction_norm": round(float(norm), 2),
            }
            print(f"  Layer {layer:2d}: Cohen's d = {d:.3f}, norm = {norm:.2f}")

        best_layer = max(LAYERS, key=lambda l: layer_stats[str(l)]["cohens_d"])
        layer_stats["best_layer"] = best_layer
        print(f"  Best layer: {best_layer} (d = {layer_stats[str(best_layer)]['cohens_d']:.3f})")

        with open(vec_dir / "layer_selection.json", "w") as f:
            json.dump(layer_stats, f, indent=2)

    print("\nDone! All persona vectors computed.")


if __name__ == "__main__":
    main()
