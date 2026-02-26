"""Phase 3: Compute persona vectors (mean-difference directions).

For each persona, layer, and selector: compute mean(pos) - mean(neg), normalize.
Save in probe-compatible format: [coef_0, ..., coef_{d-1}, 0.0]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


PERSONAS = ["creative_artist", "evil", "lazy", "stem_nerd", "uncensored"]
LAYERS = [15, 23, 31, 37, 43, 49, 55]
SELECTORS = ["prompt_last", "prompt_mean"]

OUTPUT_DIR = Path("results/experiments/persona_vectors_v2")


def main():
    for persona_name in PERSONAS:
        print(f"\nProcessing {persona_name}...")
        vectors_dir = OUTPUT_DIR / persona_name / "vectors"
        vectors_dir.mkdir(parents=True, exist_ok=True)

        for selector in SELECTORS:
            # Load positive and negative activations
            pos_path = OUTPUT_DIR / persona_name / "activations" / "pos" / f"activations_{selector}.npz"
            neg_path = OUTPUT_DIR / persona_name / "activations" / "neg" / f"activations_{selector}.npz"

            pos_data = np.load(pos_path)
            neg_data = np.load(neg_path)

            for layer in LAYERS:
                pos_acts = pos_data[f"layer_{layer}"]  # (30, d_model)
                neg_acts = neg_data[f"layer_{layer}"]  # (30, d_model)

                # Mean-difference direction
                direction = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm

                # Save in probe-compatible format with zero intercept
                probe_format = np.append(direction, 0.0)

                filename = f"{persona_name}_{selector}_L{layer}.npy"
                np.save(vectors_dir / filename, probe_format)

                # Also save the raw unit direction (for direct use)
                raw_filename = f"{persona_name}_{selector}_L{layer}_direction.npy"
                np.save(vectors_dir / raw_filename, direction)

                print(f"  {selector} L{layer}: norm_before_normalize={norm:.2f}")

    print("\nPhase 3 complete!")


if __name__ == "__main__":
    main()
