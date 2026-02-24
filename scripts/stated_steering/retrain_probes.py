"""Retrain ridge probes for gemma3_10k_heldout_std_raw.

The .npy probe files are gitignored. This script reconstructs them from
activations and Thurstonian scores (which ARE in the repo).

Uses best_alpha values from the existing manifest.json.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ACTIVATIONS_PATH = REPO_ROOT / "activations" / "gemma_3_27b" / "activations_prompt_last.npz"
THURSTONIAN_CSV = (
    REPO_ROOT / "results" / "experiments" / "gemma3_10k_run1"
    / "pre_task_active_learning"
    / "completion_preference_gemma-3-27b_completion_canonical_seed0"
    / "thurstonian_80fa9dc8.csv"
)
PROBE_OUT_DIR = REPO_ROOT / "results" / "probes" / "gemma3_10k_heldout_std_raw" / "probes"

# Best alphas from existing manifest.json (alpha sweep results kept in git)
BEST_ALPHAS = {
    15: 1000.0,
    31: 4641.588833612773,
    37: 1000.0,
    43: 1000.0,
    49: 4641.588833612773,
    55: 4641.588833612773,
}

LAYERS = [15, 31, 37, 43, 49, 55]


def main() -> None:
    PROBE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Thurstonian scores from {THURSTONIAN_CSV}")
    df = pd.read_csv(THURSTONIAN_CSV)
    score_map: dict[str, float] = dict(zip(df["task_id"], df["mu"]))
    print(f"  {len(score_map)} tasks with Thurstonian scores")

    print(f"Loading activations from {ACTIVATIONS_PATH}")
    data = np.load(ACTIVATIONS_PATH, allow_pickle=True)
    all_task_ids = data["task_ids"]
    print(f"  {len(all_task_ids)} tasks in activations file")

    # Find common task IDs
    common_ids = [tid for tid in all_task_ids if tid in score_map]
    common_set = set(common_ids)
    mask = np.array([tid in common_set for tid in all_task_ids])
    task_ids_filtered = all_task_ids[mask]
    scores = np.array([score_map[tid] for tid in task_ids_filtered])

    print(f"  {len(common_ids)} tasks with both activations and scores")

    for layer in LAYERS:
        print(f"\nTraining ridge probe at layer {layer}...")
        acts = data[f"layer_{layer}"][mask]  # (n_tasks, d_model)
        print(f"  Activations shape: {acts.shape}")

        # Standardize (standardize=True in config)
        scaler = StandardScaler()
        acts_std = scaler.fit_transform(acts)

        alpha = BEST_ALPHAS[layer]
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(acts_std, scores)

        # Convert weights back to raw activation space
        coef_raw = ridge.coef_ / scaler.scale_
        intercept = float(ridge.intercept_)

        # Save: [coef_0, ..., coef_d, intercept]
        weights = np.concatenate([coef_raw, [intercept]])
        out_path = PROBE_OUT_DIR / f"probe_ridge_L{layer}.npy"
        np.save(out_path, weights)
        print(f"  Saved to {out_path} (shape: {weights.shape})")

        # Quick sanity check: correlation with scores
        preds = acts_std @ ridge.coef_ + ridge.intercept_
        r = np.corrcoef(preds, scores)[0, 1]
        print(f"  Training R (ridge in std space): {r:.4f}")

    print("\nDone. All probe files saved.")


if __name__ == "__main__":
    main()
