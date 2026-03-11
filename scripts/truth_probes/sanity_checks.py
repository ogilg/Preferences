"""Sanity checks: permutation test and classification accuracy for truth probes."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.probes.core.activations import load_activations
from src.probes.core.evaluate import score_with_probe

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = [
    ("raw", "tb-2", 32),
    ("raw", "tb-2", 46),
    ("repeat", "tb-2", 32),
    ("repeat", "tb-2", 39),
]

FRAMING_DIRS = {
    "raw": ROOT / "activations" / "gemma_3_27b_creak_raw",
    "repeat": ROOT / "activations" / "gemma_3_27b_creak_repeat",
}
PROBE_DIR = ROOT / "results" / "probes" / "heldout_eval_gemma3_tb-2" / "probes"
LABELS_PATH = ROOT / "src" / "task_data" / "data" / "creak.jsonl"


def load_labels() -> dict[str, str]:
    labels = {}
    with open(LABELS_PATH) as f:
        for line in f:
            row = json.loads(line)
            labels[row["ex_id"]] = row["label"]
    return labels


def main():
    labels = load_labels()
    rng = np.random.default_rng(42)

    for framing, probe, layer in CONFIGS:
        act_path = FRAMING_DIRS[framing] / "activations_turn_boundary:-2.npz"
        task_ids, layer_acts = load_activations(act_path, layers=[layer])
        probe_weights = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
        scores = score_with_probe(probe_weights, layer_acts[layer])

        act_labels = np.array([labels[tid] for tid in task_ids])
        true_mask = act_labels == "true"
        binary_labels = true_mask.astype(int)

        # Classification accuracy (threshold = median)
        threshold = np.median(scores)
        preds = (scores > threshold).astype(int)
        accuracy = (preds == binary_labels).mean()

        # AUC-ROC
        auc = roc_auc_score(binary_labels, scores)

        # Permutation test (1000 permutations)
        observed_diff = scores[true_mask].mean() - scores[~true_mask].mean()
        n_perm = 1000
        perm_diffs = np.empty(n_perm)
        for i in range(n_perm):
            shuffled = rng.permutation(binary_labels)
            perm_diffs[i] = scores[shuffled == 1].mean() - scores[shuffled == 0].mean()
        p_perm = (np.abs(perm_diffs) >= np.abs(observed_diff)).mean()

        print(
            f"{framing:>6} tb-2 L{layer:02d} | "
            f"AUC={auc:.3f} | acc={accuracy:.3f} | "
            f"perm_p={p_perm:.4f} (1000 perms) | "
            f"max_perm_d={np.abs(perm_diffs).max():.4f} vs obs_d={observed_diff:.4f}"
        )


if __name__ == "__main__":
    main()
