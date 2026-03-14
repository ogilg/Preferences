"""Cross-selector probe transfer analysis.

For each of 5 probes (tb-1..tb-5) × 5 layers × 6 extraction positions,
compute Pearson r and pairwise accuracy against Thurstonian scores.
Report best layer per (probe, position) cell as 5×6 heatmaps.
"""

import csv
from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from src.probes.core.activations import load_activations

ROOT = Path(__file__).resolve().parents[2]

PROBE_DIRS = {
    f"tb-{i}": ROOT / f"results/probes/heldout_eval_gemma3_tb-{i}"
    for i in range(1, 6)
}

ACTIVATION_FILES = {}
for i in range(1, 6):
    ACTIVATION_FILES[f"tb-{i}"] = (
        ROOT / f"activations/gemma_3_27b_turn_boundary_sweep/activations_turn_boundary:-{i}.npz"
    )
ACTIVATION_FILES["task_mean"] = (
    ROOT / "activations/gemma_3_27b_turn_boundary_sweep/activations_task_mean.npz"
)

EVAL_SCORES_CSV = (
    ROOT / "results/experiments/main_probes/gemma3_4k_pre_task"
    "/pre_task_active_learning"
    "/completion_preference_gemma-3-27b_completion_canonical_seed0"
    "/thurstonian_a67822c5.csv"
)

LAYERS = [25, 32, 39, 46, 53]
PROBE_NAMES = [f"tb-{i}" for i in range(1, 6)]
POSITION_NAMES = [f"tb-{i}" for i in range(1, 6)] + ["task_mean"]

# Gemma 3 IT turn-boundary template: <end_of_turn>\n<start_of_turn>model\n
TOKEN_LABELS = {
    "tb-1": "tb-1 (\\n)",
    "tb-2": "tb-2 (model)",
    "tb-3": "tb-3 (<start_of_turn>)",
    "tb-4": "tb-4 (\\n)",
    "tb-5": "tb-5 (<end_of_turn>)",
    "task_mean": "task_mean",
}

ASSETS_DIR = ROOT / "experiments/eot_probes/turn_boundary_sweep/assets"


def load_eval_scores() -> dict[str, float]:
    scores = {}
    with open(EVAL_SCORES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row["task_id"]] = float(row["mu"])
    return scores


def pairwise_accuracy(pred_scores: np.ndarray, true_scores: np.ndarray) -> float:
    n = len(pred_scores)
    concordant = 0
    total = 0
    for i, j in combinations(range(n), 2):
        true_diff = true_scores[i] - true_scores[j]
        if true_diff == 0:
            continue
        pred_diff = pred_scores[i] - pred_scores[j]
        total += 1
        if (true_diff > 0 and pred_diff > 0) or (true_diff < 0 and pred_diff < 0):
            concordant += 1
    if total == 0:
        raise ValueError("No valid pairs for pairwise accuracy")
    return concordant / total


def main():
    eval_scores = load_eval_scores()
    eval_task_ids = set(eval_scores.keys())
    print(f"Loaded {len(eval_scores)} eval scores")

    # Preload all activations (filtered to eval tasks)
    act_cache: dict[str, tuple[np.ndarray, dict[int, np.ndarray]]] = {}
    for pos_name, act_path in ACTIVATION_FILES.items():
        assert act_path.exists(), f"Missing: {act_path}"
        task_ids, act_dict = load_activations(act_path, task_id_filter=eval_task_ids, layers=LAYERS)
        act_cache[pos_name] = (task_ids, act_dict)
        print(f"  {pos_name}: {len(task_ids)} tasks")

    # Load all probes
    probe_cache: dict[str, dict[int, np.ndarray]] = {}
    for probe_name, probe_dir in PROBE_DIRS.items():
        probe_cache[probe_name] = {}
        for layer in LAYERS:
            probe_path = probe_dir / "probes" / f"probe_ridge_L{layer}.npy"
            assert probe_path.exists(), f"Missing: {probe_path}"
            probe_cache[probe_name][layer] = np.load(probe_path)

    # Compute metrics for every (probe, position, layer)
    # results[probe_name][pos_name] = {"best_r": float, "best_acc": float, "best_layer_r": int, "best_layer_acc": int, "all_layers": {layer: (r, acc)}}
    results: dict[str, dict[str, dict]] = {}

    for probe_name in PROBE_NAMES:
        results[probe_name] = {}
        for pos_name in POSITION_NAMES:
            task_ids, act_dict = act_cache[pos_name]
            # Build aligned arrays
            tid_to_idx = {tid: i for i, tid in enumerate(task_ids)}
            matched_indices = []
            matched_true = []
            for tid in task_ids:
                if tid in eval_scores:
                    matched_indices.append(tid_to_idx[tid])
                    matched_true.append(eval_scores[tid])
            matched_indices = np.array(matched_indices)
            matched_true = np.array(matched_true)

            layer_results = {}
            for layer in LAYERS:
                weights = probe_cache[probe_name][layer]
                coef = weights[:-1]
                intercept = weights[-1]
                X = act_dict[layer][matched_indices]
                pred = X @ coef + intercept

                r_val, _ = pearsonr(pred, matched_true)
                acc = pairwise_accuracy(pred, matched_true)
                layer_results[layer] = (float(r_val), float(acc))

            best_layer_r = max(layer_results, key=lambda l: layer_results[l][0])
            best_layer_acc = max(layer_results, key=lambda l: layer_results[l][1])

            results[probe_name][pos_name] = {
                "best_r": layer_results[best_layer_r][0],
                "best_acc": layer_results[best_layer_acc][1],
                "best_layer_r": best_layer_r,
                "best_layer_acc": best_layer_acc,
                "all_layers": layer_results,
            }
            print(f"  {probe_name} × {pos_name}: best r={layer_results[best_layer_r][0]:.4f} (L{best_layer_r}), best acc={layer_results[best_layer_acc][1]:.4f} (L{best_layer_acc})")

    # Build heatmap matrices
    r_matrix = np.zeros((len(PROBE_NAMES), len(POSITION_NAMES)))
    acc_matrix = np.zeros((len(PROBE_NAMES), len(POSITION_NAMES)))
    for i, probe_name in enumerate(PROBE_NAMES):
        for j, pos_name in enumerate(POSITION_NAMES):
            r_matrix[i, j] = results[probe_name][pos_name]["best_r"]
            acc_matrix[i, j] = results[probe_name][pos_name]["best_acc"]

    # Save CSV
    csv_path = ASSETS_DIR / "cross_selector_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["probe", "position", "best_layer_r", "pearson_r", "best_layer_acc", "pairwise_acc"])
        for probe_name in PROBE_NAMES:
            for pos_name in POSITION_NAMES:
                res = results[probe_name][pos_name]
                writer.writerow([
                    probe_name, pos_name,
                    res["best_layer_r"], f"{res['best_r']:.4f}",
                    res["best_layer_acc"], f"{res['best_acc']:.4f}",
                ])
    print(f"\nSaved CSV to {csv_path}")

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, matrix, title, fmt in [
        (axes[0], r_matrix, "Pearson r (best layer)", ".3f"),
        (axes[1], acc_matrix, "Pairwise accuracy (best layer)", ".3f"),
    ]:
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(POSITION_NAMES)))
        ax.set_xticklabels([TOKEN_LABELS[p] for p in POSITION_NAMES], rotation=45, ha="right")
        ax.set_yticks(range(len(PROBE_NAMES)))
        ax.set_yticklabels([TOKEN_LABELS[p] for p in PROBE_NAMES])
        ax.set_xlabel("Extraction position")
        ax.set_ylabel("Probe trained at")
        ax.set_title(title)
        # Annotate cells
        for i in range(len(PROBE_NAMES)):
            for j in range(len(POSITION_NAMES)):
                ax.text(j, i, f"{matrix[i, j]:{fmt}}", ha="center", va="center", fontsize=9,
                        color="white" if matrix[i, j] > matrix.mean() else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Cross-selector probe transfer (Gemma-3-27B)", fontsize=13)
    fig.tight_layout()

    plot_path = ASSETS_DIR / "plot_031326_cross_selector_heatmap.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
