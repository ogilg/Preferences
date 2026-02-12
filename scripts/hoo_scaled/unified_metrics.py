"""Compute both Pearson r and pairwise accuracy for all HOO conditions.

For each fold, loads saved probe weights, computes predicted scores for
held-out tasks, then evaluates both metrics on the same held-out set.

Usage:
    python scripts/hoo_scaled/unified_metrics.py
"""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

from src.probes.bradley_terry.data import PairwiseActivationData
from src.probes.bradley_terry.training import weighted_accuracy
from src.probes.core.activations import load_activations
from src.probes.data_loading import load_thurstonian_scores, load_pairwise_measurements
from src.probes.residualization import build_task_groups


# --- Config ---

PROBE_DIRS = {
    "Ridge raw": "results/probes/hoo_scaled_raw",
    "Ridge topic-demeaned": "results/probes/hoo_scaled_demeaned",
    "Content baseline": "results/probes/hoo_scaled_st_baseline",
    "BT raw": "results/probes/hoo_scaled_bt",
}

RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/"
               "completion_preference_gemma-3-27b_completion_canonical_seed0")
ACTIVATIONS_PATH = "activations/gemma_3_27b/activations_prompt_last.npz"
ST_ACTIVATIONS_PATH = "activations/sentence_transformer/embeddings.npz"
TOPICS_JSON = Path("src/analysis/topic_classification/output/topics_v2.json")
HOO_GROUPS = ["math", "fiction", "coding", "persuasive_writing",
              "content_generation", "summarization", "knowledge_qa", "harmful_request"]
HOLD_OUT_SIZE = 3
LAYERS = [31, 43, 55]
OUTPUT_PATH = Path("results/probes/hoo_scaled_unified_metrics.json")


def load_probe(probe_dir: str, fold_idx: int, method: str, layer: int) -> np.ndarray:
    path = Path(probe_dir) / "probes" / f"probe_hoo_fold{fold_idx}_{method}_L{layer}.npy"
    return np.load(path)


def predict_scores(weights: np.ndarray, activations: np.ndarray, task_indices: np.ndarray) -> np.ndarray:
    w, b = weights[:-1], weights[-1]
    return activations[task_indices] @ w + b


def compute_pairwise_accuracy(
    predicted_scores: np.ndarray,
    task_indices: np.ndarray,
    bt_eval_data: PairwiseActivationData,
) -> float:
    """Pairwise accuracy from predicted scalar scores on held-out BT pairs."""
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(task_indices)}

    total_correct = 0.0
    total_weight = 0.0
    for k in range(len(bt_eval_data.pairs)):
        i, j = bt_eval_data.pairs[k]
        if int(i) not in idx_to_pos or int(j) not in idx_to_pos:
            continue
        s_i = predicted_scores[idx_to_pos[int(i)]]
        s_j = predicted_scores[idx_to_pos[int(j)]]
        wins_i = bt_eval_data.wins_i[k]
        wins_j = bt_eval_data.total[k] - bt_eval_data.wins_i[k]
        if s_i > s_j:
            total_correct += wins_i
        elif s_j > s_i:
            total_correct += wins_j
        else:
            total_correct += 0.5 * (wins_i + wins_j)
        total_weight += wins_i + wins_j

    if total_weight == 0:
        return float('nan')
    return total_correct / total_weight


def main():
    # Load data
    print("Loading activations...")
    task_ids, activations = load_activations(ACTIVATIONS_PATH, layers=LAYERS)

    st_task_ids, st_activations = load_activations(ST_ACTIVATIONS_PATH, layers=[0])

    print("Loading measurements...")
    measurements = load_pairwise_measurements(RUN_DIR)

    print("Loading Thurstonian scores...")
    scores = load_thurstonian_scores(RUN_DIR)

    print("Building task groups...")
    task_groups = build_task_groups(set(task_ids), "topic", TOPICS_JSON)

    # Build BT pairwise data
    print("Building pairwise data...")
    bt_data = PairwiseActivationData.from_measurements(measurements, task_ids, activations)

    # Filter to tasks that are in HOO_GROUPS and have scores
    scored_and_grouped = {
        tid for tid in task_ids
        if tid in task_groups and task_groups[tid] in HOO_GROUPS and tid in scores
    }

    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    # Generate all folds
    folds = list(combinations(sorted(HOO_GROUPS), HOLD_OUT_SIZE))
    print(f"{len(folds)} folds")

    # Load summary files to get fold ordering (they may differ)
    with open(Path(PROBE_DIRS["Ridge raw"]) / "hoo_summary.json") as f:
        raw_summary = json.load(f)

    all_results = []

    for fold_idx, fold_info in enumerate(raw_summary["folds"]):
        held_out = set(fold_info["held_out_groups"])
        held_out_label = ", ".join(sorted(held_out))

        # Split tasks
        eval_task_ids = [tid for tid in scored_and_grouped if task_groups[tid] in held_out]
        eval_indices = np.array([id_to_idx[tid] for tid in eval_task_ids])
        eval_true_scores = np.array([scores[tid] for tid in eval_task_ids])

        # Split BT pairs for held-out accuracy
        _, bt_eval = bt_data.split_by_groups(task_ids, task_groups, held_out)

        fold_results = {
            "fold_idx": fold_idx,
            "held_out": sorted(held_out),
            "n_eval_tasks": len(eval_task_ids),
            "n_eval_pairs": len(bt_eval.pairs),
            "conditions": {},
        }

        for cond_name, probe_dir in PROBE_DIRS.items():
            summary_path = Path(probe_dir) / "hoo_summary.json"
            if not summary_path.exists():
                continue

            with open(summary_path) as f:
                cond_summary = json.load(f)

            # Determine method name and layers for this condition
            if cond_name == "BT raw":
                method_name = "bradley_terry"
                cond_layers = LAYERS
            elif cond_name == "Content baseline":
                method_name = "ridge"
                cond_layers = [0]
            else:
                method_name = "ridge"
                cond_layers = LAYERS

            for layer in cond_layers:
                probe_path = Path(probe_dir) / "probes" / f"probe_hoo_fold{fold_idx}_{method_name}_L{layer:02d}.npy"
                if not probe_path.exists():
                    continue

                weights = np.load(probe_path)
                w, b = weights[:-1], weights[-1]

                # Get the right activations
                if cond_name == "Content baseline":
                    act = st_activations[0]
                else:
                    act = activations[layer]

                # Predict scores for held-out tasks
                pred_scores = act[eval_indices] @ w + b

                # Pearson r
                if len(pred_scores) > 2:
                    r, _ = stats.pearsonr(pred_scores, eval_true_scores)
                else:
                    r = float('nan')

                # Pairwise accuracy on held-out BT pairs
                pair_acc = compute_pairwise_accuracy(pred_scores, eval_indices, bt_eval)

                key = f"{cond_name}_L{layer}"
                fold_results["conditions"][key] = {
                    "pearson_r": float(r),
                    "pairwise_acc": float(pair_acc),
                }

        all_results.append(fold_results)

        if fold_idx % 10 == 0:
            print(f"  Fold {fold_idx}/{len(raw_summary['folds'])}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved unified metrics to {OUTPUT_PATH}")

    # Print summary
    print("\n" + "=" * 90)
    print("UNIFIED COMPARISON (all methods on both metrics, layer 31)")
    print("=" * 90)

    print(f"\n{'Condition':<30} {'Held-out r':>12} {'Held-out acc':>14} {'N folds':>10}")
    print("-" * 70)

    for cond_name in PROBE_DIRS:
        if cond_name == "Content baseline":
            key = f"{cond_name}_L0"
        else:
            key = f"{cond_name}_L31"

        rs = [f["conditions"][key]["pearson_r"] for f in all_results if key in f["conditions"]]
        accs = [f["conditions"][key]["pairwise_acc"] for f in all_results if key in f["conditions"]]

        if rs:
            print(f"{cond_name:<30} {np.mean(rs):>10.4f}±{np.std(rs):.3f}  "
                  f"{np.mean(accs):>10.4f}±{np.std(accs):.3f}  {len(rs):>6}")

    # Full layer breakdown
    print("\n" + "=" * 90)
    print("FULL LAYER BREAKDOWN")
    print("=" * 90)
    print(f"\n{'Condition':<30} {'Layer':>6} {'Held-out r':>12} {'Held-out acc':>14}")
    print("-" * 70)

    for cond_name in PROBE_DIRS:
        if cond_name == "Content baseline":
            layer_list = [0]
        else:
            layer_list = LAYERS
        for layer in layer_list:
            key = f"{cond_name}_L{layer}"
            rs = [f["conditions"][key]["pearson_r"] for f in all_results if key in f["conditions"]]
            accs = [f["conditions"][key]["pairwise_acc"] for f in all_results if key in f["conditions"]]
            if rs:
                print(f"{cond_name:<30} {layer:>6} {np.mean(rs):>10.4f}±{np.std(rs):.3f}  "
                      f"{np.mean(accs):>10.4f}±{np.std(accs):.3f}")


if __name__ == "__main__":
    main()
