"""Fair BT vs Ridge comparison with task-level k-fold splits.

Both methods evaluated on identical held-out test pairs per fold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.probes.bradley_terry.data import PairwiseActivationData
from src.probes.bradley_terry.training import (
    train_bt,
    train_bt_fixed_lambda,
    weighted_accuracy,
    pairwise_accuracy_from_scores,
)
from src.probes.core.activations import load_activations
from src.probes.core.linear_probe import train_and_evaluate, train_at_alpha
from src.probes.data_loading import load_thurstonian_scores, load_pairwise_measurements


# --- Config ---
PREFERENCES_ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")
ACTIVATIONS_PATH = PREFERENCES_ROOT / "activations/gemma_3_27b/activations_prompt_last.npz"
RUN_DIR = (
    PREFERENCES_ROOT
    / "results/experiments/gemma3_3k_run2/pre_task_active_learning"
    / "completion_preference_gemma-3-27b_completion_canonical_seed0"
)
LAYERS = [31, 43, 55]
K_FOLDS = 5
SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "experiments/bt_fair_comparison"


def assign_folds(task_ids_scored: list[str], k: int, seed: int) -> dict[str, int]:
    """Assign each task to a fold. Shuffle then fold = index % k."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(task_ids_scored))
    rng.shuffle(indices)
    return {task_ids_scored[i]: int(idx % k) for idx, i in enumerate(indices)}


def split_bt_data_by_fold(
    bt_data: PairwiseActivationData,
    task_ids: np.ndarray,
    task_fold_map: dict[str, int],
    test_fold: int,
) -> tuple[PairwiseActivationData, PairwiseActivationData]:
    """Split BT data into train/test based on fold assignment.

    Train: pairs where both tasks are NOT in the test fold.
    Test: pairs where both tasks ARE in the test fold.
    Cross-fold pairs are dropped.
    """
    idx_to_tid = {i: tid for i, tid in enumerate(task_ids)}

    train_mask = np.zeros(len(bt_data.pairs), dtype=bool)
    test_mask = np.zeros(len(bt_data.pairs), dtype=bool)

    for k, (i, j) in enumerate(bt_data.pairs):
        tid_i = idx_to_tid[i]
        tid_j = idx_to_tid[j]
        fold_i = task_fold_map[tid_i]
        fold_j = task_fold_map[tid_j]
        if fold_i != test_fold and fold_j != test_fold:
            train_mask[k] = True
        elif fold_i == test_fold and fold_j == test_fold:
            test_mask[k] = True

    def _subset(mask: np.ndarray) -> PairwiseActivationData:
        return PairwiseActivationData(
            activations=bt_data.activations,
            pairs=bt_data.pairs[mask],
            wins_i=bt_data.wins_i[mask],
            total=bt_data.total[mask],
            n_measurements=int(np.sum(bt_data.total[mask])),
        )

    return _subset(train_mask), _subset(test_mask)


def build_ridge_xy_for_tasks(
    task_ids: np.ndarray,
    scores: dict[str, float],
    task_set: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (indices, y) arrays for Ridge, filtered to task_set."""
    indices = []
    y_vals = []
    for i, tid in enumerate(task_ids):
        if tid in task_set and tid in scores:
            indices.append(i)
            y_vals.append(scores[tid])
    return np.array(indices, dtype=int), np.array(y_vals)


def evaluate_ridge_pairwise(
    task_scores: np.ndarray,
    test_bt_data: PairwiseActivationData,
) -> float:
    """Evaluate pre-computed Ridge scores as pairwise accuracy on BT test pairs.

    Uses the same weighted accuracy metric as BT for fair comparison.
    """
    idx_i = test_bt_data.pairs[:, 0]
    idx_j = test_bt_data.pairs[:, 1]
    wins_i = test_bt_data.wins_i
    wins_j = test_bt_data.total - test_bt_data.wins_i

    logits = task_scores[idx_i] - task_scores[idx_j]
    correct = np.where(logits > 0, wins_i, wins_j)
    total_weight = float(np.sum(wins_i + wins_j))
    return float(np.sum(correct) / total_weight)


def run_experiment():
    print("=" * 60)
    print("Fair BT vs Ridge Comparison")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    scores = load_thurstonian_scores(RUN_DIR)
    measurements = load_pairwise_measurements(RUN_DIR)
    scored_task_ids = sorted(scores.keys())
    print(f"  {len(scored_task_ids)} tasks with Thurstonian scores")
    print(f"  {len(measurements)} pairwise measurements")

    # Load activations filtered to scored tasks
    task_ids, activations = load_activations(
        ACTIVATIONS_PATH,
        task_id_filter=set(scored_task_ids),
        layers=LAYERS,
    )
    print(f"  {len(task_ids)} tasks with activations (after filtering)")

    # Build BT data
    bt_data = PairwiseActivationData.from_measurements(measurements, task_ids, activations)
    print(f"  {len(bt_data.pairs)} unique pairs, {bt_data.n_measurements} measurements")

    # Assign folds
    task_fold_map = assign_folds(list(task_ids), K_FOLDS, SEED)
    fold_sizes = {f: sum(1 for v in task_fold_map.values() if v == f) for f in range(K_FOLDS)}
    print(f"\nFold sizes: {fold_sizes}")

    # Storage for results
    results = {
        "config": {
            "k_folds": K_FOLDS,
            "seed": SEED,
            "layers": LAYERS,
            "n_tasks": len(task_ids),
            "n_pairs": len(bt_data.pairs),
            "n_measurements": bt_data.n_measurements,
        },
        "folds": [],
    }

    # Per-layer hyperparameters (selected on fold 0)
    best_alphas: dict[int, float] = {}
    best_lambdas: dict[int, float] = {}

    for fold_idx in range(K_FOLDS):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}")
        print(f"{'='*60}")

        # Determine train/test task sets
        test_tasks = {tid for tid, f in task_fold_map.items() if f == fold_idx}
        train_tasks = {tid for tid, f in task_fold_map.items() if f != fold_idx}
        print(f"  Train tasks: {len(train_tasks)}, Test tasks: {len(test_tasks)}")

        # Split BT data
        train_bt_data, test_bt_data = split_bt_data_by_fold(
            bt_data, task_ids, task_fold_map, fold_idx,
        )
        n_test_pairs = len(test_bt_data.pairs)
        n_test_measurements = test_bt_data.n_measurements
        print(f"  Train pairs: {len(train_bt_data.pairs)}, Test pairs: {n_test_pairs}")
        print(f"  Test measurements: {n_test_measurements}")

        if n_test_pairs < 10:
            print(f"  WARNING: Only {n_test_pairs} test pairs, skipping fold")
            continue

        fold_result = {
            "fold": fold_idx,
            "n_train_tasks": len(train_tasks),
            "n_test_tasks": len(test_tasks),
            "n_train_pairs": len(train_bt_data.pairs),
            "n_test_pairs": n_test_pairs,
            "n_test_measurements": n_test_measurements,
            "layers": {},
        }

        for layer in LAYERS:
            print(f"\n  --- Layer {layer} ---")
            layer_acts = activations[layer]

            # === BT ===
            if fold_idx == 0:
                print("  BT: Lambda sweep on fold 0...")
                bt_result = train_bt(train_bt_data, layer)
                best_lambdas[layer] = bt_result.best_l2_lambda
                print(f"  BT: Best lambda = {best_lambdas[layer]:.4g}")
            else:
                print(f"  BT: Training at fixed lambda = {best_lambdas[layer]:.4g}")
                bt_result = train_bt_fixed_lambda(train_bt_data, layer, best_lambdas[layer])

            # Evaluate BT on test pairs
            w_bt = bt_result.weights[:-1]  # strip trailing zero
            bt_test_acc = weighted_accuracy(
                w_bt, layer_acts,
                test_bt_data.pairs[:, 0], test_bt_data.pairs[:, 1],
                test_bt_data.wins_i, test_bt_data.total - test_bt_data.wins_i,
                float(np.sum(test_bt_data.total)),
            )
            print(f"  BT test pairwise accuracy: {bt_test_acc:.4f}")

            # === Ridge ===
            train_indices, train_y = build_ridge_xy_for_tasks(task_ids, scores, train_tasks)
            print(f"  Ridge: {len(train_indices)} training samples")

            # Fit scaler on train data, then score ALL tasks in the scaled space
            scaler = StandardScaler()
            scaler.fit(layer_acts[train_indices])
            all_acts_scaled = scaler.transform(layer_acts)

            if fold_idx == 0:
                print("  Ridge: Alpha sweep on fold 0...")
                # Pass pre-scaled data, don't double-standardize
                probe, eval_results, alpha_sweep = train_and_evaluate(
                    all_acts_scaled[train_indices], train_y,
                    cv_folds=5, alpha_sweep_size=10, standardize=False,
                )
                best_alphas[layer] = eval_results["best_alpha"]
                ridge_cv_r2 = eval_results["cv_r2_mean"]
                print(f"  Ridge: Best alpha = {best_alphas[layer]:.4g}, CV R² = {ridge_cv_r2:.4f}")
            else:
                print(f"  Ridge: Training at fixed alpha = {best_alphas[layer]:.4g}")
                probe, eval_results = train_at_alpha(
                    all_acts_scaled[train_indices], train_y,
                    alpha=best_alphas[layer], cv_folds=5, standardize=False,
                )

            # Score all tasks in the properly scaled space
            task_scores_ridge = all_acts_scaled @ probe.coef_ + probe.intercept_

            # Evaluate Ridge on the SAME test pairs as BT
            ridge_test_acc = evaluate_ridge_pairwise(
                task_scores_ridge, test_bt_data,
            )
            print(f"  Ridge test pairwise accuracy: {ridge_test_acc:.4f}")

            # Also compute Ridge pairwise accuracy using Thurstonian scores directly
            # (as a reference: how well do the Thurstonian utilities themselves predict?)
            thurst_test_acc = pairwise_accuracy_from_scores(
                scores, test_bt_data, task_ids,
            )
            print(f"  Thurstonian μ test pairwise accuracy: {thurst_test_acc:.4f}")

            fold_result["layers"][layer] = {
                "bt_test_acc": bt_test_acc,
                "bt_train_acc": bt_result.train_accuracy,
                "bt_lambda": bt_result.best_l2_lambda,
                "ridge_test_acc": ridge_test_acc,
                "ridge_cv_r2": eval_results["cv_r2_mean"],
                "ridge_alpha": eval_results["best_alpha"],
                "thurstonian_test_acc": thurst_test_acc,
            }

        results["folds"].append(fold_result)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (Layer 31)")
    print("=" * 60)
    print(f"{'Fold':>4} | {'BT acc':>8} | {'Ridge acc':>9} | {'Thurst acc':>10} | {'Test pairs':>10}")
    print("-" * 60)
    for fr in results["folds"]:
        l31 = fr["layers"][31]
        print(f"{fr['fold']:>4} | {l31['bt_test_acc']:>8.4f} | {l31['ridge_test_acc']:>9.4f} | {l31['thurstonian_test_acc']:>10.4f} | {fr['n_test_pairs']:>10}")

    # Compute means
    bt_accs = [fr["layers"][31]["bt_test_acc"] for fr in results["folds"]]
    ridge_accs = [fr["layers"][31]["ridge_test_acc"] for fr in results["folds"]]
    thurst_accs = [fr["layers"][31]["thurstonian_test_acc"] for fr in results["folds"]]

    print("-" * 60)
    print(f"{'Mean':>4} | {np.mean(bt_accs):>8.4f} | {np.mean(ridge_accs):>9.4f} | {np.mean(thurst_accs):>10.4f}")
    print(f"{'Std':>4} | {np.std(bt_accs):>8.4f} | {np.std(ridge_accs):>9.4f} | {np.std(thurst_accs):>10.4f}")


if __name__ == "__main__":
    run_experiment()
