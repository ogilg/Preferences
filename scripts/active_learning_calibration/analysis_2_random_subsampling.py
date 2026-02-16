"""Analysis 2: Random subsampling of comparisons.

For fractions f = 0.1..1.0 of unique pairs, subsample pairs (keeping all
measurements for selected pairs), refit Thurstonian, train probe, evaluate.
5 seeds per fraction.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.fitting.thurstonian_fitting.thurstonian import PairwiseData, fit_thurstonian
from src.probes.core.activations import load_activations
from src.probes.core.linear_probe import train_and_evaluate
from src.task_data import Task, OriginDataset

from scripts.active_learning_calibration.fast_loading import (
    load_measurements_as_arrays, load_full_thurstonian_scores, get_pair_indices,
)

ACTIVATIONS_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
OUTPUT_PATH = Path("experiments/probe_science/active_learning_calibration/analysis_2_results.json")
LAYER = 31
CV_FOLDS = 5
ALPHA_SWEEP_SIZE = 10
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_SEEDS = 5


def build_wins_from_pair_mask(task_a_idx, task_b_idx, winner_idx, n_tasks, pair_mask):
    """Build wins matrix using only measurements that belong to selected pairs."""
    a = task_a_idx[pair_mask]
    b = task_b_idx[pair_mask]
    w = winner_idx[pair_mask]

    # Vectorized: winner gets wins[winner, loser] += 1
    loser = np.where(w == a, b, a)
    wins = np.zeros((n_tasks, n_tasks), dtype=np.int32)
    np.add.at(wins, (w, loser), 1)
    return wins


def compute_pairwise_accuracy_numpy(
    predicted_scores, task_a_idx, task_b_idx, winner_idx, test_mask
):
    """Compute pairwise accuracy from predicted scores on test measurements.

    test_mask: boolean mask over measurements where both tasks are in the test set.
    """
    a = task_a_idx[test_mask]
    b = task_b_idx[test_mask]
    w = winner_idx[test_mask]

    pred_a = predicted_scores[a]
    pred_b = predicted_scores[b]

    # Predict winner as the one with higher score
    predicted_winner = np.where(pred_a > pred_b, a, b)

    # Handle ties — exclude them
    tie_mask = pred_a == pred_b
    valid = ~tie_mask

    if valid.sum() == 0:
        return 0.0

    correct = (predicted_winner[valid] == w[valid]).sum()
    return float(correct / valid.sum())


def main():
    print("Loading data...")
    task_a_idx, task_b_idx, winner_idx, task_id_list = load_measurements_as_arrays()
    full_scores = load_full_thurstonian_scores()
    act_task_ids, activations = load_activations(ACTIVATIONS_PATH, layers=[LAYER])

    n_tasks = len(task_id_list)
    n_measurements = len(task_a_idx)
    print(f"Loaded {n_measurements} measurements, {n_tasks} tasks")

    # Create dummy Task objects for PairwiseData (minimal, just need ids)
    task_objects = [Task(id=tid, prompt="", origin=OriginDataset.WILDCHAT, metadata={}) for tid in task_id_list]

    # Get unique pairs
    unique_pairs = get_pair_indices(task_a_idx, task_b_idx)
    n_unique_pairs = len(unique_pairs)
    print(f"Unique pairs: {n_unique_pairs}")

    # Map each measurement to its pair index for fast subsetting
    # Create a pair lookup: (min, max) -> pair_index
    pair_key = np.minimum(task_a_idx, task_b_idx) * n_tasks + np.maximum(task_a_idx, task_b_idx)
    unique_pair_key = unique_pairs[:, 0] * n_tasks + unique_pairs[:, 1]
    pair_key_to_idx = {int(k): i for i, k in enumerate(unique_pair_key)}
    measurement_pair_idx = np.array([pair_key_to_idx[int(k)] for k in pair_key])

    # Align activations with task_id_list
    act_id_to_idx = {tid: i for i, tid in enumerate(act_task_ids)}
    # Find tasks that have both activations and are in our measurement set
    common_ids = [tid for tid in task_id_list if tid in act_id_to_idx and tid in full_scores]
    meas_idx_for_common = {tid: i for i, tid in enumerate(task_id_list)}
    act_indices = np.array([act_id_to_idx[tid] for tid in common_ids])
    acts = activations[LAYER][act_indices]
    common_task_ids = np.array(common_ids)

    print(f"Tasks with activations and scores: {len(common_ids)}")

    # Pre-compute: for each measurement, which common_ids index does each task map to?
    common_id_to_idx = {tid: i for i, tid in enumerate(common_ids)}

    # Full-data scores aligned to common_ids
    full_labels = np.array([full_scores[tid] for tid in common_ids])

    results = []
    for fraction in FRACTIONS:
        print(f"\n=== Fraction {fraction:.1f} ===")

        n_seeds = 1 if fraction >= 1.0 else N_SEEDS
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)

            if fraction >= 1.0:
                pair_mask = np.ones(n_measurements, dtype=bool)
                n_selected_pairs = n_unique_pairs
            else:
                n_select = max(1, int(n_unique_pairs * fraction))
                selected_pair_indices = rng.choice(n_unique_pairs, size=n_select, replace=False)
                is_selected = np.zeros(n_unique_pairs, dtype=bool)
                is_selected[selected_pair_indices] = True
                pair_mask = is_selected[measurement_pair_idx]
                n_selected_pairs = n_select

            n_meas = int(pair_mask.sum())
            comparisons_per_task = round(n_meas / n_tasks, 1)

            # Build wins matrix and fit Thurstonian
            wins = build_wins_from_pair_mask(task_a_idx, task_b_idx, winner_idx, n_tasks, pair_mask)
            pairwise_data = PairwiseData(tasks=task_objects, wins=wins)
            fit = fit_thurstonian(pairwise_data)
            fitted_mu = {task_objects[i].id: float(fit.mu[i]) for i in range(n_tasks)}

            # Rank correlation
            fitted_arr = np.array([fitted_mu[tid] for tid in common_ids])
            rank_corr = float(spearmanr(fitted_arr, full_labels).correlation)

            # Probe training and evaluation with CV pairwise accuracy
            labels = np.array([fitted_mu[tid] for tid in common_ids])
            scaler = StandardScaler()
            acts_scaled = scaler.fit_transform(acts)

            probe, probe_results, sweep = train_and_evaluate(acts_scaled, labels, CV_FOLDS, ALPHA_SWEEP_SIZE)

            # Task-level CV for pairwise accuracy
            kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
            fold_accuracies = []

            # Pre-compute mapping: measurement task indices -> boolean "is test task"
            # Using numpy boolean arrays for fast masking
            for train_idx, test_idx in kf.split(acts_scaled):
                fold_probe = Ridge(alpha=probe_results["best_alpha"])
                fold_probe.fit(acts_scaled[train_idx], labels[train_idx])

                # Predict scores for test tasks, indexed into measurement space
                all_preds = np.full(n_tasks, np.nan)
                test_preds = fold_probe.predict(acts_scaled[test_idx])
                test_meas_indices = np.array([meas_idx_for_common[common_ids[i]] for i in test_idx])
                all_preds[test_meas_indices] = test_preds

                # Build test mask: both tasks must be test tasks
                is_test_task = np.zeros(n_tasks, dtype=bool)
                is_test_task[test_meas_indices] = True
                test_meas_mask = is_test_task[task_a_idx] & is_test_task[task_b_idx]

                acc = compute_pairwise_accuracy_numpy(
                    all_preds, task_a_idx, task_b_idx, winner_idx, test_meas_mask
                )
                fold_accuracies.append(acc)

            pairwise_acc = float(np.mean(fold_accuracies))
            pairwise_std = float(np.std(fold_accuracies))

            entry = {
                "fraction": fraction,
                "seed": seed,
                "n_measurements": n_meas,
                "n_unique_pairs": n_selected_pairs,
                "comparisons_per_task": comparisons_per_task,
                "rank_correlation_with_full": rank_corr,
                "cv_r2_mean": probe_results["cv_r2_mean"],
                "cv_r2_std": probe_results["cv_r2_std"],
                "cv_pearson_r_mean": probe_results["cv_pearson_r_mean"],
                "best_alpha": probe_results["best_alpha"],
                "pairwise_accuracy_mean": pairwise_acc,
                "pairwise_accuracy_std": pairwise_std,
            }
            results.append(entry)
            print(f"  Seed {seed}: {n_selected_pairs} pairs, rank_corr={rank_corr:.3f}, R²={probe_results['cv_r2_mean']:.3f}, pw_acc={pairwise_acc:.3f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
