"""Analysis 1: Iteration truncation.

For N = 1..9 iterations, refit Thurstonian utilities on the truncated measurements,
compute rank correlation against full-data utilities, train Ridge probe (5-fold CV),
and evaluate held-out pairwise accuracy.
"""

import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.fitting.thurstonian_fitting.thurstonian import PairwiseData, fit_thurstonian
from src.probes.core.activations import load_activations
from src.probes.core.linear_probe import train_and_evaluate
from src.probes.bradley_terry.data import PairwiseActivationData
from src.probes.bradley_terry.training import pairwise_accuracy_from_scores
from src.probes.data_loading import load_pairwise_measurements, load_thurstonian_scores

RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
ACTIVATIONS_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
OUTPUT_PATH = Path("experiments/probe_science/active_learning_calibration/analysis_1_results.json")
LAYER = 31
CV_FOLDS = 5
ALPHA_SWEEP_SIZE = 10

# Iteration boundaries: iter 0 has ~7500 pairs × 5 samples, iters 1-8 have ~2000 × 5
ITER_0_MEASUREMENTS = 7500 * 5  # 37500
SUBSEQUENT_MEASUREMENTS = 2000 * 5  # 10000


def split_measurements_by_iteration(measurements: list, n_iterations: int) -> list[list]:
    """Return cumulative measurement lists for each iteration count (1..n_iterations)."""
    boundaries = [ITER_0_MEASUREMENTS]
    for i in range(1, n_iterations):
        boundaries.append(boundaries[-1] + SUBSEQUENT_MEASUREMENTS)

    cumulative = []
    for b in boundaries:
        cumulative.append(measurements[:min(b, len(measurements))])
    return cumulative


def evaluate_probe_pairwise(
    task_ids: np.ndarray,
    activations: dict[int, np.ndarray],
    scores: dict[str, float],
    all_measurements: list,
    layer: int,
    cv_folds: int,
    alpha_sweep_size: int,
) -> dict:
    """Train Ridge probe with CV and evaluate held-out pairwise accuracy."""
    # Build aligned arrays
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    valid_ids = [tid for tid in task_ids if tid in scores]
    valid_mask = np.array([tid in scores for tid in task_ids])

    acts = activations[layer][valid_mask]
    labels = np.array([scores[tid] for tid in valid_ids])
    valid_task_ids = np.array(valid_ids)

    # Standardize
    scaler = StandardScaler()
    acts_scaled = scaler.fit_transform(acts)

    # Train probe with alpha sweep
    probe, results, sweep = train_and_evaluate(acts_scaled, labels, cv_folds, alpha_sweep_size)

    # Now evaluate held-out pairwise accuracy with task-level 5-fold CV
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_idx, test_idx in kf.split(acts_scaled):
        train_ids_set = set(valid_task_ids[train_idx])
        test_ids_set = set(valid_task_ids[test_idx])

        # Train probe on this fold
        from sklearn.linear_model import Ridge
        fold_probe = Ridge(alpha=results["best_alpha"])
        fold_probe.fit(acts_scaled[train_idx], labels[train_idx])

        # Predict for test tasks
        test_predictions = fold_probe.predict(acts_scaled[test_idx])
        test_id_to_score = dict(zip(valid_task_ids[test_idx], test_predictions))

        # Evaluate on pairs where both tasks are in test set
        # Use ALL measurements for evaluation (not just truncated ones)
        correct = 0
        total = 0
        for m in all_measurements:
            a_id = m.task_a.id
            b_id = m.task_b.id
            if a_id in test_id_to_score and b_id in test_id_to_score:
                pred_a = test_id_to_score[a_id]
                pred_b = test_id_to_score[b_id]
                if pred_a == pred_b:
                    continue
                winner = "a" if m.choice == "a" else "b"
                predicted_winner = "a" if pred_a > pred_b else "b"
                if winner == predicted_winner:
                    correct += 1
                total += 1

        if total > 0:
            fold_accuracies.append(correct / total)

    pairwise_acc = float(np.mean(fold_accuracies)) if fold_accuracies else 0.0
    pairwise_std = float(np.std(fold_accuracies)) if fold_accuracies else 0.0

    return {
        "cv_r2_mean": results["cv_r2_mean"],
        "cv_r2_std": results["cv_r2_std"],
        "cv_pearson_r_mean": results["cv_pearson_r_mean"],
        "best_alpha": results["best_alpha"],
        "pairwise_accuracy_mean": pairwise_acc,
        "pairwise_accuracy_std": pairwise_std,
        "n_test_pairs_per_fold": total,
    }


def main():
    print("Loading data...")
    all_measurements = load_pairwise_measurements(RUN_DIR)
    full_scores = load_thurstonian_scores(RUN_DIR)
    task_ids, activations = load_activations(ACTIVATIONS_PATH, layers=[LAYER])

    # Get tasks for Thurstonian fitting
    # We need Task objects for fitting
    task_objects = []
    seen = set()
    for m in all_measurements:
        for t in [m.task_a, m.task_b]:
            if t.id not in seen:
                task_objects.append(t)
                seen.add(t.id)

    print(f"Loaded {len(all_measurements)} measurements, {len(task_objects)} tasks, {len(task_ids)} activations")

    # Full data utilities (reference)
    full_mu = full_scores

    # Split measurements by iteration
    n_iterations = 9
    cumulative_measurements = split_measurements_by_iteration(all_measurements, n_iterations)

    results = []
    for n_iter, measurements_subset in enumerate(cumulative_measurements, 1):
        n_meas = len(measurements_subset)
        n_unique_pairs = len(set(
            tuple(sorted([m.task_a.id, m.task_b.id])) for m in measurements_subset
        ))

        print(f"\n=== Iteration {n_iter} ({n_meas} measurements, {n_unique_pairs} unique pairs) ===")

        # Refit Thurstonian
        data = PairwiseData.from_comparisons(measurements_subset, task_objects)
        fit = fit_thurstonian(data)
        fitted_scores = {t.id: float(fit.mu[i]) for i, t in enumerate(fit.tasks)}

        # Rank correlation against full-data utilities
        common_ids = sorted(set(fitted_scores.keys()) & set(full_mu.keys()))
        fitted_arr = np.array([fitted_scores[tid] for tid in common_ids])
        full_arr = np.array([full_mu[tid] for tid in common_ids])
        rank_corr = float(spearmanr(fitted_arr, full_arr).correlation)

        print(f"  Rank correlation with full data: {rank_corr:.4f}")
        print(f"  Thurstonian converged: {fit.converged}")

        # Train and evaluate probe
        probe_results = evaluate_probe_pairwise(
            task_ids, activations, fitted_scores, all_measurements,
            LAYER, CV_FOLDS, ALPHA_SWEEP_SIZE,
        )

        comparisons_per_task = n_meas / len(task_objects)

        entry = {
            "n_iterations": n_iter,
            "n_measurements": n_meas,
            "n_unique_pairs": n_unique_pairs,
            "comparisons_per_task": round(comparisons_per_task, 1),
            "rank_correlation_with_full": rank_corr,
            "thurstonian_converged": fit.converged,
            "thurstonian_nll": float(fit.neg_log_likelihood),
            **probe_results,
        }
        results.append(entry)
        print(f"  Probe CV R²: {probe_results['cv_r2_mean']:.3f} ± {probe_results['cv_r2_std']:.3f}")
        print(f"  Pairwise accuracy: {probe_results['pairwise_accuracy_mean']:.3f} ± {probe_results['pairwise_accuracy_std']:.3f}")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
