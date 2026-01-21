"""Probe training utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.probes.linear_probe import train_and_evaluate


def train_for_scores(
    task_ids: np.ndarray,
    activations: dict[int, np.ndarray],
    scores: dict[str, float] | list[tuple[str, float]],
    cv_folds: int,
    alpha_sweep_size: int,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    """Train probes for all layers, returning (results, {layer: weights}).

    Args:
        task_ids: Array of task IDs corresponding to activation indices
        activations: Dict mapping layer number to activation matrix
        scores: Either dict {task_id: score} or list of (task_id, score) tuples.
                List format allows duplicate task_ids (multiple measurements per task).
        cv_folds: Number of cross-validation folds

    Returns:
        Tuple of (results list, {layer: weights dict})
    """
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    valid_indices = []
    valid_scores = []

    if isinstance(scores, dict):
        # Old format: dict with unique task_ids
        for task_id, score in scores.items():
            if task_id in id_to_idx:
                valid_indices.append(id_to_idx[task_id])
                valid_scores.append(score)
    else:
        # New format: list of (task_id, score) tuples (may have duplicates)
        for task_id, score in scores:
            if task_id in id_to_idx:
                valid_indices.append(id_to_idx[task_id])
                valid_scores.append(score)

    if len(valid_indices) < cv_folds * 2:
        return [], {}

    indices = np.array(valid_indices)
    y = np.array(valid_scores)

    # Check for bad target values
    if not np.isfinite(y).all():
        bad_count = (~np.isfinite(y)).sum()
        raise ValueError(f"Found {bad_count} non-finite values in scores")

    results = []
    probes = {}
    for layer in sorted(activations.keys()):
        X = activations[layer][indices]

        # Check for bad activation values
        if not np.isfinite(X).all():
            bad_count = (~np.isfinite(X)).sum()
            raise ValueError(f"Layer {layer}: found {bad_count} non-finite values in activations")

        probe, eval_results, _ = train_and_evaluate(X, y, cv_folds=cv_folds, alpha_sweep_size=alpha_sweep_size)

        results.append({
            "layer": layer,
            "cv_r2_mean": eval_results["cv_r2_mean"],
            "cv_r2_std": eval_results["cv_r2_std"],
            "cv_mse_mean": eval_results["cv_mse_mean"],
            "cv_mse_std": eval_results["cv_mse_std"],
            "best_alpha": eval_results["best_alpha"],
            "n_samples": len(y),
            "train_test_gap": eval_results["train_test_gap"],
            "cv_stability": eval_results["cv_stability"],
        })

        # Store weights: [coef..., intercept]
        probes[layer] = np.append(probe.coef_, probe.intercept_)

    return results, probes
