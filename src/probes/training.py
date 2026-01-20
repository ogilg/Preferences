"""Probe training utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.probes.linear_probe import train_and_evaluate


def train_for_scores(
    task_ids: np.ndarray,
    activations: dict[int, np.ndarray],
    scores: dict[str, float],
    cv_folds: int,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    """Train probes for all layers, returning (results, {layer: weights})."""
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    valid_indices = []
    valid_scores = []
    for task_id, score in scores.items():
        if task_id in id_to_idx:
            valid_indices.append(id_to_idx[task_id])
            valid_scores.append(score)

    if len(valid_indices) < cv_folds * 2:
        return [], {}

    indices = np.array(valid_indices)
    y = np.array(valid_scores)

    results = []
    probes = {}
    for layer in sorted(activations.keys()):
        X = activations[layer][indices]
        probe, eval_results, _ = train_and_evaluate(X, y, cv_folds=cv_folds)

        results.append({
            "layer": layer,
            "cv_r2_mean": eval_results["cv_r2_mean"],
            "cv_r2_std": eval_results["cv_r2_std"],
            "cv_mse_mean": eval_results["cv_mse_mean"],
            "cv_mse_std": eval_results["cv_mse_std"],
            "best_alpha": eval_results["best_alpha"],
            "n_samples": len(y),
        })

        # Store weights: [coef..., intercept]
        probes[layer] = np.append(probe.coef_, probe.intercept_)

    return results, probes
