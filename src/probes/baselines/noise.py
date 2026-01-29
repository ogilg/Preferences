"""Noise baselines for probe benchmarking."""

from __future__ import annotations

import numpy as np

from src.probes.linear_probe import train_and_evaluate

from .types import BaselineResult, BaselineType

BASELINE_ALPHAS = np.array([10.0, 100.0, 1000.0, 10000.0])


def run_shuffled_labels_baseline(
    X: np.ndarray,
    y: np.ndarray,
    template: str,
    layer: int,
    cv_folds: int,
    seed: int,
) -> BaselineResult:
    """Train probe on shuffled labels."""
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y)
    _, result, _ = train_and_evaluate(X, y_shuffled, cv_folds=cv_folds, alphas=BASELINE_ALPHAS)
    return BaselineResult(
        baseline_type=BaselineType.SHUFFLED_LABELS,
        template=template,
        layer=layer,
        cv_r2_mean=result["cv_r2_mean"],
        cv_r2_std=result["cv_r2_std"],
        cv_mse_mean=result["cv_mse_mean"],
        cv_mse_std=result["cv_mse_std"],
        best_alpha=result["best_alpha"],
        n_samples=len(y),
        seed=seed,
    )


def run_random_activations_baseline(
    X: np.ndarray,
    y: np.ndarray,
    template: str,
    layer: int,
    cv_folds: int,
    seed: int,
) -> BaselineResult:
    """Train probe on random activations with same mean/std."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    rng = np.random.default_rng(seed)
    X_noise = rng.normal(loc=mean, scale=std, size=X.shape)
    _, result, _ = train_and_evaluate(X_noise, y, cv_folds=cv_folds, alphas=BASELINE_ALPHAS)
    return BaselineResult(
        baseline_type=BaselineType.RANDOM_ACTIVATIONS,
        template=template,
        layer=layer,
        cv_r2_mean=result["cv_r2_mean"],
        cv_r2_std=result["cv_r2_std"],
        cv_mse_mean=result["cv_mse_mean"],
        cv_mse_std=result["cv_mse_std"],
        best_alpha=result["best_alpha"],
        n_samples=len(y),
        seed=seed,
    )
