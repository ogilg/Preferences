"""Noise baselines for probe benchmarking."""

from __future__ import annotations

import numpy as np

from src.probes.core.linear_probe import train_and_evaluate

from .types import BaselineResult, BaselineType


def run_shuffled_labels_baseline(
    X: np.ndarray,
    y: np.ndarray,
    layer: int,
    cv_folds: int,
    seed: int,
    alpha_sweep_size: int = 10,
) -> BaselineResult:
    """Train probe on shuffled labels."""
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y)
    _, result, _ = train_and_evaluate(
        X, y_shuffled, cv_folds=cv_folds,
        alpha_sweep_size=alpha_sweep_size,
    )
    return BaselineResult.from_cv_result(
        result, BaselineType.SHUFFLED_LABELS, layer, len(y), seed,
    )


def run_random_activations_baseline(
    X: np.ndarray,
    y: np.ndarray,
    layer: int,
    cv_folds: int,
    seed: int,
    alpha_sweep_size: int = 10,
) -> BaselineResult:
    """Train probe on random activations with same mean/std."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    rng = np.random.default_rng(seed)
    X_noise = rng.normal(loc=mean, scale=std, size=X.shape)
    _, result, _ = train_and_evaluate(
        X_noise, y, cv_folds=cv_folds,
        alpha_sweep_size=alpha_sweep_size,
    )
    return BaselineResult.from_cv_result(
        result, BaselineType.RANDOM_ACTIVATIONS, layer, len(y), seed,
    )
