from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


def train_and_evaluate(
    activations: np.ndarray,
    labels: np.ndarray,
    cv_folds: int = 5,
    alpha: float = 1.0,
) -> tuple[Ridge, dict]:
    """Train linear probe with cross-validation on full dataset."""
    probe = Ridge(alpha=alpha)

    cv_scores = cross_val_score(probe, activations, labels, cv=cv_folds, scoring="r2")

    probe.fit(activations, labels)
    y_pred = probe.predict(activations)

    results = {
        "r2": float(np.corrcoef(labels, y_pred)[0, 1] ** 2),
        "mse": float(np.mean((labels - y_pred) ** 2)),
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
    }

    return probe, results
