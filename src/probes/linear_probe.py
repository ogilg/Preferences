from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate

DEFAULT_ALPHAS = np.logspace(-4, 4, 17)


@dataclass
class AlphaResult:
    alpha: float
    train_r2: float
    cv_r2_mean: float
    cv_r2_std: float
    cv_mse_mean: float | None = None
    cv_mse_std: float | None = None


def train_and_evaluate(
    activations: np.ndarray,
    labels: np.ndarray,
    cv_folds: int = 5,
    alphas: np.ndarray = DEFAULT_ALPHAS,
) -> tuple[Ridge, dict, list[AlphaResult]]:
    """Train linear probe with alpha sweep and cross-validation."""
    alpha_results = []
    best_cv_r2 = -np.inf
    best_alpha = None
    best_probe = None

    for alpha in alphas:
        probe = Ridge(alpha=alpha)
        cv_results = cross_validate(
            probe, activations, labels, cv=cv_folds,
            scoring=["r2", "neg_mean_squared_error"],
            return_estimator=True,
        )
        cv_r2_scores = cv_results["test_r2"]
        cv_mse_scores = -cv_results["test_neg_mean_squared_error"]

        probe.fit(activations, labels)
        y_pred = probe.predict(activations)
        train_r2 = float(np.corrcoef(labels, y_pred)[0, 1] ** 2)

        alpha_results.append(AlphaResult(
            alpha=float(alpha),
            train_r2=train_r2,
            cv_r2_mean=float(cv_r2_scores.mean()),
            cv_r2_std=float(cv_r2_scores.std()),
            cv_mse_mean=float(cv_mse_scores.mean()),
            cv_mse_std=float(cv_mse_scores.std()),
        ))

        if cv_r2_scores.mean() > best_cv_r2:
            best_cv_r2 = cv_r2_scores.mean()
            best_alpha = alpha
            best_probe = probe

    best_result = next(r for r in alpha_results if r.alpha == best_alpha)
    results = {
        "best_alpha": float(best_alpha),
        "train_r2": best_result.train_r2,
        "cv_r2_mean": best_result.cv_r2_mean,
        "cv_r2_std": best_result.cv_r2_std,
        "cv_mse_mean": best_result.cv_mse_mean,
        "cv_mse_std": best_result.cv_mse_std,
    }

    return best_probe, results, alpha_results
