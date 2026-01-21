from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore", category=linalg.LinAlgWarning)


def get_default_alphas(n_alphas: int = 17) -> np.ndarray:
    return np.logspace(-4, 4, n_alphas)


def train_and_evaluate(
    activations: np.ndarray,
    labels: np.ndarray,
    cv_folds: int,
    alpha_sweep_size: int,
    verbose: bool = False,
) -> tuple[RidgeCV, dict, None]:
    """Train linear probe with RidgeCV for efficient alpha selection."""
    alphas = get_default_alphas(alpha_sweep_size)

    # RidgeCV does efficient LOO or GCV internally for alpha selection
    probe = RidgeCV(alphas=alphas, cv=cv_folds)
    probe.fit(activations, labels)

    # Get CV scores at best alpha for reporting
    cv_r2_scores = cross_val_score(probe, activations, labels, cv=cv_folds, scoring="r2")
    cv_mse_scores = -cross_val_score(probe, activations, labels, cv=cv_folds, scoring="neg_mean_squared_error")

    # Train R2
    y_pred = probe.predict(activations)
    train_r2 = float(np.corrcoef(labels, y_pred)[0, 1] ** 2)

    cv_r2_mean = float(cv_r2_scores.mean())
    cv_r2_std = float(cv_r2_scores.std())

    results = {
        "best_alpha": float(probe.alpha_),
        "train_r2": train_r2,
        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": cv_r2_std,
        "cv_mse_mean": float(cv_mse_scores.mean()),
        "cv_mse_std": float(cv_mse_scores.std()),
        "train_test_gap": train_r2 - cv_r2_mean,
        "cv_stability": 1.0 - (cv_r2_std / (abs(cv_r2_mean) + 1e-10)),
    }

    return probe, results, None
