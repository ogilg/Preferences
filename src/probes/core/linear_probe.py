from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=linalg.LinAlgWarning)


def get_default_alphas(n_alphas: int = 5) -> np.ndarray:
    return np.logspace(0, 6, n_alphas)


def _alpha_sweep(
    activations: np.ndarray,
    labels: np.ndarray,
    alphas: np.ndarray,
    cv_folds: int,
    standardize: bool = False,
) -> list[dict]:
    """Evaluate each alpha with CV, returning per-alpha train and val R2."""
    sweep = []
    for alpha in alphas:
        if standardize:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        else:
            model = Ridge(alpha=alpha)
        cv_r2 = cross_val_score(model, activations, labels, cv=cv_folds, scoring="r2")
        model.fit(activations, labels)
        train_r2 = float(np.corrcoef(labels, model.predict(activations))[0, 1] ** 2)
        sweep.append({
            "alpha": float(alpha),
            "train_r2": train_r2,
            "val_r2_mean": float(cv_r2.mean()),
            "val_r2_std": float(cv_r2.std()),
        })
    return sweep


def train_and_evaluate(
    activations: np.ndarray,
    labels: np.ndarray,
    cv_folds: int,
    alpha_sweep_size: int,
    standardize: bool = False,
    verbose: bool = False,
) -> tuple[RidgeCV, dict, list[dict]]:
    """Train linear probe with RidgeCV for efficient alpha selection.

    Returns (probe, results_dict, alpha_sweep_results).
    """
    alphas = get_default_alphas(alpha_sweep_size)

    # Full alpha sweep with train/val R2
    sweep = _alpha_sweep(activations, labels, alphas, cv_folds, standardize=standardize)

    # RidgeCV for final probe (standardize externally if needed)
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(activations)
    else:
        X_scaled = activations

    probe = RidgeCV(alphas=alphas, cv=cv_folds)
    probe.fit(X_scaled, labels)

    # Get CV scores at best alpha for reporting
    cv_r2_scores = cross_val_score(probe, X_scaled, labels, cv=cv_folds, scoring="r2")
    cv_mse_scores = -cross_val_score(probe, X_scaled, labels, cv=cv_folds, scoring="neg_mean_squared_error")

    # Train R2
    y_pred = probe.predict(X_scaled)
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
        "standardized": standardize,
    }

    return probe, results, sweep
