from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

warnings.filterwarnings("ignore", category=linalg.LinAlgWarning)


def _pearson_scorer(y_true, y_pred):
    if len(y_true) < 3:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return r


def get_default_alphas(n_alphas: int = 5) -> np.ndarray:
    return np.logspace(0, 6, n_alphas)


def _alpha_sweep(
    activations: np.ndarray,
    labels: np.ndarray,
    alphas: np.ndarray,
    cv_folds: int,
) -> list[dict]:
    """Evaluate each alpha with CV, returning per-alpha train/val R2 and MSE."""
    scoring = {
        "r2": "r2",
        "neg_mse": "neg_mean_squared_error",
        "pearson_r": make_scorer(_pearson_scorer),
    }
    sweep = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        cv = cross_validate(model, activations, labels, cv=cv_folds, scoring=scoring)
        cv_r2 = cv["test_r2"]
        cv_mse = -cv["test_neg_mse"]
        cv_pearson = cv["test_pearson_r"]
        model.fit(activations, labels)
        train_r2 = float(np.corrcoef(labels, model.predict(activations))[0, 1] ** 2)
        sweep.append({
            "alpha": float(alpha),
            "train_r2": train_r2,
            "val_r2_mean": float(cv_r2.mean()),
            "val_r2_std": float(cv_r2.std()),
            "val_mse_mean": float(cv_mse.mean()),
            "val_mse_std": float(cv_mse.std()),
            "val_pearson_r_mean": float(cv_pearson.mean()),
            "val_pearson_r_std": float(cv_pearson.std()),
        })
    return sweep


def train_and_evaluate(
    activations: np.ndarray,
    labels: np.ndarray,
    cv_folds: int,
    alpha_sweep_size: int = 5,
    alphas: np.ndarray | None = None,
) -> tuple[Ridge, dict, list[dict]]:
    """Train linear probe: sweep alphas, pick best, fit final model.

    Returns (probe, results_dict, alpha_sweep_results).
    """
    if alphas is None:
        alphas = get_default_alphas(alpha_sweep_size)

    sweep = _alpha_sweep(activations, labels, alphas, cv_folds)

    # Pick best alpha from sweep
    best_entry = max(sweep, key=lambda s: s["val_r2_mean"])

    # Fit final probe at best alpha
    probe = Ridge(alpha=best_entry["alpha"])
    probe.fit(activations, labels)

    results = {
        "best_alpha": best_entry["alpha"],
        "train_r2": best_entry["train_r2"],
        "cv_r2_mean": best_entry["val_r2_mean"],
        "cv_r2_std": best_entry["val_r2_std"],
        "cv_mse_mean": best_entry["val_mse_mean"],
        "cv_mse_std": best_entry["val_mse_std"],
        "cv_pearson_r_mean": best_entry["val_pearson_r_mean"],
        "cv_pearson_r_std": best_entry["val_pearson_r_std"],
        "train_test_gap": best_entry["train_r2"] - best_entry["val_r2_mean"],
        "cv_stability": 1.0 - (best_entry["val_r2_std"] / (abs(best_entry["val_r2_mean"]) + 1e-10)),
    }

    return probe, results, sweep


def train_at_alpha(
    activations: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    cv_folds: int,
) -> tuple[Ridge, dict]:
    """Train probe at a fixed alpha, evaluate with CV. No sweep."""
    probe = Ridge(alpha=alpha)

    scoring = {
        "r2": "r2",
        "neg_mse": "neg_mean_squared_error",
        "pearson_r": make_scorer(_pearson_scorer),
    }
    cv = cross_validate(probe, activations, labels, cv=cv_folds, scoring=scoring)

    probe.fit(activations, labels)

    results = {
        "best_alpha": alpha,
        "train_r2": float(np.corrcoef(labels, probe.predict(activations))[0, 1] ** 2),
        "cv_r2_mean": float(cv["test_r2"].mean()),
        "cv_r2_std": float(cv["test_r2"].std()),
        "cv_mse_mean": float((-cv["test_neg_mse"]).mean()),
        "cv_mse_std": float((-cv["test_neg_mse"]).std()),
        "cv_pearson_r_mean": float(cv["test_pearson_r"].mean()),
        "cv_pearson_r_std": float(cv["test_pearson_r"].std()),
        "train_test_gap": 0.0,
        "cv_stability": 0.0,
    }

    return probe, results
