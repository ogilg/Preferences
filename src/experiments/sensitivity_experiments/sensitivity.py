from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import linalg


def compute_sensitivity_regression(
    correlations: list[dict],
    correlation_key: str = "correlation",
) -> dict[str, dict]:
    """
    Use regression to estimate independent contribution of each field.
    Model: correlation ~ β₀ + Σ βᵢ(field_i_same)
    """
    all_fields: set[str] = set()
    for c in correlations:
        if "tags_a" in c:
            all_fields.update(c["tags_a"].keys())

    fields = sorted(all_fields)
    if not fields:
        return {}

    X = []
    y = []
    for c in correlations:
        if "tags_a" not in c or "tags_b" not in c:
            continue
        corr = c[correlation_key]
        if np.isnan(corr):
            continue

        row = []
        for field in fields:
            same = c["tags_a"].get(field) == c["tags_b"].get(field)
            row.append(1 if same else 0)
        X.append(row)
        y.append(corr)

    if len(X) < len(fields) + 1:
        return {}

    X = np.array(X)
    y = np.array(y)

    # Add intercept column
    X_with_const = np.column_stack([np.ones(len(X)), X])

    # OLS via least squares
    beta, residuals, rank, s = linalg.lstsq(X_with_const, y)

    # Compute standard errors using pseudoinverse for stability
    n, p = X_with_const.shape
    y_pred = X_with_const @ beta
    sse = np.sum((y - y_pred) ** 2)
    mse = sse / max(n - p, 1)
    XtX_pinv = linalg.pinv(X_with_const.T @ X_with_const)
    var_beta = mse * XtX_pinv
    std_errs = np.sqrt(np.maximum(np.diag(var_beta), 0))

    # R-squared
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - sse / ss_tot if ss_tot > 0 else 0.0

    results = {
        "_meta": {
            "intercept": float(beta[0]),
            "intercept_se": float(std_errs[0]),
            "r_squared": float(r_squared),
            "n_pairs": len(y),
        }
    }

    for i, field in enumerate(fields):
        results[field] = {
            "coefficient": float(beta[i + 1]),
            "std_err": float(std_errs[i + 1]),
        }

    return results


def compute_sensitivities(
    correlations: list[dict],
    correlation_key: str = "pearson_correlation",
) -> dict[str, dict]:
    """
    For each field, compute mean correlation when that field differs vs matches.

    Returns dict mapping field name to stats:
    {
        "phrasing": {
            "mean_when_same": 0.98,
            "mean_when_diff": 0.85,
            "sensitivity": 0.13,  # drop in correlation when field differs
            "n_same": 50,
            "n_diff": 30,
        },
        ...
    }
    """
    all_fields: set[str] = set()
    for c in correlations:
        if "tags_a" in c:
            all_fields.update(c["tags_a"].keys())
        if "tags_b" in c:
            all_fields.update(c["tags_b"].keys())

    same: dict[str, list[float]] = defaultdict(list)
    diff: dict[str, list[float]] = defaultdict(list)

    for c in correlations:
        if "tags_a" not in c or "tags_b" not in c:
            continue
        corr = c[correlation_key]
        tags_a, tags_b = c["tags_a"], c["tags_b"]

        for field in all_fields:
            val_a = tags_a.get(field)
            val_b = tags_b.get(field)
            if val_a == val_b:
                same[field].append(corr)
            else:
                diff[field].append(corr)

    results = {}
    for field in all_fields:
        same_vals = same[field]
        diff_vals = diff[field]

        if not diff_vals:
            continue

        mean_same = float(np.mean(same_vals)) if same_vals else float("nan")
        mean_diff = float(np.mean(diff_vals))
        sensitivity = mean_same - mean_diff if same_vals else float("nan")

        results[field] = {
            "mean_when_same": mean_same,
            "mean_when_diff": mean_diff,
            "sensitivity": sensitivity,
            "std_when_diff": float(np.std(diff_vals)),
            "n_same": len(same_vals),
            "n_diff": len(diff_vals),
        }

    return results
