"""Correlation computation between measurement runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.analysis.correlation.loading import LoadedRun
from src.analysis.correlation.utils import safe_correlation


@dataclass
class CorrelationResult:
    run_a: LoadedRun
    run_b: LoadedRun
    pearson: float
    spearman: float
    n_overlap: int
    common_task_ids: list[str]

    @property
    def label(self) -> str:
        return f"{self.run_a.label} vs {self.run_b.label}"


def correlate_runs(
    run_a: LoadedRun,
    run_b: LoadedRun,
    min_overlap: int = 10,
) -> CorrelationResult | None:
    """Compute correlation between two runs on overlapping tasks."""
    dict_a = run_a.as_dict()
    dict_b = run_b.as_dict()

    common = sorted(set(dict_a.keys()) & set(dict_b.keys()))
    if len(common) < min_overlap:
        return None

    vals_a = np.array([dict_a[tid] for tid in common])
    vals_b = np.array([dict_b[tid] for tid in common])

    pearson = safe_correlation(vals_a, vals_b, "pearson")
    spearman = safe_correlation(vals_a, vals_b, "spearman")

    return CorrelationResult(
        run_a=run_a,
        run_b=run_b,
        pearson=pearson,
        spearman=spearman,
        n_overlap=len(common),
        common_task_ids=common,
    )


def build_correlation_matrix(
    runs: list[LoadedRun],
    method: Literal["pearson", "spearman"] = "pearson",
    min_overlap: int = 10,
) -> tuple[np.ndarray, list[str]]:
    """Build correlation matrix across all runs.

    Returns (matrix, labels) where matrix[i,j] is correlation between runs i and j.
    """
    n = len(runs)
    matrix = np.eye(n)
    labels = [r.label for r in runs]

    for i in range(n):
        for j in range(i + 1, n):
            result = correlate_runs(runs[i], runs[j], min_overlap)
            if result is not None:
                corr = result.pearson if method == "pearson" else result.spearman
                matrix[i, j] = corr
                matrix[j, i] = corr
            else:
                matrix[i, j] = np.nan
                matrix[j, i] = np.nan

    return matrix, labels


def get_aligned_values(
    run_a: LoadedRun,
    run_b: LoadedRun,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Get aligned value arrays for two runs (for scatter plots)."""
    dict_a = run_a.as_dict()
    dict_b = run_b.as_dict()

    common = sorted(set(dict_a.keys()) & set(dict_b.keys()))
    if not common:
        return None

    vals_a = np.array([dict_a[tid] for tid in common])
    vals_b = np.array([dict_b[tid] for tid in common])

    return vals_a, vals_b, common
