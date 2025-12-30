"""Sensitivity experiments for preference measurements.

This module contains tools for analyzing the robustness of preference
measurements across different experimental conditions (e.g., phrasing variations).
"""

from src.sensitivity_experiments.binary_correlation import (
    win_rate_correlation,
    utility_correlation,
    compute_pairwise_correlations,
    save_correlations,
    save_experiment_config,
)

__all__ = [
    "win_rate_correlation",
    "utility_correlation",
    "compute_pairwise_correlations",
    "save_correlations",
    "save_experiment_config",
]
