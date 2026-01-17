"""Sensitivity experiments for preference measurements.

This module contains tools for analyzing the robustness of preference
measurements across different experimental conditions (e.g., phrasing variations).
"""

from src.running_measurements.utils.correlation import (
    compute_pairwise_correlations,
    save_experiment_config,
    utility_vector_correlation,
)
from src.analysis.sensitivity.revealed_correlation import (
    win_rate_correlation,
    save_correlations,
)
from src.analysis.sensitivity.stated_correlation import (
    scores_to_vector,
)

__all__ = [
    "compute_pairwise_correlations",
    "utility_vector_correlation",
    "win_rate_correlation",
    "scores_to_vector",
    "save_correlations",
    "save_experiment_config",
]
