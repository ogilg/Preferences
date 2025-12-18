"""Ranking utilities for pairwise preference data.

This module provides tools to convert pairwise binary comparisons
into utility scores using Thurstonian models.

Example:
    >>> from preferences.ranking import PairwiseData, fit_thurstonian
    >>> from preferences.measure_preferences import measure_binary_preferences
    >>>
    >>> # After measuring binary comparisons
    >>> comparisons = measure_binary_preferences(model, pairs, builder)
    >>> data = PairwiseData.from_comparisons(comparisons, tasks)
    >>> fit = fit_thurstonian(data)
    >>>
    >>> # Get utilities
    >>> for task in fit.ranking():
    ...     print(f"{task.id}: {fit.utility(task):.2f} ± {fit.uncertainty(task):.2f}")
"""

from .thurstonian import (
    PairwiseData,
    ThurstonianResult,
    fit_thurstonian,
)

__all__ = [
    "PairwiseData",
    "ThurstonianResult",
    "fit_thurstonian",
]
