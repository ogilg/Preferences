"""Transitivity measurement for pairwise preference data.

Measures how often preferences violate transitivity (form cycles).
A cycle: A ≻ B, B ≻ C, but C ≻ A.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class TransitivityResult:
    """Results from transitivity analysis."""

    cycle_probability: float  # P(cycle) across all triads
    n_triads: int
    n_cycles: int

    @property
    def log_cycle_prob(self) -> float:
        """Log10 of cycle probability (as in the paper's Figure 7)."""
        if self.cycle_probability <= 0:
            return float("-inf")
        return float(np.log10(self.cycle_probability))


def measure_transitivity(wins: np.ndarray) -> TransitivityResult:
    """Measure transitivity violations in pairwise preference data.

    Args:
        wins: Matrix where wins[i,j] = number of times i beat j.

    Returns:
        TransitivityResult with cycle probability and counts.
    """
    n = wins.shape[0]

    if n < 3:
        return TransitivityResult(cycle_probability=0.0, n_triads=0, n_cycles=0)

    # Preference probabilities: P(i > j)
    total = wins + wins.T
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.where(total > 0, wins / total, 0.5)

    n_triads = 0
    cycle_prob_sum = 0.0

    for i, j, k in itertools.combinations(range(n), 3):
        n_triads += 1

        # Probabilistic cycle: P(i>j>k>i) + P(i>k>j>i)
        p_cycle_1 = probs[i, j] * probs[j, k] * probs[k, i]
        p_cycle_2 = probs[i, k] * probs[k, j] * probs[j, i]
        cycle_prob_sum += p_cycle_1 + p_cycle_2

    avg_cycle_prob = cycle_prob_sum / n_triads if n_triads > 0 else 0.0

    # Count hard cycles (using majority preferences)
    prefers = wins > wins.T
    n_cycles = 0
    for i, j, k in itertools.combinations(range(n), 3):
        if (prefers[i, j] and prefers[j, k] and prefers[k, i]) or (
            prefers[i, k] and prefers[k, j] and prefers[j, i]
        ):
            n_cycles += 1

    return TransitivityResult(
        cycle_probability=avg_cycle_prob,
        n_triads=n_triads,
        n_cycles=n_cycles,
    )
