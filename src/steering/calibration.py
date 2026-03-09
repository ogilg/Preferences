"""Coefficient calibration based on activation norms."""

from __future__ import annotations

from pathlib import Path

from src.probes.core.activations import compute_activation_norms


def suggest_coefficient_range(
    activations_path: Path,
    layer: int,
    multipliers: list[float] | None = None,
) -> list[float]:
    """Return steering coefficients as multiples of the mean activation norm at the given layer."""
    if multipliers is None:
        multipliers = [-0.1, -0.05, 0.0, 0.05, 0.1]

    norms = compute_activation_norms(activations_path, layers=[layer])
    mean_norm = norms[layer]

    return [mean_norm * m for m in multipliers]
