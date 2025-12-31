"""Plotting functions for Thurstonian model results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.container import BarContainer
    from src.preferences.storage import ThurstonianData, BinaryRunConfig


def plot_utility_ranking(
    data: "ThurstonianData",
    config: "BinaryRunConfig | None" = None,
    figsize: tuple[float, float] = (10, 6),
    color: str = "steelblue",
    picker: bool = False,
) -> tuple[plt.Figure, "BarContainer"]:
    """Bar chart of utilities (mu) with uncertainty (sigma) error bars.

    Tasks are sorted by utility (highest first).

    Args:
        data: ThurstonianData with mu, sigma, task_ids.
        config: Optional BinaryRunConfig for title metadata.
        figsize: Figure size in inches.
        color: Bar color.
        picker: Enable click picking on bars.

    Returns:
        Tuple of (Figure, BarContainer).
    """
    # Sort by utility (highest first)
    order = data.ranking_order()
    sorted_ids = [data.task_ids[i] for i in order]
    sorted_mu = data.mu[order]
    sorted_sigma = data.sigma[order]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(sorted_ids))
    bars = ax.bar(
        x, sorted_mu, yerr=sorted_sigma, capsize=3,
        color=color, alpha=0.8, picker=picker,
    )

    ax.set_xlabel("Task")
    ax.set_ylabel("Utility (mu)")
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_ids, rotation=45, ha="right")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    # Build title
    if config is not None:
        title = f"Utility Ranking: {config.template_id} / {config.model_short}"
    else:
        title = "Utility Ranking"

    if not data.converged:
        title += " (not converged)"

    ax.set_title(title)
    fig.tight_layout()

    return fig, bars
