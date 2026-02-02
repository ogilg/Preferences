from __future__ import annotations

from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import BarContainer

from src.fitting.thurstonian_fitting import ThurstonianResult


class RunConfig(Protocol):
    template_name: str
    model_short: str


def plot_utility_ranking(
    result: ThurstonianResult,
    config: RunConfig | None = None,
    figsize: tuple[float, float] = (10, 6),
    color: str = "steelblue",
    picker: bool = False,
) -> tuple[plt.Figure, BarContainer]:
    """Bar chart of utilities with sigma error bars, sorted highest first."""
    order = np.argsort(-result.mu)
    sorted_ids = [result.tasks[i].id for i in order]
    sorted_mu = result.mu[order]
    sorted_sigma = result.sigma[order]

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

    if config is not None:
        title = f"Utility Ranking: {config.template_name} / {config.model_short}"
    else:
        title = "Utility Ranking"

    if not result.converged:
        title += " (not converged)"

    ax.set_title(title)
    fig.tight_layout()

    return fig, bars
