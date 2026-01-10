from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import spearmanr

from src.preferences.ranking.thurstonian import ThurstonianResult
from src.experiments.thurstonian_analysis.al_comparison import ConvergenceTrajectory


def plot_utility_scatter(
    al_result: ThurstonianResult,
    full_mle_result: ThurstonianResult,
    true_mu: np.ndarray | None = None,
    n_tasks: int | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """Scatter plot comparing active learning utilities to full MLE.

    Left panel: AL mu vs Full MLE mu
    Right panel (if true_mu provided): Both vs true mu
    """
    has_true = true_mu is not None
    n_panels = 2 if has_true else 1

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    # Ensure same task order
    al_mu = al_result.mu
    full_mle_mu = full_mle_result.mu

    # Infer n_tasks if not provided
    if n_tasks is None:
        n_tasks = len(al_mu)

    # Panel 1: AL vs Full MLE
    ax = axes[0]
    spearman = spearmanr(al_mu, full_mle_mu).correlation

    ax.scatter(full_mle_mu, al_mu, alpha=0.7, edgecolors="k", linewidths=0.5)

    # Identity line
    lims = [
        min(full_mle_mu.min(), al_mu.min()),
        max(full_mle_mu.max(), al_mu.max()),
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y = x")

    ax.set_xlabel("Full MLE utility (mu)")
    ax.set_ylabel("Active Learning utility (mu)")
    ax.set_title(f"AL vs Full MLE (N={n_tasks}, Spearman rho={spearman:.3f})")
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")

    # Panel 2: Both vs True (if available)
    if has_true:
        ax2 = axes[1]
        spearman_al = spearmanr(al_mu, true_mu).correlation
        spearman_full = spearmanr(full_mle_mu, true_mu).correlation

        ax2.scatter(true_mu, al_mu, alpha=0.7, label=f"AL (rho={spearman_al:.3f})", marker="o")
        ax2.scatter(true_mu, full_mle_mu, alpha=0.7, label=f"Full MLE (rho={spearman_full:.3f})", marker="s")

        lims = [
            min(true_mu.min(), al_mu.min(), full_mle_mu.min()),
            max(true_mu.max(), al_mu.max(), full_mle_mu.max()),
        ]
        ax2.plot(lims, lims, "k--", alpha=0.5)

        ax2.set_xlabel("True utility (mu)")
        ax2.set_ylabel("Estimated utility (mu)")
        ax2.set_title(f"Recovery of True Utilities (N={n_tasks})")
        ax2.legend(loc="lower right")
        ax2.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    return fig


def plot_convergence_curve(
    trajectory: ConvergenceTrajectory,
    full_mle_accuracy: float | None = None,
    n_tasks: int | None = None,
    figsize: tuple[float, float] = (12, 4),
) -> Figure:
    """Multi-panel convergence plot showing metrics vs pairs queried.

    Panel 1: Spearman correlation with Full MLE
    Panel 2: Held-out accuracy
    Panel 3 (if available): Spearman correlation with true utilities
    """
    has_true = trajectory.spearman_vs_true is not None and len(trajectory.spearman_vs_true) > 0
    n_panels = 3 if has_true else 2

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    x = trajectory.cumulative_pairs

    # Build N label
    n_label = f" (N={n_tasks})" if n_tasks is not None else ""

    # Panel 1: Spearman vs Full MLE
    ax1 = axes[0]
    ax1.plot(x, trajectory.spearman_vs_full_mle, "o-", color="C0", markersize=4)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect agreement")
    ax1.set_xlabel("Pairs queried")
    ax1.set_ylabel("Spearman rho")
    ax1.set_title(f"Convergence to Full MLE{n_label}")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right")

    # Panel 2: Held-out accuracy
    ax2 = axes[1]
    ax2.plot(x, trajectory.held_out_accuracy, "o-", color="C1", markersize=4, label="Active Learning")
    if full_mle_accuracy is not None:
        ax2.axhline(y=full_mle_accuracy, color="C2", linestyle="--", label=f"Full MLE ({full_mle_accuracy:.3f})")
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random baseline")
    ax2.set_xlabel("Pairs queried")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Held-out Prediction Accuracy")
    ax2.set_ylim(0.4, 1.0)
    ax2.legend(loc="lower right")

    # Panel 3: Spearman vs True (if available)
    if has_true:
        ax3 = axes[2]
        ax3.plot(x, trajectory.spearman_vs_true, "o-", color="C3", markersize=4)
        ax3.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect recovery")
        ax3.set_xlabel("Pairs queried")
        ax3.set_ylabel("Spearman rho")
        ax3.set_title("Recovery of True Utilities")
        ax3.set_ylim(0, 1.05)
        ax3.legend(loc="lower right")

    fig.tight_layout()
    return fig


def plot_held_out_comparison(
    al_accuracy: float,
    full_mle_accuracy: float,
    al_pairs: int,
    full_mle_pairs: int,
    n_tasks: int | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> Figure:
    """Bar chart comparing held-out accuracy and efficiency."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Build N label
    n_label = f" (N={n_tasks})" if n_tasks is not None else ""

    # Panel 1: Accuracy comparison
    ax1 = axes[0]
    labels = ["Active Learning", "Full MLE"]
    accuracies = [al_accuracy, full_mle_accuracy]
    colors = ["C0", "C2"]

    bars = ax1.bar(labels, accuracies, color=colors, edgecolor="k", linewidth=0.5)
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random baseline")

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax1.set_ylabel("Held-out Accuracy")
    ax1.set_title(f"Prediction Accuracy on Held-out Pairs{n_label}")
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc="lower right")

    # Panel 2: Efficiency (pairs used)
    ax2 = axes[1]
    pairs = [al_pairs, full_mle_pairs]
    efficiency = al_pairs / full_mle_pairs if full_mle_pairs > 0 else 0

    bars = ax2.bar(labels, pairs, color=colors, edgecolor="k", linewidth=0.5)

    for bar, p in zip(bars, pairs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + full_mle_pairs * 0.02,
            f"{p}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax2.set_ylabel("Number of Pairs")
    ax2.set_title(f"Sample Efficiency (AL uses {efficiency:.1%} of pairs)")

    fig.tight_layout()
    return fig
