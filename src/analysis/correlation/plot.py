"""Plotting functions for correlation analysis."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.analysis.correlation.loading import LoadedRun
from src.analysis.correlation.compute import (
    CorrelationResult,
    get_aligned_values,
    build_correlation_matrix,
)


def plot_scatter(
    run_a: LoadedRun,
    run_b: LoadedRun,
    corr_result: CorrelationResult | None = None,
    output_path: Path | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter plot of two runs' values on overlapping tasks."""
    aligned = get_aligned_values(run_a, run_b)
    if aligned is None:
        raise ValueError(f"No overlapping tasks between {run_a.label} and {run_b.label}")

    vals_a, vals_b, common = aligned

    own_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(vals_a, vals_b, alpha=0.6, s=20)

    # Fit line
    if len(vals_a) > 1 and np.std(vals_a) > 1e-10:
        z = np.polyfit(vals_a, vals_b, 1)
        p = np.poly1d(z)
        x_line = np.linspace(vals_a.min(), vals_a.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=1)

    ax.set_xlabel(run_a.label)
    ax.set_ylabel(run_b.label)

    if title:
        ax.set_title(title)
    elif corr_result:
        ax.set_title(f"r={corr_result.pearson:.3f}, ρ={corr_result.spearman:.3f}, n={corr_result.n_overlap}")
    else:
        ax.set_title(f"n={len(common)}")

    if own_fig and output_path:
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    return ax


def plot_scatter_grid(
    runs: list[LoadedRun],
    output_path: Path,
    model_name: str,
) -> None:
    """Grid of scatter plots for all pairs of runs."""
    n = len(runs)
    if n < 2:
        print(f"Need at least 2 runs for scatter grid, got {n}")
        return

    fig, axes = plt.subplots(n - 1, n - 1, figsize=(3 * (n - 1), 3 * (n - 1)))
    if n == 2:
        axes = np.array([[axes]])

    for i in range(n - 1):
        for j in range(n - 1):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue

            run_a = runs[j]
            run_b = runs[i + 1]

            aligned = get_aligned_values(run_a, run_b)
            if aligned is None:
                ax.text(0.5, 0.5, "No overlap", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            vals_a, vals_b, common = aligned
            ax.scatter(vals_a, vals_b, alpha=0.5, s=10)

            from src.analysis.correlation.utils import safe_correlation
            r = safe_correlation(vals_a, vals_b, "pearson")
            ax.set_title(f"r={r:.2f}", fontsize=9)

            if i == n - 2:
                ax.set_xlabel(run_a.label, fontsize=8)
            if j == 0:
                ax.set_ylabel(run_b.label, fontsize=8)

            ax.tick_params(labelsize=7)

    fig.suptitle(f"Correlation Grid: {model_name}", fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_correlation_matrix(
    runs: list[LoadedRun],
    output_path: Path,
    model_name: str,
    method: Literal["pearson", "spearman"] = "pearson",
    min_overlap: int = 10,
) -> None:
    """Heatmap of correlation matrix."""
    matrix, labels = build_correlation_matrix(runs, method, min_overlap)

    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 0.8), max(6, len(runs) * 0.6)))

    mask = np.isnan(matrix)
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        mask=mask,
        ax=ax,
        annot_kws={"size": 8},
    )

    ax.set_title(f"Correlation Matrix ({method.title()}): {model_name}")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Print summary
    triu_vals = matrix[np.triu_indices(len(runs), k=1)]
    triu_vals = triu_vals[~np.isnan(triu_vals)]
    if len(triu_vals) > 0:
        print(f"  Mean correlation: {np.mean(triu_vals):.3f}")
        print(f"  Range: [{np.min(triu_vals):.3f}, {np.max(triu_vals):.3f}]")


def plot_type_comparison(
    runs: list[LoadedRun],
    output_path: Path,
    model_name: str,
) -> None:
    """Bar chart comparing correlations between measurement types."""
    from collections import defaultdict
    from src.analysis.correlation.compute import correlate_runs
    from src.analysis.correlation.loading import MeasurementType

    # Group runs by measurement type
    by_type: dict[MeasurementType, list[LoadedRun]] = defaultdict(list)
    for run in runs:
        by_type[run.measurement_type].append(run)

    # Compute cross-type correlations
    type_pairs: list[tuple[str, list[float]]] = []
    types = list(MeasurementType)

    for i, type_a in enumerate(types):
        for type_b in types[i + 1 :]:
            runs_a = by_type.get(type_a, [])
            runs_b = by_type.get(type_b, [])

            if not runs_a or not runs_b:
                continue

            correlations = []
            for ra in runs_a:
                for rb in runs_b:
                    result = correlate_runs(ra, rb)
                    if result is not None:
                        correlations.append(result.pearson)

            if correlations:
                label = f"{type_a.short_name}\nvs\n{type_b.short_name}"
                type_pairs.append((label, correlations))

    if not type_pairs:
        print("No cross-type correlations found")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(type_pairs) * 1.5), 6))

    labels = [p[0] for p in type_pairs]
    means = [np.mean(p[1]) for p in type_pairs]
    stds = [np.std(p[1]) for p in type_pairs]
    counts = [len(p[1]) for p in type_pairs]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)

    for bar, mean, n in zip(bars, means, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{mean:.2f}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Pearson Correlation")
    ax.set_title(f"Cross-Type Correlations: {model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def _get_type_pair_label(result: CorrelationResult) -> str:
    """Get a label for the measurement type pair (e.g., 'qual vs stated')."""
    type_a = result.run_a.config.template_name
    type_b = result.run_b.config.template_name

    def simplify(name: str) -> str:
        if "qualitative" in name:
            return "qual"
        elif "stated" in name or "rating" in name:
            return "stated"
        elif "revealed" in name:
            return "revealed"
        return "other"

    a, b = sorted([simplify(type_a), simplify(type_b)])
    return f"{a} vs {b}"


def plot_slope_vs_correlation(
    runs: list[LoadedRun],
    output_path: Path,
    title: str,
    min_overlap: int = 10,
    color_by_type: bool = True,
    template_filter: str | None = None,
) -> None:
    """Scatter plot of slope vs correlation for all run pairs.

    Args:
        color_by_type: If True, color points by measurement type pair.
        template_filter: If set, only include runs whose template_name contains this string.
    """
    from src.analysis.correlation.compute import correlate_runs

    if template_filter:
        runs = [r for r in runs if template_filter in r.config.template_name]
        print(f"Filtered to {len(runs)} runs containing '{template_filter}'")

    results = []
    for i, run_a in enumerate(runs):
        for run_b in runs[i + 1:]:
            result = correlate_runs(run_a, run_b, min_overlap)
            if result is not None:
                results.append(result)

    if not results:
        print("No valid pairs for slope vs correlation plot")
        return

    correlations = [r.pearson for r in results]
    slopes = [r.slope for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))

    if color_by_type:
        type_labels = [_get_type_pair_label(r) for r in results]
        unique_labels = sorted(set(type_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        color_map = dict(zip(unique_labels, colors))

        for label in unique_labels:
            mask = [tl == label for tl in type_labels]
            corr_subset = [c for c, m in zip(correlations, mask) if m]
            slope_subset = [s for s, m in zip(slopes, mask) if m]
            ax.scatter(corr_subset, slope_subset, alpha=0.6, s=40,
                      c=[color_map[label]], edgecolor="white", linewidth=0.5,
                      label=f"{label} (n={len(corr_subset)})")
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.scatter(correlations, slopes, alpha=0.6, s=40, c="steelblue",
                  edgecolor="white", linewidth=0.5)

    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="slope=1 (same scale)")

    ax.set_xlabel("Pearson Correlation", fontsize=11)
    ax.set_ylabel("Slope", fontsize=11)
    ax.set_title(title, fontsize=12)

    high_corr = [r for r in results if r.pearson > 0.7]
    if high_corr:
        slopes_high = [r.slope for r in high_corr]
        text = f"r>0.7: n={len(high_corr)}, slope=[{min(slopes_high):.2f}, {max(slopes_high):.2f}]"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
