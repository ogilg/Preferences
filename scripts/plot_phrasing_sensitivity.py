"""Plot phrasing sensitivity results.

Reads correlations.yaml and creates visualizations of the sensitivity analysis.

Usage:
    python scripts/plot_phrasing_sensitivity.py results/phrasing_sensitivity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_correlations(results_dir: Path) -> list[dict]:
    """Load correlations from YAML file."""
    with open(results_dir / "correlations.yaml") as f:
        return yaml.safe_load(f)


def plot_correlation_matrix(correlations: list[dict], output_path: Path) -> None:
    """Plot correlation matrix heatmap."""
    # Get unique phrasing IDs
    phrasing_ids = sorted(set(
        c["phrasing_a"] for c in correlations
    ) | set(
        c["phrasing_b"] for c in correlations
    ))

    n = len(phrasing_ids)
    id_to_idx = {pid: i for i, pid in enumerate(phrasing_ids)}

    # Build matrices
    win_rate_matrix = np.ones((n, n))
    utility_matrix = np.ones((n, n))

    for c in correlations:
        i = id_to_idx[c["phrasing_a"]]
        j = id_to_idx[c["phrasing_b"]]
        win_rate_matrix[i, j] = c["win_rate_correlation"]
        win_rate_matrix[j, i] = c["win_rate_correlation"]
        utility_matrix[i, j] = c["utility_correlation"]
        utility_matrix[j, i] = c["utility_correlation"]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Win rate correlation
    im1 = axes[0].imshow(win_rate_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[0].set_title("Win Rate Correlation")
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels([f"P{p}" for p in phrasing_ids])
    axes[0].set_yticklabels([f"P{p}" for p in phrasing_ids])

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = win_rate_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

    # Utility correlation
    im2 = axes[1].imshow(utility_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("Utility Correlation")
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels([f"P{p}" for p in phrasing_ids])
    axes[1].set_yticklabels([f"P{p}" for p in phrasing_ids])

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = utility_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            axes[1].text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

    # Add colorbar
    fig.colorbar(im2, ax=axes, shrink=0.8, label="Pearson r")

    plt.suptitle("Phrasing Sensitivity: Correlation Between Templates")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved correlation matrix to {output_path}")
    plt.close()


def plot_bar_comparison(correlations: list[dict], output_path: Path) -> None:
    """Plot bar chart comparing win rate vs utility correlations."""
    labels = [f"P{c['phrasing_a']} vs P{c['phrasing_b']}" for c in correlations]
    win_rates = [c["win_rate_correlation"] for c in correlations]
    utilities = [c["utility_correlation"] for c in correlations]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, win_rates, width, label="Win Rate r", color="steelblue")
    bars2 = ax.bar(x + width / 2, utilities, width, label="Utility r", color="coral")

    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Phrasing Sensitivity: Correlation Between Template Pairs")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.axhline(y=0.9, color="green", linestyle="--", linewidth=0.5, label="High (0.9)")
    ax.set_ylim(-1.1, 1.1)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved bar comparison to {output_path}")
    plt.close()


def print_summary(correlations: list[dict]) -> None:
    """Print summary statistics."""
    win_rates = [c["win_rate_correlation"] for c in correlations]
    utilities = [c["utility_correlation"] for c in correlations]

    print("\n=== Phrasing Sensitivity Summary ===\n")

    print("Win Rate Correlations:")
    print(f"  Mean:   {np.mean(win_rates):.3f}")
    print(f"  Std:    {np.std(win_rates):.3f}")
    print(f"  Min:    {np.min(win_rates):.3f}")
    print(f"  Max:    {np.max(win_rates):.3f}")

    print("\nUtility Correlations:")
    print(f"  Mean:   {np.mean(utilities):.3f}")
    print(f"  Std:    {np.std(utilities):.3f}")
    print(f"  Min:    {np.min(utilities):.3f}")
    print(f"  Max:    {np.max(utilities):.3f}")

    print("\nPairwise Details:")
    for c in correlations:
        print(f"  P{c['phrasing_a']} vs P{c['phrasing_b']}: "
              f"WR={c['win_rate_correlation']:.3f}, "
              f"UT={c['utility_correlation']:.3f}")

    # Interpretation
    avg_corr = (np.mean(win_rates) + np.mean(utilities)) / 2
    if avg_corr > 0.9:
        interpretation = "HIGH: Preferences are robust to phrasing variations."
    elif avg_corr > 0.7:
        interpretation = "MODERATE: Some sensitivity to phrasing detected."
    else:
        interpretation = "LOW: Preferences are highly sensitive to phrasing."

    print(f"\nInterpretation: {interpretation}")


def main():
    parser = argparse.ArgumentParser(description="Plot phrasing sensitivity results")
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing correlations.yaml",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="phrasing_sensitivity",
        help="Prefix for output plot files",
    )

    args = parser.parse_args()

    if not (args.results_dir / "correlations.yaml").exists():
        raise FileNotFoundError(f"correlations.yaml not found in {args.results_dir}")

    correlations = load_correlations(args.results_dir)

    if not correlations:
        print("No correlations found in file.")
        return

    # Print summary
    print_summary(correlations)

    # Generate plots
    plot_correlation_matrix(
        correlations,
        args.results_dir / f"{args.output_prefix}_matrix.png",
    )
    plot_bar_comparison(
        correlations,
        args.results_dir / f"{args.output_prefix}_bars.png",
    )

    print(f"\nPlots saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
