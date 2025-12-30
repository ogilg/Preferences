"""Plot phrasing sensitivity results.

Reads correlations.yaml and creates visualizations of the sensitivity analysis.

Usage:
    python -m src.sensitivity_experiments.plot results/phrasing_sensitivity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_results(results_dir: Path) -> dict:
    """Load correlations and config from YAML files."""
    with open(results_dir / "correlations.yaml") as f:
        correlations_data = yaml.safe_load(f)

    config = None
    config_path = results_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    return {
        "summary": correlations_data["summary"],
        "pairwise": correlations_data["pairwise"],
        "config": config,
    }


def plot_correlation_matrix(
    pairwise: list[dict],
    summary: dict,
    config: dict | None,
    output_path: Path,
) -> None:
    """Plot correlation matrix heatmap."""
    # Get unique phrasing IDs
    phrasing_ids = sorted(set(
        c["phrasing_a"] for c in pairwise
    ) | set(
        c["phrasing_b"] for c in pairwise
    ))

    n = len(phrasing_ids)
    id_to_idx = {pid: i for i, pid in enumerate(phrasing_ids)}

    # Build matrices
    win_rate_matrix = np.ones((n, n))
    utility_matrix = np.ones((n, n))

    for c in pairwise:
        i = id_to_idx[c["phrasing_a"]]
        j = id_to_idx[c["phrasing_b"]]
        win_rate_matrix[i, j] = c["win_rate_correlation"]
        win_rate_matrix[j, i] = c["win_rate_correlation"]
        utility_matrix[i, j] = c["utility_correlation"]
        utility_matrix[j, i] = c["utility_correlation"]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Win rate correlation
    axes[0].imshow(win_rate_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[0].set_title(f"Win Rate Correlation\n(mean={summary['mean_win_rate_correlation']:.2f})")
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
    axes[1].set_title(f"Utility Correlation\n(mean={summary['mean_utility_correlation']:.2f})")
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

    # Build title
    title = "Phrasing Sensitivity: Correlation Between Templates"
    if config:
        title += f"\n{config['model']} | T={config['temperature']} | {config['n_tasks']} tasks"

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved correlation matrix to {output_path}")
    plt.close()


def print_summary(summary: dict, pairwise: list[dict], config: dict | None) -> None:
    """Print summary statistics."""
    print("\n=== Phrasing Sensitivity Summary ===\n")

    if config:
        print(f"Model: {config['model']}")
        print(f"Temperature: {config['temperature']}")
        print(f"Tasks: {config['n_tasks']}")
        print()

    print(f"Mean win rate correlation:  {summary['mean_win_rate_correlation']:.3f}")
    print(f"Mean utility correlation:   {summary['mean_utility_correlation']:.3f}")

    print("\nPairwise Details:")
    for c in pairwise:
        print(f"  P{c['phrasing_a']} vs P{c['phrasing_b']}: "
              f"WR={c['win_rate_correlation']:.3f}, "
              f"UT={c['utility_correlation']:.3f}")

    # Interpretation
    avg_corr = (summary['mean_win_rate_correlation'] + summary['mean_utility_correlation']) / 2
    if avg_corr > 0.9:
        interpretation = "HIGH: Preferences are robust to phrasing variations."
    elif avg_corr > 0.7:
        interpretation = "MODERATE: Some sensitivity to phrasing detected."
    else:
        interpretation = "LOW: Preferences are highly sensitive to phrasing."

    print(f"\nInterpretation: {interpretation}")

    if config and config.get("templates"):
        print("\n=== Templates Used ===\n")
        for t in config["templates"]:
            print(f"Phrasing {t['phrasing_id']} ({t['name']}):")
            # Show first line of prompt (the instruction)
            first_line = t["prompt"].strip().split("\n")[0]
            print(f"  {first_line}")
            print()


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

    results = load_results(args.results_dir)

    if not results["pairwise"]:
        print("No correlations found in file.")
        return

    # Print summary
    print_summary(results["summary"], results["pairwise"], results["config"])

    # Generate plot
    plot_correlation_matrix(
        results["pairwise"],
        results["summary"],
        results["config"],
        args.results_dir / f"{args.output_prefix}_matrix.png",
    )

    print(f"\nPlot saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
