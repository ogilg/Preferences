"""Usage: python -m src.experiments.sensitivity_experiments.plot results/"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.experiments.correlation import compute_pairwise_correlations
from src.preferences.storage import (
    list_runs,
    load_thurstonian_data,
    BinaryRunConfig,
    ThurstonianData,
    RESULTS_DIR,
)
from src.experiments.sensitivity_experiments.sensitivity import compute_sensitivities


def load_all_runs(results_dir: Path) -> list[tuple[BinaryRunConfig, ThurstonianData]]:
    runs = list_runs(results_dir)
    loaded = []
    for config in runs:
        run_dir = results_dir / f"{config.template_id}_{config.model_short}"
        try:
            thurstonian = load_thurstonian_data(run_dir)
            loaded.append((config, thurstonian))
        except FileNotFoundError:
            print(f"Warning: Could not load thurstonian data for {config.template_name}")
    return loaded


def compute_all_field_sensitivities(
    runs: list[tuple[BinaryRunConfig, ThurstonianData]],
) -> list[dict]:
    # Prepare data for unified correlation function
    results = {
        config.template_name: (thurs.mu, thurs.task_ids)
        for config, thurs in runs
    }
    tags = {
        config.template_name: config.template_tags
        for config, thurs in runs
    }

    correlations = compute_pairwise_correlations(results, tags=tags)
    sensitivities = compute_sensitivities(correlations, correlation_key="correlation")

    # Convert to list format expected by print/plot functions
    return [
        {
            "field": field,
            "mean": stats["mean"],
            "std": stats["std"],
            "n_pairs": stats["n_pairs"],
            "values": [],  # Not tracked in new implementation
        }
        for field, stats in sensitivities.items()
    ]


def print_sensitivity_report(
    sensitivities: list[dict],
    runs: list[tuple[BinaryRunConfig, ThurstonianData]],
) -> None:
    print("\n" + "=" * 60)
    print("PREFERENCE SENSITIVITY ANALYSIS")
    print("=" * 60)

    print(f"\nLoaded {len(runs)} measurement runs")

    if not sensitivities:
        print("\nNo pairwise comparisons available.")
        return

    print("\n" + "-" * 60)
    print("SENSITIVITY BY FIELD (varying one field at a time)")
    print("-" * 60)
    print(f"{'Field':<25} {'Mean Corr':<12} {'Std':<10} {'N pairs':<10} {'Values'}")
    print("-" * 60)

    for s in sorted(sensitivities, key=lambda x: x["mean"] if not np.isnan(x["mean"]) else -1, reverse=True):
        values_str = ", ".join(str(v) for v in s["values"][:5])
        if len(s["values"]) > 5:
            values_str += "..."

        mean_str = f"{s['mean']:.3f}" if not np.isnan(s["mean"]) else "N/A"
        std_str = f"{s['std']:.3f}" if not np.isnan(s["std"]) else "N/A"

        print(f"{s['field']:<25} {mean_str:<12} {std_str:<10} {s['n_pairs']:<10} {values_str}")

    print("-" * 60)

    valid_sensitivities = [s for s in sensitivities if not np.isnan(s["mean"])]
    if valid_sensitivities:
        min_sens = min(valid_sensitivities, key=lambda x: x["mean"])
        max_sens = max(valid_sensitivities, key=lambda x: x["mean"])

        print(f"\nMost sensitive to: {min_sens['field']} (mean r = {min_sens['mean']:.3f})")
        print(f"Least sensitive to: {max_sens['field']} (mean r = {max_sens['mean']:.3f})")

        overall_mean = np.mean([s["mean"] for s in valid_sensitivities])
        if overall_mean > 0.9:
            print(f"\nOverall: HIGH robustness (mean r = {overall_mean:.3f})")
        elif overall_mean > 0.7:
            print(f"\nOverall: MODERATE robustness (mean r = {overall_mean:.3f})")
        else:
            print(f"\nOverall: LOW robustness (mean r = {overall_mean:.3f})")


def plot_sensitivity_bars(
    sensitivities: list[dict],
    output_path: Path,
) -> None:
    if not sensitivities:
        return

    fields = [s["field"] for s in sensitivities]
    means = [s["mean"] for s in sensitivities]
    stds = [s["std"] for s in sensitivities]

    _, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(fields))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)

    ax.set_xlabel("Field Varied")
    ax.set_ylabel("Mean Utility Correlation (r)")
    ax.set_title("Preference Sensitivity by Template Field\n(higher = more robust)")
    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.9, color="green", linestyle="--", alpha=0.5, label="High robustness")
    ax.axhline(y=0.7, color="orange", linestyle="--", alpha=0.5, label="Moderate")
    ax.legend()

    for bar, mean in zip(bars, means):
        if not np.isnan(mean):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze preference sensitivity to template variations")
    parser.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=RESULTS_DIR,
        help="Directory containing measurement runs (default: results/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for plot (default: <results_dir>/sensitivity.png)",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return

    print(f"Loading runs from {args.results_dir}...")
    runs = load_all_runs(args.results_dir)

    if not runs:
        print("No measurement runs found.")
        return

    sensitivities = compute_all_field_sensitivities(runs)
    print_sensitivity_report(sensitivities, runs)

    if sensitivities:
        output_path = args.output or (args.results_dir / "sensitivity.png")
        plot_sensitivity_bars(sensitivities, output_path)


if __name__ == "__main__":
    main()
