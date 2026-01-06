"""Usage: python -m src.experiments.sensitivity_experiments.plot results/"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

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
) -> tuple[list[dict], list[dict]]:
    """Returns (sensitivities, correlations)."""
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

    sensitivities_list = [
        {
            "field": field,
            "mean": stats["mean"],
            "std": stats["std"],
            "n_pairs": stats["n_pairs"],
        }
        for field, stats in sensitivities.items()
    ]

    return sensitivities_list, correlations


def save_sensitivity_report(
    sensitivities: list[dict],
    correlations: list[dict],
    n_runs: int,
    output_path: Path,
) -> None:
    """Save sensitivity analysis results to YAML."""
    valid = [s for s in sensitivities if not np.isnan(s["mean"])]

    report = {
        "n_runs": n_runs,
        "by_field": {
            s["field"]: {
                "mean_correlation": float(s["mean"]),
                "std": float(s["std"]),
                "n_pairs": s["n_pairs"],
            }
            for s in valid
        },
        "pairwise_correlations": correlations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)


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
        help="Output directory (default: <results_dir>)",
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

    print(f"Loaded {len(runs)} runs, computing correlations...")
    sensitivities, correlations = compute_all_field_sensitivities(runs)

    output_dir = args.output or args.results_dir

    report_path = output_dir / "sensitivity.yaml"
    save_sensitivity_report(sensitivities, correlations, len(runs), report_path)
    print(f"Saved report to {report_path}")

    if sensitivities:
        plot_path = output_dir / "sensitivity.png"
        plot_sensitivity_bars(sensitivities, plot_path)
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
