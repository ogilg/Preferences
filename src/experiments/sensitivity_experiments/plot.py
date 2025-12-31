"""Analyze and plot sensitivity of preferences to template variations.

Loads all measurement runs from storage, groups by template tags,
and computes correlations to measure sensitivity to each field.

Usage:
    python -m src.experiments.sensitivity_experiments.plot results/
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from src.preferences.storage import (
    list_runs,
    load_thurstonian_data,
    BinaryRunConfig,
    ThurstonianData,
    RESULTS_DIR,
)


def load_all_runs(results_dir: Path) -> list[tuple[BinaryRunConfig, ThurstonianData]]:
    """Load all measurement runs with their Thurstonian data."""
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


def utility_correlation(thurs_a: ThurstonianData, thurs_b: ThurstonianData) -> float:
    """Compute Pearson correlation of utilities between two runs."""
    if len(thurs_a.mu) != len(thurs_b.mu):
        return np.nan

    if thurs_a.task_ids != thurs_b.task_ids:
        id_to_idx_b = {tid: i for i, tid in enumerate(thurs_b.task_ids)}
        try:
            reorder = [id_to_idx_b[tid] for tid in thurs_a.task_ids]
            mu_b = thurs_b.mu[reorder]
        except KeyError:
            return np.nan
    else:
        mu_b = thurs_b.mu

    mu_a = thurs_a.mu

    if len(mu_a) < 2 or np.std(mu_a) < 1e-10 or np.std(mu_b) < 1e-10:
        return np.nan

    r, _ = pearsonr(mu_a, mu_b)
    return float(r) if not np.isnan(r) else np.nan


def get_tag_fields(runs: list[tuple[BinaryRunConfig, ThurstonianData]]) -> set[str]:
    """Get all unique tag field names across runs."""
    fields = set()
    for config, _ in runs:
        for key in config.template_tags:
            fields.add(key)
    return fields


def compute_field_sensitivity(
    runs: list[tuple[BinaryRunConfig, ThurstonianData]],
    field: str,
) -> dict:
    """Compute sensitivity when varying a single field.

    Groups runs by all tags EXCEPT the given field, then computes
    pairwise correlations within each group (where only that field varies).
    """
    groups: dict[tuple, list[tuple[BinaryRunConfig, ThurstonianData]]] = defaultdict(list)

    for config, thurs in runs:
        tags = config.template_tags
        key_items = [(k, v) for k, v in sorted(tags.items()) if k != field]
        key = tuple(key_items)
        groups[key].append((config, thurs))

    all_correlations = []
    field_values = set()

    for group_runs in groups.values():
        if len(group_runs) < 2:
            continue

        for (config_a, thurs_a), (config_b, thurs_b) in combinations(group_runs, 2):
            val_a = config_a.template_tags[field]
            val_b = config_b.template_tags[field]
            field_values.add(val_a)
            field_values.add(val_b)

            corr = utility_correlation(thurs_a, thurs_b)
            if not np.isnan(corr):
                all_correlations.append(corr)

    return {
        "field": field,
        "values": sorted(field_values),
        "correlations": all_correlations,
        "mean": float(np.mean(all_correlations)) if all_correlations else np.nan,
        "std": float(np.std(all_correlations)) if all_correlations else np.nan,
        "n_pairs": len(all_correlations),
    }


def compute_all_field_sensitivities(
    runs: list[tuple[BinaryRunConfig, ThurstonianData]],
) -> list[dict]:
    """Compute sensitivity for each tag field."""
    fields = get_tag_fields(runs)
    results = []
    for field in sorted(fields):
        result = compute_field_sensitivity(runs, field)
        if result["n_pairs"] > 0:
            results.append(result)
    return results


def print_sensitivity_report(
    sensitivities: list[dict],
    runs: list[tuple[BinaryRunConfig, ThurstonianData]],
) -> None:
    """Print sensitivity analysis report."""
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
    """Plot bar chart of mean correlations by field."""
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
