"""Unified seed sensitivity analysis for all measurement types.

Analyzes how stable preference measurements are across different random seeds.
Computes correlations between runs with different seeds but same template/model.

Usage:
    python -m src.analysis.sensitivity.plot_seed_sensitivity --experiment-id stability_v1
    python -m src.analysis.sensitivity.plot_seed_sensitivity --experiment-id stability_v1 --type stated
    python -m src.analysis.sensitivity.plot_seed_sensitivity --experiment-id stability_v1 --cross-type
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.analysis.correlation.utils import utility_vector_correlation
from src.measurement_storage import EXPERIMENTS_DIR


MeasurementType = Literal["stated", "revealed"]
OUTPUT_DIR = Path(__file__).parent / "plots"


def parse_run_name(name: str) -> dict[str, str | int]:
    """Parse run directory name to extract template, model, format, seeds."""
    parts = {}

    if "_rseed" in name:
        base, rseed = name.rsplit("_rseed", 1)
        parts["rating_seed"] = int(rseed)
    else:
        base = name
        parts["rating_seed"] = 0

    if "_cseed" in base:
        base, cseed = base.rsplit("_cseed", 1)
        parts["completion_seed"] = int(cseed)
    else:
        parts["completion_seed"] = 0

    parts["base_name"] = base
    return parts


def load_run_scores(run_dir: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Load scores from a stated preference run. Returns (scores, task_ids, origins)."""
    measurements_path = run_dir / "measurements.yaml"
    if not measurements_path.exists():
        return np.array([]), [], []

    with open(measurements_path) as f:
        data = yaml.safe_load(f)

    if not data:
        return np.array([]), [], []

    # Filter to numeric scores only
    numeric_data = [m for m in data if isinstance(m["score"], (int, float))]
    if not numeric_data:
        return np.array([]), [], []

    task_ids = [m["task_id"] for m in numeric_data]
    scores = np.array([m["score"] for m in numeric_data])
    # Origin may not exist in old data - fall back to parsing from task_id
    origins = []
    for m in numeric_data:
        if "origin" in m:
            origins.append(m["origin"])
        else:
            # Fallback for old data without origin field
            tid = m["task_id"]
            if tid.startswith("wildchat_"):
                origins.append("WILDCHAT")
            elif tid.startswith("alpaca_"):
                origins.append("ALPACA")
            elif tid.startswith("competition_math_"):
                origins.append("MATH")
            elif tid.startswith("bailbench_"):
                origins.append("BAILBENCH")
            else:
                origins.append("UNKNOWN")
    return scores, task_ids, origins


def load_run_utilities(run_dir: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Load utilities from a revealed preference run (Thurstonian fit). Returns (utilities, task_ids, origins)."""
    csv_files = list(run_dir.glob("thurstonian_*.csv"))
    if not csv_files:
        return np.array([]), [], []

    csv_path = csv_files[0]
    task_ids = []
    utilities = []

    with open(csv_path) as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                task_ids.append(parts[0])
                utilities.append(float(parts[1]))

    # For revealed, we need to infer origin from task_id (Thurstonian CSV doesn't have it)
    origins = []
    for tid in task_ids:
        if tid.startswith("wildchat_"):
            origins.append("WILDCHAT")
        elif tid.startswith("alpaca_"):
            origins.append("ALPACA")
        elif tid.startswith("competition_math_"):
            origins.append("MATH")
        elif tid.startswith("bailbench_"):
            origins.append("BAILBENCH")
        else:
            origins.append("UNKNOWN")

    return np.array(utilities), task_ids, origins


# Type for run data: (values, task_ids, origins)
RunData = tuple[np.ndarray, list[str], list[str]]


def load_runs_by_seed(
    experiment_dir: Path,
    measurement_type: MeasurementType,
    template_filter: str | None = None,
) -> dict[str, dict[int, RunData]]:
    """Load runs grouped by template base name, then by seed."""
    if measurement_type == "stated":
        subdirs = ["post_task_stated"]
        load_fn = load_run_scores
    else:
        subdirs = ["post_task_revealed", "post_task_active_learning"]
        load_fn = load_run_utilities

    by_template: dict[str, dict[int, RunData]] = defaultdict(dict)

    for subdir in subdirs:
        results_dir = experiment_dir / subdir
        if not results_dir.exists():
            continue

        for run_dir in sorted(results_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            parsed = parse_run_name(run_dir.name)
            base_name = parsed["base_name"]
            seed = parsed["rating_seed"]

            if template_filter and template_filter not in base_name:
                continue

            values, task_ids, origins = load_fn(run_dir)
            if len(values) == 0:
                continue

            by_template[base_name][seed] = (values, task_ids, origins)

    return dict(by_template)


def compute_cross_seed_correlations(
    seed_data: dict[int, RunData],
    method: Literal["pearson", "spearman"] = "pearson",
    origin_filter: str | None = None,
) -> list[tuple[int, int, float]]:
    """Compute correlations between all seed pairs, optionally filtered by origin."""
    correlations = []
    seeds = sorted(seed_data.keys())

    for seed_a, seed_b in combinations(seeds, 2):
        vals_a, ids_a, origins_a = seed_data[seed_a]
        vals_b, ids_b, origins_b = seed_data[seed_b]

        # Filter by origin if specified
        if origin_filter:
            mask_a = np.array([o == origin_filter for o in origins_a])
            mask_b = np.array([o == origin_filter for o in origins_b])
            vals_a = vals_a[mask_a]
            ids_a = [i for i, m in zip(ids_a, mask_a) if m]
            vals_b = vals_b[mask_b]
            ids_b = [i for i, m in zip(ids_b, mask_b) if m]

        if len(vals_a) == 0 or len(vals_b) == 0:
            continue

        corr = utility_vector_correlation(vals_a, ids_a, vals_b, ids_b, method)
        if not np.isnan(corr):
            correlations.append((seed_a, seed_b, corr))

    return correlations


def plot_seed_sensitivity(
    by_template: dict[str, dict[int, RunData]],
    output_path: Path,
    title: str,
    method: Literal["pearson", "spearman"] = "pearson",
) -> dict:
    """Create seed sensitivity visualization."""
    template_correlations: dict[str, list[float]] = {}
    all_correlations: list[float] = []

    for base_name, seed_data in by_template.items():
        if len(seed_data) < 2:
            continue

        corrs = compute_cross_seed_correlations(seed_data, method)
        if corrs:
            corr_values = [c[2] for c in corrs]
            template_correlations[base_name] = corr_values
            all_correlations.extend(corr_values)

    if not all_correlations:
        print("No cross-seed correlations found (need 2+ seeds per template)")
        return {}

    mean_corr = np.mean(all_correlations)
    n_pairs = len(all_correlations)

    fig, ax = plt.subplots(figsize=(8, 5))

    template_names = sorted(template_correlations.keys())
    means = [np.mean(template_correlations[t]) for t in template_names]
    stds = [np.std(template_correlations[t]) for t in template_names]

    x = np.arange(len(template_names))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    short_names = [n.replace("_llama-3.1-8b_regex", "").replace("_canonical", "") for n in template_names]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Cross-Seed Correlation", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(mean_corr, color="red", linestyle="--", alpha=0.7, linewidth=2)

    ax.text(
        0.98, mean_corr + 0.03,
        f"Mean: {mean_corr:.2f}",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=10, color="red"
    )

    ax.set_title(f"{title}\n({n_pairs} seed pairs)", fontsize=12)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

    return {
        "n_templates": len(template_correlations),
        "n_pairs": n_pairs,
        "mean_correlation": float(mean_corr),
        "std_correlation": float(np.std(all_correlations)),
        "min_correlation": float(np.min(all_correlations)),
        "max_correlation": float(np.max(all_correlations)),
        "by_template": {
            name: {
                "mean": float(np.mean(corrs)),
                "std": float(np.std(corrs)),
                "n_pairs": len(corrs),
            }
            for name, corrs in template_correlations.items()
        },
    }


def plot_seed_sensitivity_by_origin(
    by_template: dict[str, dict[int, RunData]],
    output_path: Path,
    title: str,
    method: Literal["pearson", "spearman"] = "pearson",
) -> dict:
    """Create seed sensitivity visualization broken down by origin dataset."""
    origins = ["WILDCHAT", "ALPACA", "MATH", "BAILBENCH"]
    origin_correlations: dict[str, list[float]] = {o: [] for o in origins}

    for base_name, seed_data in by_template.items():
        if len(seed_data) < 2:
            continue

        for origin in origins:
            corrs = compute_cross_seed_correlations(seed_data, method, origin_filter=origin)
            if corrs:
                origin_correlations[origin].extend([c[2] for c in corrs])

    # Filter to origins with data
    origins_with_data = [o for o in origins if origin_correlations[o]]
    if not origins_with_data:
        print("No origin-based correlations found")
        return {}

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(origins_with_data))
    means = [np.mean(origin_correlations[o]) for o in origins_with_data]
    stds = [np.std(origin_correlations[o]) for o in origins_with_data]
    counts = [len(origin_correlations[o]) for o in origins_with_data]

    colors = {"WILDCHAT": "#4ECDC4", "ALPACA": "#FF6B6B", "MATH": "#45B7D1", "BAILBENCH": "#96CEB4"}
    bar_colors = [colors.get(o, "steelblue") for o in origins_with_data]

    bars = ax.bar(x, means, yerr=stds, capsize=4, color=bar_colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(origins_with_data, fontsize=11)
    ax.set_ylabel("Cross-Seed Correlation", fontsize=11)
    ax.set_ylim(0, 1.05)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] + 0.02,
                f"n={count}", ha="center", va="bottom", fontsize=9)

    overall_mean = np.mean([c for corrs in origin_correlations.values() for c in corrs])
    ax.axhline(overall_mean, color="red", linestyle="--", alpha=0.7, linewidth=2)
    ax.text(0.98, overall_mean + 0.02, f"Overall: {overall_mean:.2f}",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=10, color="red")

    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

    return {
        "by_origin": {
            origin: {
                "mean": float(np.mean(origin_correlations[origin])),
                "std": float(np.std(origin_correlations[origin])),
                "n_pairs": len(origin_correlations[origin]),
            }
            for origin in origins_with_data
        },
        "overall_mean": float(overall_mean),
    }


def compute_cross_type_correlations(
    stated_by_seed: dict[int, RunData],
    revealed_by_seed: dict[int, RunData],
    method: Literal["pearson", "spearman"] = "pearson",
) -> list[float]:
    """Compute stated-revealed correlation for each matching seed."""
    correlations = []
    common_seeds = set(stated_by_seed.keys()) & set(revealed_by_seed.keys())

    for seed in sorted(common_seeds):
        vals_s, ids_s, _ = stated_by_seed[seed]
        vals_r, ids_r, _ = revealed_by_seed[seed]
        corr = utility_vector_correlation(vals_s, ids_s, vals_r, ids_r, method)
        if not np.isnan(corr):
            correlations.append(corr)

    return correlations


def plot_cross_type_sensitivity(
    stated_by_template: dict[str, dict[int, RunData]],
    revealed_by_seed: dict[int, RunData],
    output_path: Path,
    title: str,
    method: Literal["pearson", "spearman"] = "pearson",
) -> dict:
    """Plot how stable stated-revealed correlation is across seeds."""
    template_correlations: dict[str, list[float]] = {}
    all_correlations: list[float] = []

    for template_name, stated_seeds in stated_by_template.items():
        corrs = compute_cross_type_correlations(stated_seeds, revealed_by_seed, method)
        if corrs:
            template_correlations[template_name] = corrs
            all_correlations.extend(corrs)

    if not all_correlations:
        print("No cross-type correlations found (need matching seeds)")
        return {}

    mean_corr = np.mean(all_correlations)
    n_pairs = len(all_correlations)

    fig, ax = plt.subplots(figsize=(8, 5))

    template_names = sorted(template_correlations.keys())
    means = [np.mean(template_correlations[t]) for t in template_names]
    stds = [np.std(template_correlations[t]) for t in template_names]

    x = np.arange(len(template_names))
    ax.bar(x, means, yerr=stds, capsize=4, color="coral", alpha=0.8)
    ax.set_xticks(x)
    short_names = [n.replace("_llama-3.1-8b_regex", "") for n in template_names]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Stated-Revealed Correlation", fontsize=11)
    ax.set_ylim(-0.5, 1.05)
    ax.axhline(mean_corr, color="red", linestyle="--", alpha=0.7, linewidth=2)
    ax.axhline(0, color="black", linestyle="-", alpha=0.3, linewidth=1)

    ax.text(
        0.98, mean_corr + 0.03,
        f"Mean: {mean_corr:.2f}",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=10, color="red"
    )

    ax.set_title(f"{title}\n({n_pairs} seed-matched pairs)", fontsize=12)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

    return {
        "n_templates": len(template_correlations),
        "n_pairs": n_pairs,
        "mean_correlation": float(mean_corr),
        "std_correlation": float(np.std(all_correlations)),
        "min_correlation": float(np.min(all_correlations)),
        "max_correlation": float(np.max(all_correlations)),
        "by_template": {
            name: {
                "mean": float(np.mean(corrs)),
                "std": float(np.std(corrs)),
                "n_seeds": len(corrs),
            }
            for name, corrs in template_correlations.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze seed sensitivity of preference measurements"
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="Experiment ID to analyze",
    )
    parser.add_argument(
        "--type",
        choices=["stated", "revealed", "both"],
        default="both",
        help="Measurement type to analyze (default: both)",
    )
    parser.add_argument(
        "--cross-type",
        action="store_true",
        help="Also analyze stated-revealed correlation stability across seeds",
    )
    parser.add_argument(
        "--by-origin",
        action="store_true",
        help="Also plot seed sensitivity broken down by origin dataset",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Filter to templates containing this string",
    )
    parser.add_argument(
        "--method",
        choices=["pearson", "spearman"],
        default="pearson",
        help="Correlation method (default: pearson)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: src/analysis/sensitivity/plots)",
    )
    args = parser.parse_args()

    experiment_dir = EXPERIMENTS_DIR / args.experiment_id
    if not experiment_dir.exists():
        print(f"Experiment not found: {experiment_dir}")
        return

    output_dir = args.output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%m%d%y")

    types_to_analyze: list[MeasurementType] = []
    if args.type in ("stated", "both"):
        types_to_analyze.append("stated")
    if args.type in ("revealed", "both"):
        types_to_analyze.append("revealed")

    all_summaries = {}
    loaded_data: dict[MeasurementType, dict] = {}

    for mtype in types_to_analyze:
        print(f"\nAnalyzing {mtype} preferences...")
        by_template = load_runs_by_seed(experiment_dir, mtype, args.template)
        loaded_data[mtype] = by_template

        if not by_template:
            print(f"  No {mtype} runs found")
            continue

        n_runs = sum(len(seeds) for seeds in by_template.values())
        print(f"  Found {len(by_template)} templates, {n_runs} total runs")

        multi_seed_templates = {k: v for k, v in by_template.items() if len(v) >= 2}
        if not multi_seed_templates:
            print(f"  No templates with 2+ seeds found")
            continue

        template_suffix = f"_{args.template}" if args.template else ""
        output_path = output_dir / f"plot_{date_str}_seed_sensitivity_{mtype}{template_suffix}.png"
        title = f"Seed Stability: {args.experiment_id} ({mtype})"

        summary = plot_seed_sensitivity(
            multi_seed_templates, output_path, title, args.method
        )

        if summary:
            all_summaries[mtype] = summary
            print(f"  Mean cross-seed correlation: {summary['mean_correlation']:.3f} ({summary['n_pairs']} pairs)")
            print(f"  Range: [{summary['min_correlation']:.3f}, {summary['max_correlation']:.3f}]")

        # Origin-based analysis
        if args.by_origin:
            origin_output_path = output_dir / f"plot_{date_str}_seed_sensitivity_{mtype}_by_origin{template_suffix}.png"
            origin_title = f"Seed Stability by Origin: {args.experiment_id} ({mtype})"
            origin_summary = plot_seed_sensitivity_by_origin(
                multi_seed_templates, origin_output_path, origin_title, args.method
            )
            if origin_summary:
                all_summaries[f"{mtype}_by_origin"] = origin_summary
                print(f"  By origin:")
                for origin, stats in origin_summary.get("by_origin", {}).items():
                    print(f"    {origin}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['n_pairs']})")

    # Cross-type analysis: stated vs revealed correlation stability
    if args.cross_type and "stated" in loaded_data and "revealed" in loaded_data:
        stated_data = loaded_data["stated"]
        revealed_data = loaded_data["revealed"]

        if stated_data and revealed_data:
            print(f"\nAnalyzing stated-revealed correlation stability...")

            # Get the revealed data (should be single template with multiple seeds)
            revealed_template = list(revealed_data.keys())[0]
            revealed_by_seed = revealed_data[revealed_template]

            template_suffix = f"_{args.template}" if args.template else ""
            output_path = output_dir / f"plot_{date_str}_seed_sensitivity_cross_type{template_suffix}.png"
            title = f"Stated-Revealed Correlation: {args.experiment_id}"

            summary = plot_cross_type_sensitivity(
                stated_data, revealed_by_seed, output_path, title, args.method
            )

            if summary:
                all_summaries["cross_type"] = summary
                print(f"  Mean stated-revealed correlation: {summary['mean_correlation']:.3f} ({summary['n_pairs']} pairs)")
                print(f"  Range: [{summary['min_correlation']:.3f}, {summary['max_correlation']:.3f}]")

    if all_summaries:
        summary_path = output_dir / f"seed_sensitivity_{args.experiment_id}.yaml"
        with open(summary_path, "w") as f:
            yaml.dump(all_summaries, f, default_flow_style=False, sort_keys=False)
        print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
