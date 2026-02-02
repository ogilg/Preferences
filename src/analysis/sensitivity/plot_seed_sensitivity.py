"""Unified seed sensitivity analysis for all measurement types.

Analyzes how stable preference measurements are across different random seeds.
Computes correlations between runs with different seeds but same template/model.

Usage:
    python -m src.analysis.sensitivity.plot_seed_sensitivity --experiment-id stability_v1
    python -m src.analysis.sensitivity.plot_seed_sensitivity --experiment-id stability_v1 --type stated
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.analysis.correlation.utils import utility_vector_correlation, ReliabilityMethod, compute_informative_correlation
from src.measurement_storage import EXPERIMENTS_DIR
from src.measurement_storage.run_parsing import extract_model_from_run_dir


MeasurementType = Literal["stated", "revealed"]
OUTPUT_DIR = Path(__file__).parent / "plots"

def load_run_config(run_dir: Path) -> dict | None:
    """Load config.yaml from a run directory."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_model_name(model_str: str) -> str:
    """Extract short model name from full model string (e.g., 'qwen/qwen3-32b' -> 'qwen3-32b')."""
    if "/" in model_str:
        return model_str.split("/")[-1]
    return model_str


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


@dataclass
class RunData:
    values: np.ndarray
    task_ids: list[str]
    origins: list[str]
    model: str
    template: str


def load_runs(
    experiment_dir: Path,
    measurement_type: MeasurementType,
    template_filter: str | None = None,
) -> list[tuple[int, RunData]]:
    """Load all runs with their seed and metadata."""
    if measurement_type == "stated":
        subdirs = ["post_task_stated"]
        load_fn = load_run_scores
    else:
        subdirs = ["post_task_revealed", "post_task_active_learning"]
        load_fn = load_run_utilities

    runs: list[tuple[int, RunData]] = []

    for subdir in subdirs:
        results_dir = experiment_dir / subdir
        if not results_dir.exists():
            continue

        for run_dir in sorted(results_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            config = load_run_config(run_dir)
            if not config:
                continue

            template = config.get("template_name", "unknown")
            # Prefer model from dir name (captures variants like qwen3-32b-nothink)
            model = extract_model_from_run_dir(run_dir.name)
            if model is None:
                model = extract_model_name(config.get("model", "unknown"))
            seed = config.get("rating_seed", 0)

            if template_filter and template_filter not in template:
                continue

            values, task_ids, origins = load_fn(run_dir)
            if len(values) == 0:
                continue

            runs.append((seed, RunData(values, task_ids, origins, model, template)))

    return runs


def group_runs_by_key(
    runs: list[tuple[int, RunData]],
    key_fn,
) -> dict[str, dict[int, RunData]]:
    """Group runs by a key function, then by seed."""
    grouped: dict[str, dict[int, RunData]] = defaultdict(dict)
    for seed, run in runs:
        key = key_fn(run)
        grouped[key][seed] = run
    return dict(grouped)


def compute_cross_seed_correlations(
    seed_data: dict[int, RunData],
    method: ReliabilityMethod = "pearson",
    origin_filter: str | None = None,
) -> list[tuple[int, int, float, float | None]]:
    """Compute reliability metrics between all seed pairs, optionally filtered by origin.

    Returns list of (seed_a, seed_b, correlation, discrimination_rate).
    discrimination_rate is only populated for method="informative".
    """
    correlations = []
    seeds = sorted(seed_data.keys())

    for seed_a, seed_b in combinations(seeds, 2):
        run_a, run_b = seed_data[seed_a], seed_data[seed_b]
        vals_a, ids_a, origins_a = run_a.values, run_a.task_ids, run_a.origins
        vals_b, ids_b, origins_b = run_b.values, run_b.task_ids, run_b.origins

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

        # For informative method, also compute discrimination rate
        disc_rate = None
        if method == "informative":
            # Need to align the vectors first
            common_ids = set(ids_a) & set(ids_b)
            if len(common_ids) < 10:
                continue
            id_to_idx_a = {tid: i for i, tid in enumerate(ids_a)}
            id_to_idx_b = {tid: i for i, tid in enumerate(ids_b)}
            common_list = sorted(common_ids)
            aligned_a = np.array([vals_a[id_to_idx_a[tid]] for tid in common_list])
            aligned_b = np.array([vals_b[id_to_idx_b[tid]] for tid in common_list])
            corr, disc_rate = compute_informative_correlation(aligned_a, aligned_b)
        else:
            corr = utility_vector_correlation(vals_a, ids_a, vals_b, ids_b, method)

        if not np.isnan(corr):
            correlations.append((seed_a, seed_b, corr, disc_rate))

    return correlations


def compute_correlations_by_grouping(
    grouped: dict[str, dict[int, RunData]],
    method: ReliabilityMethod = "pearson",
    origin_filter: str | None = None,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Compute cross-seed correlations for each group.

    Returns (correlations_dict, discrimination_rates_dict).
    discrimination_rates_dict is only populated for method="informative".
    """
    result: dict[str, list[float]] = {}
    disc_rates: dict[str, list[float]] = {}
    for key, seed_data in grouped.items():
        if len(seed_data) < 2:
            continue
        corrs = compute_cross_seed_correlations(seed_data, method, origin_filter)
        if corrs:
            result[key] = [c[2] for c in corrs]
            if method == "informative":
                disc_rates[key] = [c[3] for c in corrs if c[3] is not None]
    return result, disc_rates


def compute_correlations_by_origin(
    runs: list[tuple[int, RunData]],
    method: ReliabilityMethod = "pearson",
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Compute cross-seed correlations grouped by origin dataset.

    Returns (correlations_dict, discrimination_rates_dict).
    """
    origins = ["WILDCHAT", "ALPACA", "MATH", "BAILBENCH"]
    # Group by (model, template) to get seed pairs, then compute per-origin correlations
    by_run = group_runs_by_key(runs, lambda r: f"{r.template}_{r.model}")

    origin_correlations: dict[str, list[float]] = {o: [] for o in origins}
    origin_disc_rates: dict[str, list[float]] = {o: [] for o in origins}
    for seed_data in by_run.values():
        if len(seed_data) < 2:
            continue
        for origin in origins:
            corrs = compute_cross_seed_correlations(seed_data, method, origin_filter=origin)
            if corrs:
                origin_correlations[origin].extend([c[2] for c in corrs])
                if method == "informative":
                    origin_disc_rates[origin].extend([c[3] for c in corrs if c[3] is not None])

    return (
        {k: v for k, v in origin_correlations.items() if v},
        {k: v for k, v in origin_disc_rates.items() if v},
    )


def _plot_bar_panel(
    ax: plt.Axes,
    correlations: dict[str, list[float]],
    title: str,
    colors: dict[str, str] | str = "steelblue",
    show_n: bool = False,
    disc_rates: dict[str, list[float]] | None = None,
):
    """Plot a single bar chart panel."""
    if not correlations:
        ax.set_visible(False)
        return

    names = sorted(correlations.keys())
    means = [np.mean(correlations[n]) for n in names]
    stds = [np.std(correlations[n]) for n in names]
    counts = [len(correlations[n]) for n in names]

    x = np.arange(len(names))
    if isinstance(colors, dict):
        bar_colors = [colors.get(n, "steelblue") for n in names]
    else:
        bar_colors = colors

    bars = ax.bar(x, means, yerr=stds, capsize=3, color=bar_colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)

    all_corrs = [c for corrs in correlations.values() for c in corrs]
    overall_mean = np.mean(all_corrs)
    ax.axhline(overall_mean, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.text(0.97, overall_mean + 0.02, f"{overall_mean:.2f}",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9, color="red")

    # Show annotations
    for i, (bar, count, name) in enumerate(zip(bars, counts, names)):
        label_parts = []
        if show_n:
            label_parts.append(f"n={count}")
        if disc_rates and name in disc_rates and disc_rates[name]:
            mean_disc = np.mean(disc_rates[name])
            label_parts.append(f"d={mean_disc:.0%}")

        if label_parts:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] + 0.02,
                    "\n".join(label_parts), ha="center", va="bottom", fontsize=7)

    ax.set_title(title, fontsize=10)


def aggregate_correlations_by(
    run_correlations: dict[str, list[float]],
    runs: list[tuple[int, RunData]],
    key_fn,
) -> dict[str, list[float]]:
    """Aggregate correlations from run_correlations into groups defined by key_fn."""
    # Build mapping from run key to aggregation key
    run_to_agg: dict[str, str] = {}
    for _, run in runs:
        run_key = f"{run.template}_{run.model}"
        agg_key = key_fn(run)
        run_to_agg[run_key] = agg_key

    # Aggregate
    result: dict[str, list[float]] = defaultdict(list)
    for run_key, corrs in run_correlations.items():
        agg_key = run_to_agg[run_key]
        result[agg_key].extend(corrs)

    return dict(result)


def plot_seed_sensitivity_grid(
    runs: list[tuple[int, RunData]],
    output_path: Path,
    experiment_id: str,
    method: ReliabilityMethod = "pearson",
) -> dict:
    """Create a 2x2 grid of seed sensitivity plots: overall, by origin, by model, by template."""
    # Group runs by (template, model) - the only valid grouping for seed comparison
    by_run = group_runs_by_key(runs, lambda r: f"{r.template}_{r.model}")

    # Compute correlations per run
    overall_corrs, overall_disc = compute_correlations_by_grouping(by_run, method)
    origin_corrs, origin_disc = compute_correlations_by_origin(runs, method)

    # Aggregate run correlations by model and template
    model_corrs = aggregate_correlations_by(overall_corrs, runs, lambda r: r.model)
    template_corrs = aggregate_correlations_by(overall_corrs, runs, lambda r: r.template)

    # Also aggregate discrimination rates if available
    model_disc = aggregate_correlations_by(overall_disc, runs, lambda r: r.model) if overall_disc else {}
    template_disc = aggregate_correlations_by(overall_disc, runs, lambda r: r.template) if overall_disc else {}

    if not overall_corrs:
        print("No cross-seed correlations found (need 2+ seeds per run)")
        return {}

    all_corrs = [c for corrs in overall_corrs.values() for c in corrs]
    n_pairs = len(all_corrs)
    mean_corr = np.mean(all_corrs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    origin_colors = {"WILDCHAT": "#4ECDC4", "ALPACA": "#FF6B6B", "MATH": "#45B7D1", "BAILBENCH": "#96CEB4"}

    _plot_bar_panel(axes[0, 0], overall_corrs, "By Run (template + model)", show_n=False, disc_rates=overall_disc)
    _plot_bar_panel(axes[0, 1], origin_corrs, "By Origin", colors=origin_colors, show_n=True, disc_rates=origin_disc)
    _plot_bar_panel(axes[1, 0], model_corrs, "By Model", show_n=True, disc_rates=model_disc)
    _plot_bar_panel(axes[1, 1], template_corrs, "By Template", show_n=True, disc_rates=template_disc)

    # Set ylabel based on method
    method_labels = {
        "pearson": "Cross-Seed Correlation (Pearson r)",
        "spearman": "Cross-Seed Correlation (Spearman ρ)",
        "discrimination": "Discrimination Ratio (Var_between / Var_total)",
        "informative": "Correlation on Non-Modal Tasks",
    }
    ylabel = method_labels.get(method, "Reliability")

    for ax in axes.flat:
        ax.set_ylabel(ylabel, fontsize=9)

    metric_symbol = "DR" if method == "discrimination" else "r"
    fig.suptitle(f"Seed Stability: {experiment_id} ({n_pairs} seed pairs, mean {metric_symbol}={mean_corr:.2f})", fontsize=12)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

    return {
        "n_pairs": n_pairs,
        "mean_correlation": float(mean_corr),
        "std_correlation": float(np.std(all_corrs)),
        "min_correlation": float(np.min(all_corrs)),
        "max_correlation": float(np.max(all_corrs)),
        "by_run": {
            name: {"mean": float(np.mean(corrs)), "std": float(np.std(corrs)), "n_pairs": len(corrs)}
            for name, corrs in overall_corrs.items()
        },
        "by_origin": {
            name: {"mean": float(np.mean(corrs)), "std": float(np.std(corrs)), "n_pairs": len(corrs)}
            for name, corrs in origin_corrs.items()
        },
        "by_model": {
            name: {"mean": float(np.mean(corrs)), "std": float(np.std(corrs)), "n_pairs": len(corrs)}
            for name, corrs in model_corrs.items()
        },
        "by_template": {
            name: {"mean": float(np.mean(corrs)), "std": float(np.std(corrs)), "n_pairs": len(corrs)}
            for name, corrs in template_corrs.items()
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
        "--template",
        type=str,
        default=None,
        help="Filter to templates containing this string",
    )
    parser.add_argument(
        "--method",
        choices=["pearson", "spearman", "discrimination", "informative"],
        default="pearson",
        help="Reliability method: pearson, spearman, discrimination, or informative (default: pearson)",
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

    for mtype in types_to_analyze:
        print(f"\nAnalyzing {mtype} preferences...")
        runs = load_runs(experiment_dir, mtype, args.template)

        if not runs:
            print(f"  No {mtype} runs found")
            continue

        models = set(r.model for _, r in runs)
        templates = set(r.template for _, r in runs)
        print(f"  Found {len(runs)} runs ({len(models)} models, {len(templates)} templates)")

        template_suffix = f"_{args.template}" if args.template else ""
        safe_experiment_id = args.experiment_id.replace("/", "_")
        output_path = output_dir / f"plot_{date_str}_{safe_experiment_id}_{mtype}_{args.method}{template_suffix}.png"

        summary = plot_seed_sensitivity_grid(
            runs, output_path, args.experiment_id, args.method
        )

        if summary:
            all_summaries[mtype] = summary
            print(f"  Mean cross-seed correlation: {summary['mean_correlation']:.3f} ({summary['n_pairs']} pairs)")
            print(f"  Range: [{summary['min_correlation']:.3f}, {summary['max_correlation']:.3f}]")

    if all_summaries:
        # Sanitize experiment_id for filename (replace / with _)
        safe_experiment_id = args.experiment_id.replace("/", "_")
        summary_path = output_dir / f"seed_sensitivity_{safe_experiment_id}.yaml"
        with open(summary_path, "w") as f:
            yaml.dump(all_summaries, f, default_flow_style=False, sort_keys=False)
        print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
