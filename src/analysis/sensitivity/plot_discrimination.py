"""Scatter plot of KL divergence vs ICC for seed sensitivity analysis.

Visualizes the trade-off between scale usage (KL from uniform) and
cross-seed consistency (ICC) for different templates/models/datasets.

Usage:
    python -m src.analysis.sensitivity.plot_discrimination --experiment-id multi_model_seed_sensitivity
"""

from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.analysis.sensitivity.plot_seed_sensitivity import (
    load_runs,
    group_runs_by_key,
    RunData,
    OUTPUT_DIR,
)
from src.measurement.storage import EXPERIMENTS_DIR


def kl_from_uniform(scores: list[float], n_bins: int | None = None) -> float:
    """KL divergence of score distribution from uniform over the observed scale.

    Automatically detects the scale range from min/max of rounded scores.
    """
    if not scores:
        return float("nan")

    rounded = [int(round(s)) for s in scores]
    min_val, max_val = min(rounded), max(rounded)

    if min_val == max_val:
        return float("nan")

    # Use actual range as bins
    bins = list(range(min_val, max_val + 1))
    n_bins = len(bins)

    counts = Counter(rounded)
    observed = np.array([counts.get(i, 0) for i in bins], dtype=float)
    if observed.sum() == 0:
        return float("nan")
    observed = observed / observed.sum()

    uniform = np.ones(n_bins) / n_bins

    eps = 1e-10
    observed = np.clip(observed, eps, 1)
    return float(np.sum(observed * np.log(observed / uniform)))


def compute_icc(task_ratings: dict[str, list[float]]) -> float:
    """Compute ICC: Var(between tasks) / (Var(between) + Var(within))."""
    tasks_with_multiple = {k: v for k, v in task_ratings.items() if len(v) >= 2}
    if len(tasks_with_multiple) < 2:
        return float("nan")

    task_means = [np.mean(ratings) for ratings in tasks_with_multiple.values()]
    within_vars = [np.var(ratings, ddof=1) for ratings in tasks_with_multiple.values()]

    var_between = np.var(task_means, ddof=1)
    var_within = np.mean(within_vars)
    var_total = var_between + var_within

    if var_total < 1e-10:
        return float("nan")

    return float(var_between / var_total)


def compute_metrics_for_group(
    seed_data: dict[int, RunData],
    origin_filter: str | None = None,
) -> dict:
    """Compute KL and ICC for a group of runs across seeds."""
    task_scores: dict[str, list[float]] = defaultdict(list)

    for seed, run in seed_data.items():
        for i, task_id in enumerate(run.task_ids):
            if origin_filter and run.origins[i] != origin_filter:
                continue
            task_scores[task_id].append(run.values[i])

    if not task_scores:
        return {}

    all_scores = [s for scores in task_scores.values() for s in scores]

    return {
        "kl": kl_from_uniform(all_scores),
        "icc": compute_icc(task_scores),
        "n_tasks": len(task_scores),
        "n_scores": len(all_scores),
    }


def compute_all_metrics(
    runs: list[tuple[int, RunData]],
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], dict[str, dict]]:
    """Compute metrics grouped by run, template, model, and origin."""
    by_run = group_runs_by_key(runs, lambda r: f"{r.template}_{r.model}")

    run_metrics = {}
    for key, seed_data in by_run.items():
        if len(seed_data) < 2:
            continue
        metrics = compute_metrics_for_group(seed_data)
        if metrics:
            run_metrics[key] = metrics

    by_template = group_runs_by_key(runs, lambda r: r.template)
    template_metrics = {}
    for key, seed_data in by_template.items():
        metrics = compute_metrics_for_group(seed_data)
        if metrics:
            template_metrics[key] = metrics

    by_model = group_runs_by_key(runs, lambda r: r.model)
    model_metrics = {}
    for key, seed_data in by_model.items():
        metrics = compute_metrics_for_group(seed_data)
        if metrics:
            model_metrics[key] = metrics

    origins = ["WILDCHAT", "ALPACA", "MATH", "BAILBENCH"]
    origin_metrics = {}
    for origin in origins:
        all_seed_data: dict[int, RunData] = {}
        for seed, run in runs:
            if seed not in all_seed_data:
                mask = [o == origin for o in run.origins]
                if any(mask):
                    filtered_values = run.values[mask]
                    filtered_ids = [tid for tid, m in zip(run.task_ids, mask) if m]
                    filtered_origins = [o for o, m in zip(run.origins, mask) if m]
                    all_seed_data[seed] = RunData(
                        filtered_values, filtered_ids, filtered_origins, run.model, run.template
                    )

        if len(all_seed_data) >= 2:
            metrics = compute_metrics_for_group(all_seed_data)
            if metrics:
                origin_metrics[origin] = metrics

    return run_metrics, template_metrics, model_metrics, origin_metrics


def plot_scatter_panel(
    ax: plt.Axes,
    metrics: dict[str, dict],
    title: str,
    show_labels: bool = True,
    marker_size: int = 120,
    x_max: float | None = None,
):
    """Plot a scatter panel with color gradient."""
    if not metrics:
        ax.set_visible(False)
        return

    names = list(metrics.keys())
    kls = [metrics[n]["kl"] for n in names]
    iccs = [metrics[n]["icc"] for n in names]

    valid = [(n, k, i) for n, k, i in zip(names, kls, iccs) if not (np.isnan(k) or np.isnan(i))]
    if not valid:
        ax.set_visible(False)
        return

    names, kls, iccs = zip(*valid)
    kls = np.array(kls)
    iccs = np.array(iccs)

    # Color by goodness (low KL + high ICC)
    kl_norm = (kls - kls.min()) / (kls.max() - kls.min() + 1e-10)
    icc_norm = (iccs - iccs.min()) / (iccs.max() - iccs.min() + 1e-10)
    goodness = (1 - kl_norm) * 0.5 + icc_norm * 0.5

    scatter = ax.scatter(
        kls, iccs, c=goodness, cmap="RdYlGn", s=marker_size, alpha=0.9,
        edgecolors="black", linewidths=0.5, vmin=0, vmax=1
    )

    if show_labels:
        for name, kl, icc in zip(names, kls, iccs):
            ax.annotate(
                name, (kl, icc),
                fontsize=11, fontweight="bold",
                xytext=(5, 5), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
            )

    ax.set_xlabel("KL from Uniform (lower = better)", fontsize=10)
    ax.set_ylabel("ICC (higher = better)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.05)
    ax.axvline(0.8, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(0.7, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.grid(True, alpha=0.3)


def plot_discrimination_grid(
    runs: list[tuple[int, RunData]],
    output_path: Path,
    experiment_id: str,
):
    """Create a 2x2 grid with table below."""
    run_metrics, template_metrics, model_metrics, origin_metrics = compute_all_metrics(runs)

    if not run_metrics:
        print("No metrics computed (need 2+ seeds per run)")
        return

    # Create figure: 2x2 grid on top, table below
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.25, wspace=0.25)

    ax_origin = fig.add_subplot(gs[0, 0])
    ax_model = fig.add_subplot(gs[0, 1])
    ax_run = fig.add_subplot(gs[1, 0])
    ax_template = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[2, :])

    # Find x-axis limit using 95th percentile to exclude outliers
    all_metrics = [run_metrics, template_metrics, model_metrics, origin_metrics]
    all_kls = []
    for metrics in all_metrics:
        for m in metrics.values():
            if not np.isnan(m["kl"]):
                all_kls.append(m["kl"])
    x_max = np.percentile(all_kls, 95) * 1.2 if all_kls else 1.5

    plot_scatter_panel(ax_origin, origin_metrics, "By Origin", x_max=x_max)
    plot_scatter_panel(ax_model, model_metrics, "By Model", x_max=x_max)
    plot_scatter_panel(ax_run, run_metrics, "By Run", show_labels=False, marker_size=80, x_max=x_max)
    plot_scatter_panel(ax_template, template_metrics, "By Template", x_max=x_max)

    # Create table for run metrics
    ax_table.axis("off")

    # Sort by goodness (low KL, high ICC)
    run_data = []
    for name, m in run_metrics.items():
        kl, icc = m["kl"], m["icc"]
        if not (np.isnan(kl) or np.isnan(icc)):
            goodness = (1 - kl / 1.5) * 0.5 + icc * 0.5
            run_data.append((name, kl, icc, m["n_tasks"], goodness))

    run_data.sort(key=lambda x: -x[4])

    cell_text = [[name[:45], f"{kl:.3f}", f"{icc:.3f}", str(n)] for name, kl, icc, n, _ in run_data]
    col_labels = ["Run (template_model)", "KL", "ICC", "Tasks"]

    table = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="left",
        colWidths=[0.55, 0.12, 0.12, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.4)

    for i, (_, _, _, _, goodness) in enumerate(run_data):
        color = plt.cm.RdYlGn(goodness)
        for j in range(4):
            table[(i + 1, j)].set_facecolor((*color[:3], 0.3))

    ax_table.set_title("Runs ranked by discrimination quality", fontsize=12, fontweight="bold", pad=10)

    fig.suptitle(f"Discrimination Analysis: {experiment_id}", fontsize=14, fontweight="bold", y=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot KL vs ICC scatter for discrimination analysis"
    )
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--type", choices=["stated", "revealed"], default="stated")
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    args = parser.parse_args()

    experiment_dir = EXPERIMENTS_DIR / args.experiment_id
    if not experiment_dir.exists():
        print(f"Experiment not found: {experiment_dir}")
        return

    output_dir = args.output_dir or OUTPUT_DIR
    date_str = datetime.now().strftime("%m%d%y")

    runs = load_runs(experiment_dir, args.type, args.template)
    if not runs:
        print(f"No {args.type} runs found")
        return

    models = set(r.model for _, r in runs)
    templates = set(r.template for _, r in runs)
    print(f"Found {len(runs)} runs ({len(models)} models, {len(templates)} templates)")

    safe_experiment_id = args.experiment_id.replace("/", "_")
    template_suffix = f"_{args.template}" if args.template else ""
    output_path = output_dir / f"plot_{date_str}_{safe_experiment_id}_discrimination{template_suffix}.png"

    plot_discrimination_grid(runs, output_path, args.experiment_id)


if __name__ == "__main__":
    main()
