"""Sigma analysis: mean sigma by dataset, refusal-sigma correlation, tasks ranked by sigma.

Usage:
    python -m src.analysis.active_learning.plot_sigma_vs_mu --experiment-id gemma3_revealed_v1
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from src.analysis.active_learning.utils import (
    DATASET_COLORS,
    load_ranked_tasks,
    plot_output_path,
)


def plot_sigma_analysis(tasks: list[dict], output_path, title: str) -> None:
    has_refusal = "refusal_rate" in tasks[0]

    if has_refusal:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={"width_ratios": [1, 1, 2]})
    else:
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 2]})
        ax2 = None

    # Panel 1: Mean sigma by dataset
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for t in tasks:
        by_dataset[t["dataset"]].append(t["sigma"])

    stats = [(ds, np.mean(sigs), np.std(sigs) / np.sqrt(len(sigs)), len(sigs))
             for ds, sigs in by_dataset.items()]
    stats.sort(key=lambda x: x[1], reverse=True)

    ds_names = [s[0] for s in stats]
    means = [s[1] for s in stats]
    sems = [s[2] for s in stats]
    ns = [s[3] for s in stats]
    colors = [DATASET_COLORS[ds] for ds in ds_names]

    bars = ax1.bar(ds_names, means, yerr=sems, capsize=5, color=colors, alpha=0.8, edgecolor="black")
    for bar, n in zip(bars, ns):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"n={n}", ha="center", va="bottom", fontsize=9)

    ax1.set_ylabel("Mean Uncertainty (σ)")
    ax1.set_title("Mean σ by Dataset")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Refusal rate vs sigma scatter
    if ax2 is not None:
        all_sigmas = np.array([t["sigma"] for t in tasks])
        all_refusals = np.array([t["refusal_rate"] for t in tasks])

        for ds in sorted(set(t["dataset"] for t in tasks)):
            ds_tasks = [t for t in tasks if t["dataset"] == ds]
            ax2.scatter(
                [t["refusal_rate"] * 100 for t in ds_tasks],
                [t["sigma"] for t in ds_tasks],
                c=DATASET_COLORS[ds], label=ds, alpha=0.7, s=40,
                edgecolors="white", linewidth=0.5,
            )

        r, p = pearsonr(all_refusals, all_sigmas)
        ax2.set_xlabel("Refusal Rate (%)")
        ax2.set_ylabel("Uncertainty (σ)")
        ax2.set_title(f"Refusal Rate vs σ\nr={r:.3f}, p={p:.2g}")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)

        # Trend line
        z = np.polyfit(all_refusals * 100, all_sigmas, 1)
        x_line = np.linspace(0, all_refusals.max() * 100, 50)
        ax2.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.4, linewidth=1)

    # Panel 3: Tasks ranked by sigma (horizontal bars)
    tasks_by_sigma = sorted(tasks, key=lambda t: t["sigma"])
    sigmas = np.array([t["sigma"] for t in tasks_by_sigma])
    datasets = [t["dataset"] for t in tasks_by_sigma]
    bar_colors = [DATASET_COLORS[ds] for ds in datasets]

    y_pos = np.arange(len(tasks_by_sigma))
    ax3.barh(y_pos, sigmas, color=bar_colors, alpha=0.8, height=0.8)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([t["task_id"] for t in tasks_by_sigma], fontsize=4.5)
    ax3.set_xlabel("Uncertainty (σ)")
    ax3.set_title("Tasks Ranked by σ")

    all_datasets_sorted = sorted(set(datasets))
    handles = [plt.Rectangle((0, 0), 1, 1, color=DATASET_COLORS[ds], alpha=0.8) for ds in all_datasets_sorted]
    ax3.legend(handles, all_datasets_sorted, loc="lower right", fontsize=7)
    ax3.grid(axis="x", alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")

    # Print stats
    for ds, mean, sem, n in stats:
        print(f"  {ds:<12} σ = {mean:.2f} ± {sem:.2f} (n={n})")


def main():
    parser = argparse.ArgumentParser(description="Plot sigma analysis")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    tasks = load_ranked_tasks(args.experiment_id, args.run_name)
    print(f"Loaded {len(tasks)} tasks")

    display_name = f"{args.experiment_id} ({args.run_name})" if args.run_name else args.experiment_id
    output = plot_output_path(args.experiment_id, "sigma_analysis", args.run_name)
    plot_sigma_analysis(tasks, output, f"Uncertainty Analysis\n{display_name}")


if __name__ == "__main__":
    main()
