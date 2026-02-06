"""Horizontal bar chart of tasks ranked by mu, with sigma error bars, colored by dataset.

Shows every Nth task along the ranking to keep the plot readable.

Usage:
    python -m src.analysis.active_learning.plot_ranked_tasks --experiment-id gemma3_revealed_v1
    python -m src.analysis.active_learning.plot_ranked_tasks --experiment-id gemma3_revealed_v1 --every 5
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.active_learning.utils import (
    DATASET_COLORS,
    load_ranked_tasks,
    plot_output_path,
)


def plot_ranked_tasks(tasks: list[dict], output_path, title: str, every: int = 3) -> None:
    tasks_sorted = sorted(tasks, key=lambda t: t["mu"])
    sampled = tasks_sorted[::every]

    mus = np.array([t["mu"] for t in sampled])
    sigmas = np.array([t["sigma"] for t in sampled])
    datasets = [t["dataset"] for t in sampled]
    colors = [DATASET_COLORS[ds] for ds in datasets]

    fig, ax = plt.subplots(figsize=(10, max(6, len(sampled) * 0.18)))

    y_pos = np.arange(len(sampled))
    ax.barh(y_pos, mus, xerr=sigmas, color=colors, alpha=0.8, height=0.8,
            capsize=1.5, error_kw={"ecolor": "gray", "elinewidth": 0.5})

    ax.set_yticks(y_pos)
    ax.set_yticklabels([t["task_id"] for t in sampled], fontsize=6)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Utility (μ) ± σ")
    ax.set_title(title)

    # Legend
    all_datasets = sorted(set(datasets))
    handles = [plt.Rectangle((0, 0), 1, 1, color=DATASET_COLORS[ds], alpha=0.8) for ds in all_datasets]
    ax.legend(handles, all_datasets, loc="lower right", fontsize=8)

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path} (showing {len(sampled)}/{len(tasks_sorted)} tasks, every {every}th)")


def main():
    parser = argparse.ArgumentParser(description="Plot ranked tasks with error bars")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--every", type=int, default=3, help="Show every Nth task")
    args = parser.parse_args()

    tasks = load_ranked_tasks(args.experiment_id, args.run_name)
    print(f"Loaded {len(tasks)} tasks")

    display_name = f"{args.experiment_id} ({args.run_name})" if args.run_name else args.experiment_id
    output = plot_output_path(args.experiment_id, "ranked_tasks", args.run_name)
    plot_ranked_tasks(tasks, output, f"Task Preferences Ranked by Utility\n{display_name}", every=args.every)


if __name__ == "__main__":
    main()
