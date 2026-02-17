"""Plot pair selection oracle results from Experiment 2."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("experiments/probe_science/bt_scaling/experiment2_results.json")
OUTPUT_PATH = Path("experiments/probe_science/bt_scaling/assets/plot_021626_pair_overlap.png")


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    iterations = [d["target_iteration"] for d in data]
    pair_overlap = [d["pair_overlap_pct"] for d in data]
    task_coverage = [d["task_coverage_overlap_pct"] for d in data]
    rank_corr = [d["rank_correlation"] for d in data]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(iterations))
    width = 0.3

    bars1 = ax1.bar(x - width / 2, pair_overlap, width, label="Pair Overlap %", color="#4878CF", alpha=0.85)
    bars2 = ax1.bar(x + width / 2, task_coverage, width, label="Task Coverage Overlap %", color="#6ACC65", alpha=0.85)

    ax1.axhline(y=80, color="gray", linestyle="--", linewidth=1, label="80% Threshold")

    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Overlap (%)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(iterations)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.plot(x, rank_corr, color="#D65F5F", marker="o", linewidth=2, markersize=7, label="Rank Correlation")
    ax2.set_ylabel("Rank Correlation", fontsize=12, color="#D65F5F")
    ax2.tick_params(axis="y", labelcolor="#D65F5F")
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title("BT vs Thurstonian: Pair Selection Divergence", fontsize=13, fontweight="bold")
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
