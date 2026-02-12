"""Visualize Ridge probe ablation sweep results.

Reads summary.json from the sweep and generates:
1. Heatmap: layers × conditions (8 conditions = 4 demean × 2 standardize)
2. Line plot: R² by layer, one line per condition
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SUMMARY_PATH = Path("results/probes/ablation_sweep/summary.json")
OUTPUT_DIR = Path("results/probes/ablation_sweep")


def load_results() -> list[dict]:
    with open(SUMMARY_PATH) as f:
        data = json.load(f)
    return data["results"]


def condition_label(row: dict) -> str:
    scale = "scaled" if row["standardize"] else "raw"
    return f"{row['demean']} / {scale}"


def plot_heatmap(results: list[dict]) -> None:
    layers = sorted(set(r["layer"] for r in results))
    conditions = []
    for r in sorted(results, key=lambda x: (x["demean"], not x["standardize"])):
        label = condition_label(r)
        if label not in conditions:
            conditions.append(label)

    matrix = np.zeros((len(conditions), len(layers)))
    for r in results:
        ci = conditions.index(condition_label(r))
        li = layers.index(r["layer"])
        matrix[ci, li] = r["cv_r2_mean"]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Condition (demean / scaling)")
    ax.set_title("Ridge Probe CV R² — Ablation Sweep")

    for ci in range(len(conditions)):
        for li in range(len(layers)):
            val = matrix[ci, li]
            color = "white" if val > 0.6 else "black"
            ax.text(li, ci, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="CV R²")
    plt.tight_layout()

    date_str = datetime.now().strftime("%m%d%y")
    path = OUTPUT_DIR / f"plot_{date_str}_ablation_heatmap.png"
    fig.savefig(path, dpi=150)
    print(f"Saved heatmap: {path}")
    plt.close(fig)


def plot_lines(results: list[dict]) -> None:
    layers = sorted(set(r["layer"] for r in results))
    conditions = []
    for r in sorted(results, key=lambda x: (x["demean"], not x["standardize"])):
        label = condition_label(r)
        if label not in conditions:
            conditions.append(label)

    # Color by demean condition, linestyle by scaling
    demean_colors = {}
    palette = plt.cm.tab10.colors
    for i, demean in enumerate(sorted(set(r["demean"] for r in results))):
        demean_colors[demean] = palette[i]

    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in conditions:
        cond_results = [r for r in results if condition_label(r) == cond]
        cond_results.sort(key=lambda x: x["layer"])
        demean = cond_results[0]["demean"]
        scaled = cond_results[0]["standardize"]
        xs = [r["layer"] for r in cond_results]
        ys = [r["cv_r2_mean"] for r in cond_results]
        errs = [r["cv_r2_std"] for r in cond_results]
        ax.errorbar(
            xs, ys, yerr=errs,
            marker="o" if scaled else "x",
            linestyle="-" if scaled else "--",
            color=demean_colors[demean],
            label=cond, capsize=3,
            linewidth=2 if scaled else 1.2,
            markersize=7,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("CV R²")
    ax.set_title("Ridge Probe CV R² by Layer and Condition")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylim(-0.5, 1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    date_str = datetime.now().strftime("%m%d%y")
    path = OUTPUT_DIR / f"plot_{date_str}_ablation_lines.png"
    fig.savefig(path, dpi=150)
    print(f"Saved line plot: {path}")
    plt.close(fig)


def main() -> None:
    results = load_results()
    print(f"Loaded {len(results)} results from {SUMMARY_PATH}")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demean", nargs="*", help="Filter to these demean conditions (e.g. none topic)")
    args = parser.parse_args()

    if args.demean is not None:
        results = [r for r in results if r["demean"] in args.demean]
        print(f"Filtered to {len(results)} results (demean: {args.demean})")

    plot_heatmap(results)
    plot_lines(results)


if __name__ == "__main__":
    main()
