"""Plot results from held-one-out probe evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.measurement_storage.base import find_project_root


def plot_hoo_results(
    results_json: Path,
    output_dir: Path | None = None,
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """Plot held-one-out validation results.

    Args:
        results_json: path to hoo_summary.json from run_held_one_out.py
        output_dir: optional output directory for plots (defaults to same as results_json)
        figsize: figure size (width, height) in inches
    """
    # Load results
    with open(results_json) as f:
        results = json.load(f)

    if not results["folds"]:
        print("No folds to plot")
        return

    # Determine output directory
    if output_dir is None:
        output_dir = results_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    eval_datasets = []
    r2_values = []
    r2_adj_values = []
    pearson_values = []
    n_samples = []

    for fold in results["folds"]:
        eval_datasets.append(fold["eval_dataset"].upper())

        # Aggregate metrics across probes
        r2_scores = [
            p["eval_metrics"]["r2"]
            for p in fold["probes"]
            if p["eval_metrics"]["r2"] is not None
        ]
        r2_adj_scores = [
            p["eval_metrics"].get("r2_adjusted")
            for p in fold["probes"]
            if p["eval_metrics"].get("r2_adjusted") is not None
        ]
        pearson_scores = [
            p["eval_metrics"]["pearson_r"]
            for p in fold["probes"]
            if p["eval_metrics"]["pearson_r"] is not None
        ]
        samples = [p["eval_metrics"]["n_samples"] for p in fold["probes"]]

        r2_values.append(np.median(r2_scores) if r2_scores else None)
        r2_adj_values.append(np.median(r2_adj_scores) if r2_adj_scores else None)
        pearson_values.append(np.median(pearson_scores) if pearson_scores else None)
        n_samples.append(np.mean(samples) if samples else 0)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Held-One-Out Probe Evaluation", fontsize=14, fontweight="bold")

    # Plot 1: R² by eval dataset
    ax = axes[0, 0]
    colors = ["#2ecc71" if r2 > 0 else "#e74c3c" for r2 in r2_values]
    bars = ax.bar(eval_datasets, r2_values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_ylabel("Median R²", fontsize=11, fontweight="bold")
    ax.set_xlabel("Held-Out Dataset", fontsize=11, fontweight="bold")
    ax.set_title("Standard R² Score by Evaluation Dataset", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for bar, val in zip(bars, r2_values):
        if val is not None:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

    # Plot 2: Mean-adjusted R² by eval dataset
    ax = axes[0, 1]
    colors = ["#3498db" if r2 > 0 else "#e67e22" for r2 in r2_adj_values]
    bars = ax.bar(eval_datasets, r2_adj_values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_ylabel("Median R² (Mean-Adjusted)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Held-Out Dataset", fontsize=11, fontweight="bold")
    ax.set_title("Mean-Adjusted R² (corrects for dataset mean differences)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for bar, val in zip(bars, r2_adj_values):
        if val is not None:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

    # Plot 3: Pearson r by eval dataset
    ax = axes[1, 0]
    colors = ["#3498db" if abs(p) < 0.5 else "#9b59b6" for p in pearson_values]
    bars = ax.bar(eval_datasets, pearson_values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_ylabel("Median Pearson r", fontsize=11, fontweight="bold")
    ax.set_xlabel("Held-Out Dataset", fontsize=11, fontweight="bold")
    ax.set_title("Pearson Correlation by Evaluation Dataset", fontsize=12, fontweight="bold")
    ax.set_ylim(-1, 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for bar, val in zip(bars, pearson_values):
        if val is not None:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

    # Plot 4: Improvement from mean adjustment
    ax = axes[1, 1]
    improvements = []
    for r2, r2_adj in zip(r2_values, r2_adj_values):
        if r2 is not None and r2_adj is not None:
            improvements.append(r2_adj - r2)
        else:
            improvements.append(0)

    colors_imp = ["#2ecc71" if imp > 0 else "#e74c3c" for imp in improvements]
    bars = ax.bar(eval_datasets, improvements, color=colors_imp, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_ylabel("R² Improvement", fontsize=11, fontweight="bold")
    ax.set_xlabel("Held-Out Dataset", fontsize=11, fontweight="bold")
    ax.set_title("R² Improvement from Mean Adjustment", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for bar, val in zip(bars, improvements):
        if val != 0:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()

    # Save plot
    timestamp = results["created_at"].split("T")[0].replace("-", "")
    plot_path = output_dir / f"plot_{timestamp}_hoo_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")
    plt.close()

    # Create detailed heatmap
    plot_hoo_heatmap(results, output_dir)


def plot_hoo_heatmap(results: dict, output_dir: Path) -> None:
    """Plot heatmap of R² by probe layer and eval dataset.

    Args:
        results: results dict from hoo_summary.json
        output_dir: output directory for plot
    """
    if not results["folds"]:
        return

    # Build matrix: rows=layers, cols=eval_datasets, values=R²
    layers_set = set()
    datasets = []

    for fold in results["folds"]:
        datasets.append(fold["eval_dataset"].upper())
        for probe in fold["probes"]:
            layers_set.add(probe["layer"])

    layers = sorted(layers_set)
    datasets_sorted = sorted(datasets)

    # Initialize matrix
    r2_matrix = np.full((len(layers), len(datasets_sorted)), np.nan)

    # Fill matrix
    for fold in results["folds"]:
        dataset_idx = datasets_sorted.index(fold["eval_dataset"].upper())
        for probe in fold["probes"]:
            layer_idx = layers.index(probe["layer"])
            r2_val = probe["eval_metrics"]["r2"]
            if r2_val is not None:
                r2_matrix[layer_idx, dataset_idx] = r2_val

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine color range
    vmin = np.nanmin(r2_matrix)
    vmax = np.nanmax(r2_matrix)
    vmax = max(vmax, 0)  # Include 0 in range

    im = ax.imshow(r2_matrix, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(datasets_sorted)))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_xticklabels(datasets_sorted)
    ax.set_yticklabels(layers)

    ax.set_xlabel("Held-Out Dataset", fontsize=11, fontweight="bold")
    ax.set_ylabel("Layer", fontsize=11, fontweight="bold")
    ax.set_title("Probe R² by Layer and Evaluation Dataset", fontsize=12, fontweight="bold")

    # Add text annotations
    for i in range(len(layers)):
        for j in range(len(datasets_sorted)):
            val = r2_matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("R² Score", fontsize=10, fontweight="bold")

    plt.tight_layout()

    timestamp = results["created_at"].split("T")[0].replace("-", "")
    plot_path = output_dir / f"plot_{timestamp}_hoo_heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to {plot_path}")
    plt.close()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Plot held-one-out validation results")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to hoo_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for plots (defaults to results directory)",
    )

    args = parser.parse_args()

    if not args.results.exists():
        raise FileNotFoundError(f"Results file not found: {args.results}")

    plot_hoo_results(args.results, args.output_dir)


if __name__ == "__main__":
    main()
