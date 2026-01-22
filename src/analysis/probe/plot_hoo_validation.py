"""Plot in-distribution validation results for HOO probes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_validation_results(
    validation_json: Path,
    hoo_json: Path,
    manifest_json: Path,
    output_dir: Path | None = None,
    figsize: tuple[int, int] = (16, 7),
) -> None:
    """Plot train, CV, and test R² for HOO probes.

    Args:
        validation_json: path to hoo_validation_results.json
        hoo_json: path to hoo_evaluation_summary.json
        manifest_json: path to manifest.json
        output_dir: optional output directory for plots (defaults to same as validation_json)
        figsize: figure size (width, height) in inches
    """
    # Load results
    with open(validation_json) as f:
        validation = json.load(f)
    with open(hoo_json) as f:
        hoo = json.load(f)
    with open(manifest_json) as f:
        manifest = json.load(f)

    if not validation["folds"]:
        print("No validation folds to plot")
        return

    # Determine output directory
    if output_dir is None:
        output_dir = validation_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    probe_ids = []
    probe_labels = []
    train_r2_values = []
    cv_r2_values = []
    cv_r2_std_values = []
    test_r2_values = []
    test_adj_r2_values = []
    pearson_values = []

    # Map probes by ID
    probe_map = {p["id"]: p for p in manifest["probes"]}
    val_map = {str(f["probe_id"]): f for f in validation["folds"]}

    # Determine number of train/test datasets from first probe
    first_fold = validation["folds"][0] if validation["folds"] else {}
    n_train_datasets = len(first_fold.get("train_datasets", []))
    n_test_datasets = len(first_fold.get("eval_datasets") or [first_fold.get("eval_dataset", "")])

    for fold in validation["folds"]:
        probe_id = fold["probe_id"]
        probe_ids.append(probe_id)

        # Create label showing train/test datasets
        train_datasets = fold.get("train_datasets", [])
        # Handle both old singular and new plural eval dataset formats
        eval_datasets = fold.get("eval_datasets") or [fold.get("eval_dataset", "")]
        label = f"{probe_id}\n(train: {','.join(train_datasets[:2])}{'...' if len(train_datasets) > 2 else ''})\n(test: {','.join(eval_datasets[:2])}{'...' if len(eval_datasets) > 2 else ''})"
        probe_labels.append(label)

        train_r2_values.append(fold["metrics"]["r2"])
        pearson_values.append(fold["metrics"]["pearson_r"])

        # Get CV R² from manifest
        probe = probe_map[probe_id]
        cv_r2_values.append(probe["cv_r2_mean"])
        cv_r2_std_values.append(probe["cv_r2_std"])

        # Get test R² from HOO results
        test_r2 = None
        test_adj_r2 = None
        for hoo_fold in hoo["folds"]:
            for p in hoo_fold["probes"]:
                if p["id"] == probe_id:
                    test_r2 = p["eval_metrics"]["r2"]
                    test_adj_r2 = p["eval_metrics"].get("r2_adjusted")
                    break
        test_r2_values.append(test_r2)
        test_adj_r2_values.append(test_adj_r2)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("HOO Probe Generalization: Train → Test Performance", fontsize=14, fontweight="bold")

    x = np.arange(len(probe_ids))
    width = 0.2

    # Plot 1: R² across pipeline
    ax = ax1
    bars1 = ax.bar(x - width, train_r2_values, width, label=f"Train R²\n({n_train_datasets} datasets)", color="#2ecc71", alpha=0.8, edgecolor="black")
    bars2 = ax.bar(x, cv_r2_values, width, label=f"CV R²\n({n_train_datasets} datasets)", color="#3498db", alpha=0.8, edgecolor="black")
    bars3 = ax.bar(x + width, test_adj_r2_values, width, label=f"Test R² (Adjusted)\n({n_test_datasets} held-out dataset{'s' if n_test_datasets > 1 else ''})", color="#e74c3c", alpha=0.8, edgecolor="black")

    ax.set_xlabel("Probe Configuration", fontsize=11, fontweight="bold")
    ax.set_ylabel("R² Score", fontsize=11, fontweight="bold")
    ax.set_title(f"R² Across Train ({n_train_datasets} datasets) → Test ({n_test_datasets} held-out) Pipeline", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(probe_labels, fontsize=9)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=9,
            )

    # Plot 2: Pearson correlation (directional agreement)
    ax = ax2
    bars = ax.bar(probe_ids, pearson_values, color="#9b59b6", alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_xlabel("Probe ID", fontsize=11, fontweight="bold")
    ax.set_ylabel("Pearson r (on held-out test set)", fontsize=11, fontweight="bold")
    ax.set_title("Directional Agreement: Test Set Rankings", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels
    for bar, val in zip(bars, pearson_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    # Save plot
    timestamp = validation["created_at"].split("T")[0].replace("-", "")
    plot_path = output_dir / f"plot_{timestamp}_hoo_validation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")
    plt.close()

    # Create metrics summary table as text
    print("\nGeneralization Summary:")
    print("=" * 80)
    print(f"{'Probe':<8} {'Train R²':<12} {'CV R²':<12} {'Test R² (Adj.)':<16} {'Pearson r':<12} {'Gap':<10}")
    print("-" * 80)

    for i, probe_id in enumerate(probe_ids):
        gap = train_r2_values[i] - test_adj_r2_values[i]
        print(
            f"{probe_id:<8} {train_r2_values[i]:<12.4f} {cv_r2_values[i]:<12.4f} "
            f"{test_adj_r2_values[i]:<16.4f} {pearson_values[i]:<12.4f} {gap:<10.4f}"
        )

    avg_train = np.mean(train_r2_values)
    avg_cv = np.mean(cv_r2_values)
    avg_test = np.mean(test_adj_r2_values)
    avg_pearson = np.mean(pearson_values)
    avg_gap = avg_train - avg_test

    print("-" * 80)
    print(
        f"{'MEAN':<8} {avg_train:<12.4f} {avg_cv:<12.4f} "
        f"{avg_test:<16.4f} {avg_pearson:<12.4f} {avg_gap:<10.4f}"
    )
    print("=" * 80)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Plot HOO probe validation results")
    parser.add_argument(
        "--validation",
        type=Path,
        required=True,
        help="Path to hoo_validation_results.json",
    )
    parser.add_argument(
        "--hoo",
        type=Path,
        required=True,
        help="Path to hoo_evaluation_summary.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for plots (defaults to validation directory)",
    )

    args = parser.parse_args()

    if not args.validation.exists():
        raise FileNotFoundError(f"Validation file not found: {args.validation}")
    if not args.hoo.exists():
        raise FileNotFoundError(f"HOO file not found: {args.hoo}")
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest file not found: {args.manifest}")

    plot_validation_results(args.validation, args.hoo, args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
