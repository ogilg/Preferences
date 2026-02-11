"""Plot fair BT vs Ridge comparison results."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path(__file__).resolve().parents[2] / "experiments/bt_fair_comparison/results.json"
ASSETS_DIR = Path(__file__).resolve().parents[2] / "experiments/bt_fair_comparison/assets"
LOG_ASSETS_DIR = Path(__file__).resolve().parents[2] / "docs/logs/assets/bt_fair_comparison"


def load_results(path: Path = RESULTS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_per_fold_comparison(results: dict, layer: int = 31, save_dir: Path = ASSETS_DIR):
    """Bar chart: BT vs Ridge per fold + mean, for a given layer."""
    folds = results["folds"]
    fold_labels = [f"Fold {fr['fold']}" for fr in folds]

    bt_accs = [fr["layers"][str(layer)]["bt_test_acc"] for fr in folds]
    ridge_accs = [fr["layers"][str(layer)]["ridge_test_acc"] for fr in folds]
    thurst_accs = [fr["layers"][str(layer)]["thurstonian_test_acc"] for fr in folds]

    x = np.arange(len(folds) + 1)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    # Per-fold bars
    ax.bar(x[:-1] - width, bt_accs, width, label="BT", color="#2196F3", alpha=0.8)
    ax.bar(x[:-1], ridge_accs, width, label="Ridge", color="#FF9800", alpha=0.8)
    ax.bar(x[:-1] + width, thurst_accs, width, label="Thurstonian μ", color="#4CAF50", alpha=0.8)

    # Mean bars
    ax.bar(x[-1] - width, np.mean(bt_accs), width, color="#2196F3", alpha=0.5, edgecolor="#2196F3", linewidth=2)
    ax.bar(x[-1], np.mean(ridge_accs), width, color="#FF9800", alpha=0.5, edgecolor="#FF9800", linewidth=2)
    ax.bar(x[-1] + width, np.mean(thurst_accs), width, color="#4CAF50", alpha=0.5, edgecolor="#4CAF50", linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels + ["Mean"])
    ax.set_ylabel("Held-out Pairwise Accuracy")
    ax.set_title(f"Fair BT vs Ridge Comparison (Layer {layer})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bt_accs + [np.mean(bt_accs)], ridge_accs + [np.mean(ridge_accs)], thurst_accs + [np.mean(thurst_accs)]]:
        for i, val in enumerate(bars):
            offset = -width if bars is (bt_accs + [np.mean(bt_accs)]) else (0 if bars is (ridge_accs + [np.mean(ridge_accs)]) else width)
            ax.text(i + offset, val + 0.002, f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%m%d%y")
    filename = f"plot_{date_str}_bt_ridge_per_fold_L{layer}.png"
    fig.savefig(save_dir / filename, dpi=150)
    print(f"Saved: {save_dir / filename}")

    # Also save to log assets
    LOG_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LOG_ASSETS_DIR / filename, dpi=150)
    print(f"Saved: {LOG_ASSETS_DIR / filename}")

    plt.close(fig)
    return filename


def plot_across_layers(results: dict, save_dir: Path = ASSETS_DIR):
    """Line chart: BT vs Ridge mean accuracy across layers."""
    folds = results["folds"]
    layers = sorted(int(l) for l in folds[0]["layers"].keys())

    bt_means = []
    ridge_means = []
    thurst_means = []
    bt_stds = []
    ridge_stds = []

    for layer in layers:
        bt_accs = [fr["layers"][str(layer)]["bt_test_acc"] for fr in folds]
        ridge_accs = [fr["layers"][str(layer)]["ridge_test_acc"] for fr in folds]
        thurst_accs = [fr["layers"][str(layer)]["thurstonian_test_acc"] for fr in folds]
        bt_means.append(np.mean(bt_accs))
        ridge_means.append(np.mean(ridge_accs))
        thurst_means.append(np.mean(thurst_accs))
        bt_stds.append(np.std(bt_accs))
        ridge_stds.append(np.std(ridge_accs))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(layers, bt_means, yerr=bt_stds, marker="o", label="BT", capsize=5)
    ax.errorbar(layers, ridge_means, yerr=ridge_stds, marker="s", label="Ridge", capsize=5)
    ax.plot(layers, thurst_means, marker="^", label="Thurstonian μ", linestyle="--", alpha=0.7)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Held-out Pairwise Accuracy (mean ± std)")
    ax.set_title("BT vs Ridge Across Layers (Task-level 5-fold CV)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%m%d%y")
    filename = f"plot_{date_str}_bt_ridge_across_layers.png"
    fig.savefig(save_dir / filename, dpi=150)
    print(f"Saved: {save_dir / filename}")

    LOG_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LOG_ASSETS_DIR / filename, dpi=150)
    print(f"Saved: {LOG_ASSETS_DIR / filename}")

    plt.close(fig)
    return filename


def print_summary_table(results: dict):
    """Print markdown-formatted results tables."""
    folds = results["folds"]
    layers = sorted(int(l) for l in folds[0]["layers"].keys())

    for layer in layers:
        print(f"\n### Layer {layer}")
        print(f"\n| Fold | BT acc | Ridge acc | Thurstonian acc | Test pairs | Test measurements |")
        print(f"|------|--------|-----------|-----------------|------------|-------------------|")

        bt_accs = []
        ridge_accs = []
        thurst_accs = []
        for fr in folds:
            l = fr["layers"][str(layer)]
            bt_accs.append(l["bt_test_acc"])
            ridge_accs.append(l["ridge_test_acc"])
            thurst_accs.append(l["thurstonian_test_acc"])
            print(f"| {fr['fold']} | {l['bt_test_acc']:.4f} | {l['ridge_test_acc']:.4f} | {l['thurstonian_test_acc']:.4f} | {fr['n_test_pairs']} | {fr['n_test_measurements']} |")

        print(f"| **Mean** | **{np.mean(bt_accs):.4f}** | **{np.mean(ridge_accs):.4f}** | **{np.mean(thurst_accs):.4f}** | | |")
        print(f"| **Std** | **{np.std(bt_accs):.4f}** | **{np.std(ridge_accs):.4f}** | **{np.std(thurst_accs):.4f}** | | |")

    # Overall summary
    print("\n### Summary (all layers)")
    print(f"\n| Layer | BT (mean ± std) | Ridge (mean ± std) | Thurstonian (mean ± std) |")
    print(f"|-------|-----------------|--------------------|-----------------------|")
    for layer in layers:
        bt_accs = [fr["layers"][str(layer)]["bt_test_acc"] for fr in folds]
        ridge_accs = [fr["layers"][str(layer)]["ridge_test_acc"] for fr in folds]
        thurst_accs = [fr["layers"][str(layer)]["thurstonian_test_acc"] for fr in folds]
        print(f"| {layer} | {np.mean(bt_accs):.4f} ± {np.std(bt_accs):.4f} | {np.mean(ridge_accs):.4f} ± {np.std(ridge_accs):.4f} | {np.mean(thurst_accs):.4f} ± {np.std(thurst_accs):.4f} |")


if __name__ == "__main__":
    results = load_results()
    print_summary_table(results)
    f1 = plot_per_fold_comparison(results, layer=31)
    f2 = plot_across_layers(results)
    print(f"\nPlots: {f1}, {f2}")
