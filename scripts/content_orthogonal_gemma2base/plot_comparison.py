"""Plot content-orthogonal probe comparison: sentence-transformer vs Gemma-2 base.

Usage:
  python scripts/content_orthogonal_gemma2base/plot_comparison.py [--results path/to/results.json]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("results/probes/content_orthogonal_gemma2base/comparison_results.json")
ASSETS_DIR = Path("docs/logs/assets/content_orthogonal_gemma2base")


def plot_comparison(results_path: Path) -> None:
    data = json.loads(results_path.read_text())
    results = data["results"]
    baselines = data["content_baselines"]

    layers = [r["layer"] for r in results]
    standard = [r["standard_r2"] for r in results]

    has_st = "st_co_r2" in results[0]
    has_g2 = "gemma2_co_r2" in results[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: R² comparison
    ax = axes[0]
    x = np.arange(len(layers))
    width = 0.2
    offset = 0

    ax.bar(x + offset, standard, width, label="Standard probe", color="#3498db", alpha=0.85, edgecolor="black")
    offset += width

    if has_st:
        st_co = [r["st_co_r2"] for r in results]
        ax.bar(x + offset, st_co, width, label="Content-orth (ST 384d)", color="#e74c3c", alpha=0.85, edgecolor="black")
        offset += width

    if has_g2:
        g2_co = [r["gemma2_co_r2"] for r in results]
        ax.bar(x + offset, g2_co, width, label="Content-orth (Gemma-2 3584d)", color="#9b59b6", alpha=0.85, edgecolor="black")
        offset += width

    # Add value labels
    for bars in ax.containers:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax.set_ylabel("cv R²", fontsize=11, fontweight="bold")
    ax.set_title("Standard vs Content-Orthogonal Probes", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Plot 2: % Retained comparison
    ax = axes[1]
    x = np.arange(len(layers))
    width = 0.3

    if has_st:
        st_pct = [r["st_co_r2"] / r["standard_r2"] * 100 for r in results]
        bars1 = ax.bar(x - width / 2, st_pct, width, label="ST (384d)", color="#e74c3c", alpha=0.85, edgecolor="black")
        for bar, pct in zip(bars1, st_pct):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    if has_g2:
        g2_pct = [max(r["gemma2_co_r2"] / r["standard_r2"] * 100, 0) for r in results]
        bars2 = ax.bar(x + width / 2, g2_pct, width, label="Gemma-2 9B base (3584d)", color="#9b59b6", alpha=0.85, edgecolor="black")
        for bar, pct in zip(bars2, g2_pct):
            label = f"{pct:.1f}%" if pct > 0.1 else "~0%"
            ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.5, label,
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax.set_ylabel("% of Standard Probe R² Retained", fontsize=11, fontweight="bold")
    ax.set_title("Content-Orthogonal Signal Retention", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylim(0, 35)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("Content Encoder Comparison: Sentence Transformer vs Gemma-2 9B Base", fontsize=13, fontweight="bold")
    plt.tight_layout()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = ASSETS_DIR / "plot_021126_encoder_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {plot_path}")


def plot_content_r2(results_path: Path) -> None:
    """Plot content→activation R² for both encoders."""
    data = json.loads(results_path.read_text())
    results = data["results"]

    has_st = "st_content_r2_cv" in results[0]
    has_g2 = "gemma2_content_r2_cv" in results[0]

    if not (has_st and has_g2):
        print("Need both encoders for content R² comparison plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    layers = [r["layer"] for r in results]
    x = np.arange(len(layers))
    width = 0.2

    st_train = [r["st_content_r2_train"] for r in results]
    st_cv = [r["st_content_r2_cv"] for r in results]
    g2_train = [r["gemma2_content_r2_train"] for r in results]
    g2_cv = [r["gemma2_content_r2_cv"] for r in results]

    ax.bar(x - 1.5 * width, st_train, width, label="ST train R²", color="#e74c3c", alpha=0.5, edgecolor="black")
    ax.bar(x - 0.5 * width, st_cv, width, label="ST cv R²", color="#e74c3c", alpha=0.85, edgecolor="black")
    ax.bar(x + 0.5 * width, g2_train, width, label="Gemma-2 train R²", color="#9b59b6", alpha=0.5, edgecolor="black")
    ax.bar(x + 1.5 * width, g2_cv, width, label="Gemma-2 cv R²", color="#9b59b6", alpha=0.85, edgecolor="black")

    for bars in ax.containers:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold", rotation=45)

    ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax.set_ylabel("R²", fontsize=11, fontweight="bold")
    ax.set_title("Content → Activation R² (train vs CV)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plot_path = ASSETS_DIR / "plot_021126_content_r2_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    plot_comparison(args.results)
    plot_content_r2(args.results)
