"""Plot content-orthogonal probe comparison results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/probes/content_orthogonal_comparison")
ASSETS_DIR = Path("docs/logs/assets/content_orthogonal")


def plot_decomposition() -> None:
    """Stacked bar: content-predictable vs content-orthogonal vs unexplained."""
    raw = json.loads((RESULTS_DIR / "comparison_results_raw.json").read_text())
    resid = json.loads((RESULTS_DIR / "comparison_results.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title, score_label in [
        (axes[0], raw, "Raw Scores", "Raw Thurstonian scores"),
        (axes[1], resid, "Topic-Residualized Scores", "Topic-residualized scores"),
    ]:
        layers = [r["layer"] for r in data]
        standard = [r["standard_r2"] for r in data]
        content_orth = [r["content_orth_r2"] for r in data]
        content_only = data[0].get("content_only_r2", 0.112)

        x = np.arange(len(layers))
        width = 0.25

        bars1 = ax.bar(x - width, standard, width, label="Standard probe", color="#3498db", alpha=0.85, edgecolor="black")
        bars2 = ax.bar(x, content_orth, width, label="Content-orthogonal probe", color="#e74c3c", alpha=0.85, edgecolor="black")
        bars3 = ax.bar(x + width, [content_only] * len(layers), width, label="Content-only baseline", color="#95a5a6", alpha=0.85, edgecolor="black")

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
        ax.set_ylabel("cv R²", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("Preference Probe Signal Decomposition", fontsize=14, fontweight="bold")
    plt.tight_layout()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = ASSETS_DIR / "plot_021026_probe_decomposition.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {plot_path}")


def plot_variance_decomposition() -> None:
    """Pie-style stacked bar showing where the variance goes."""
    raw = json.loads((RESULTS_DIR / "comparison_results_raw.json").read_text())

    fig, ax = plt.subplots(figsize=(8, 6))

    layers = [r["layer"] for r in raw]
    standard = [r["standard_r2"] for r in raw]
    content_orth = [r["content_orth_r2"] for r in raw]
    content_only = raw[0]["content_only_r2"]

    # Decompose: content-shared + content-orthogonal + unexplained
    content_shared = [s - co for s, co in zip(standard, content_orth)]
    unexplained = [1.0 - s for s in standard]

    x = np.arange(len(layers))
    width = 0.5

    ax.bar(x, content_orth, width, label="Content-orthogonal signal", color="#e74c3c", alpha=0.85)
    ax.bar(x, content_shared, width, bottom=content_orth, label="Content-shared signal", color="#f39c12", alpha=0.85)
    ax.bar(x, unexplained, width, bottom=standard, label="Unexplained", color="#ecf0f1", alpha=0.85, edgecolor="#bdc3c7")

    # Annotations
    for i in range(len(layers)):
        # Content-orthogonal
        ax.text(x[i], content_orth[i] / 2, f"{content_orth[i]:.3f}", ha="center", va="center", fontsize=10, fontweight="bold")
        # Content-shared
        ax.text(x[i], content_orth[i] + content_shared[i] / 2, f"{content_shared[i]:.3f}", ha="center", va="center", fontsize=10, fontweight="bold")

    ax.axhline(y=content_only, color="#95a5a6", linestyle="--", linewidth=1.5, label=f"Content-only baseline (R²={content_only:.3f})")

    ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax.set_ylabel("R² (cumulative)", fontsize=11, fontweight="bold")
    ax.set_title("Variance Decomposition of Preference Signal", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plot_path = ASSETS_DIR / "plot_021026_variance_decomposition.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {plot_path}")


if __name__ == "__main__":
    plot_decomposition()
    plot_variance_decomposition()
