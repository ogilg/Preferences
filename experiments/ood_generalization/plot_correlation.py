"""Plot behavioral delta vs probe delta correlation for OOD generalization."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

EXP_DIR = Path("experiments/ood_generalization")
ASSETS_DIR = Path("docs/logs/assets/ood_generalization")


def load_results(filename: str = "probe_behavioral_comparison.json") -> list[dict]:
    with open(EXP_DIR / "results" / filename) as f:
        return json.load(f)


def plot_correlation(results: list[dict], layer: int, output_path: Path):
    beh_deltas = np.array([r["behavioral_delta"] for r in results])
    probe_deltas = np.array([r[f"probe_delta_L{layer}"] for r in results])

    pearson_r, pearson_p = stats.pearsonr(beh_deltas, probe_deltas)
    spearman_r, _ = stats.spearmanr(beh_deltas, probe_deltas)

    fig, ax = plt.subplots(figsize=(8, 7))

    categories = sorted(set(r["target_category"] for r in results))
    cat_colors = {
        "math": "#e17055",
        "coding": "#0984e3",
        "fiction": "#6c5ce7",
        "knowledge_qa": "#00b894",
        "content_generation": "#fdcb6e",
        "harmful_request": "#d63031",
    }

    for r in results:
        color = cat_colors.get(r["target_category"], "gray")
        marker = "^" if r["direction"] == "positive" else "v"
        ax.scatter(
            r["behavioral_delta"],
            r[f"probe_delta_L{layer}"],
            c=color, marker=marker, s=80, zorder=3, edgecolors="white", linewidths=0.5,
        )

    # Regression line
    slope, intercept, _, _, _ = stats.linregress(beh_deltas, probe_deltas)
    x_line = np.linspace(beh_deltas.min() - 0.05, beh_deltas.max() + 0.05, 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.4, linewidth=1)

    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Behavioral Delta: P(choose target | manip) - P(choose target | baseline)")
    ax.set_ylabel(f"Probe Delta (Layer {layer}): probe(manip) - probe(baseline)")
    ax.set_title(f"OOD Generalization: Behavioral vs Probe Delta (Layer {layer})\n"
                 f"Pearson r={pearson_r:.3f} (p={pearson_p:.1e}), "
                 f"Spearman ρ={spearman_r:.3f}")

    # Legend for categories
    for cat in categories:
        color = cat_colors.get(cat, "gray")
        ax.scatter([], [], c=color, s=60, label=cat)
    ax.scatter([], [], marker="^", c="gray", s=60, label="positive")
    ax.scatter([], [], marker="v", c="gray", s=60, label="negative")
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


def plot_all_layers(results: list[dict], output_path: Path):
    layers = [31, 43, 55]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    cat_colors = {
        "math": "#e17055",
        "coding": "#0984e3",
        "fiction": "#6c5ce7",
        "knowledge_qa": "#00b894",
        "content_generation": "#fdcb6e",
        "harmful_request": "#d63031",
    }

    beh_deltas = np.array([r["behavioral_delta"] for r in results])

    for ax, layer in zip(axes, layers):
        probe_deltas = np.array([r[f"probe_delta_L{layer}"] for r in results])
        pearson_r, pearson_p = stats.pearsonr(beh_deltas, probe_deltas)

        for r in results:
            color = cat_colors.get(r["target_category"], "gray")
            marker = "^" if r["direction"] == "positive" else "v"
            ax.scatter(r["behavioral_delta"], r[f"probe_delta_L{layer}"],
                       c=color, marker=marker, s=60, zorder=3,
                       edgecolors="white", linewidths=0.5)

        slope, intercept, _, _, _ = stats.linregress(beh_deltas, probe_deltas)
        x_line = np.linspace(-1.1, 0.8, 100)
        ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.4)
        ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Behavioral Delta")
        ax.set_ylabel(f"Probe Delta (L{layer})")
        ax.set_title(f"Layer {layer}: r={pearson_r:.3f} (p={pearson_p:.1e})")

    # Shared legend
    for cat in sorted(cat_colors):
        axes[0].scatter([], [], c=cat_colors[cat], s=40, label=cat)
    axes[0].legend(fontsize=7, loc="upper left")

    plt.suptitle("OOD Generalization: Behavioral vs Probe Deltas Across Layers", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


def main():
    results = load_results()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    plot_correlation(results, 31, ASSETS_DIR / "plot_021026_correlation_L31.png")
    plot_all_layers(results, ASSETS_DIR / "plot_021026_correlation_all_layers.png")


if __name__ == "__main__":
    main()
