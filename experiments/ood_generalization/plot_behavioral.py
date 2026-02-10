"""Plot behavioral measurement results for OOD generalization experiment."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path("experiments/ood_generalization")
ASSETS_DIR = Path("docs/logs/assets/ood_generalization")


def load_results(filename: str) -> list[dict]:
    with open(EXP_DIR / "results" / filename) as f:
        return json.load(f)


def plot_behavioral_deltas(results: list[dict], output_path: Path):
    results_sorted = sorted(results, key=lambda r: r["delta"])

    labels = [r["prompt_id"] for r in results_sorted]
    deltas = [r["delta"] for r in results_sorted]
    directions = [r["direction"] for r in results_sorted]

    colors = ["#d63031" if d == "negative" else "#00b894" for d in directions]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, deltas, color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Behavioral Delta: P(choose target | manip) - P(choose target | baseline)")
    ax.set_title("System Prompt Manipulation Effects on Pairwise Choice")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlim(-1.1, 1.1)

    neg_patch = plt.Rectangle((0, 0), 1, 1, fc="#d63031", label="Negative")
    pos_patch = plt.Rectangle((0, 0), 1, 1, fc="#00b894", label="Positive")
    ax.legend(handles=[neg_patch, pos_patch], title="Direction", loc="lower right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def plot_baseline_vs_manipulation(results: list[dict], output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 8))

    for r in results:
        color = "#d63031" if r["direction"] == "negative" else "#00b894"
        ax.scatter(r["baseline_rate"], r["manipulation_rate"], c=color, s=60, zorder=3)
        ax.annotate(r["prompt_id"], (r["baseline_rate"], r["manipulation_rate"]),
                    fontsize=6, ha="left", va="bottom", rotation=20)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="No effect")
    ax.set_xlabel("P(choose target | baseline)")
    ax.set_ylabel("P(choose target | manipulation)")
    ax.set_title("Baseline vs Manipulated Choice Rates")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "behavioral_all_20.json"
    results = load_results(filename)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    plot_behavioral_deltas(
        results,
        ASSETS_DIR / "plot_021026_behavioral_deltas.png"
    )
    plot_baseline_vs_manipulation(
        results,
        ASSETS_DIR / "plot_021026_baseline_vs_manipulation.png"
    )


if __name__ == "__main__":
    main()
