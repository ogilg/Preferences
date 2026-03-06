import json

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "experiments/ood_pt/analysis_results_full.json"
ASSETS_DIR = "experiments/ood_pt/assets"

EXPERIMENTS = ["exp1a", "exp1b", "exp1c", "exp1d"]
EXP_LABELS = {"exp1a": "1a: Category", "exp1b": "1b: Hidden", "exp1c": "1c: Crossed", "exp1d": "1d: Competing"}
CONDITIONS = ["PT probe / PT acts", "PT probe / IT acts", "IT probe / PT acts"]

# IT probe / IT acts results from the original OOD report
IT_IT_R = {"exp1a": 0.61, "exp1b": 0.65, "exp1c": 0.66, "exp1d": 0.78}


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def plot_scatter_all(data: dict) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))

    for row, exp in enumerate(EXPERIMENTS):
        for col, cond in enumerate(CONDITIONS):
            ax = axes[row, col]
            d = data[exp][cond]
            beh = np.array(d["behavioral_deltas"])
            probe = np.array(d["probe_deltas"])
            r = d["pearson_r"]
            sign = d["sign_agreement"]

            ax.scatter(beh, probe, alpha=0.3, s=8, color="steelblue")

            # Reference lines
            ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")

            # Stats text box
            text = f"r={r:.2f}\nsign={sign * 100:.1f}%"
            ax.text(
                0.05, 0.95, text,
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
            )

            # Labels
            if row == 3:
                ax.set_xlabel("Behavioral delta")
            if col == 0:
                ax.set_ylabel("Probe delta")

            # Column titles on top row
            if row == 0:
                ax.set_title(cond, fontsize=11, fontweight="bold")

            # Row labels on left column
            if col == 0:
                ax.annotate(
                    EXP_LABELS[exp], xy=(0, 0.5), xytext=(-0.35, 0.5),
                    xycoords="axes fraction", textcoords="axes fraction",
                    fontsize=12, fontweight="bold", rotation=90,
                    ha="center", va="center",
                )

    fig.suptitle("OOD Pre-trained Probe: Scatter Plots", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.05, 0, 1, 0.96])
    out = f"{ASSETS_DIR}/plot_030626_scatter_all.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_comparison_bar(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    bar_labels = ["IT probe / IT acts", "IT probe / PT acts", "PT probe / IT acts", "PT probe / PT acts"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    n_groups = len(EXPERIMENTS)
    n_bars = len(bar_labels)
    bar_width = 0.18
    x = np.arange(n_groups)

    for i, (label, color) in enumerate(zip(bar_labels, colors)):
        vals = []
        for exp in EXPERIMENTS:
            if label == "IT probe / IT acts":
                vals.append(IT_IT_R[exp])
            else:
                vals.append(data[exp][label]["pearson_r"])

        offset = (i - (n_bars - 1) / 2) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=label, color=color, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([EXP_LABELS[e] for e in EXPERIMENTS], fontsize=11)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title("OOD Correlation: Full 2x2 Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = f"{ASSETS_DIR}/plot_030626_comparison_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    data = load_data()
    plot_scatter_all(data)
    plot_comparison_bar(data)
