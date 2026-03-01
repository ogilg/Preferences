import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = Path("experiments/ood_system_prompts/utility_fitting/on_target_results.json")
ASSETS_DIR = Path("experiments/ood_system_prompts/utility_fitting/assets")

with open(DATA_PATH) as f:
    data = json.load(f)

COLORS = {"pos": "#2ca89a", "neg": "#e87461"}


def clean_topic(topic: str) -> str:
    return topic.replace("_", " ").title()


# ---------- Plot 1: Cross-group pairwise accuracy ----------

def plot_cross_acc():
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    for ax, exp in zip(axes, ["exp1b", "exp1c"]):
        subset = [d for d in data if d["experiment"] == exp]
        topics = sorted(set(d["topic"] for d in subset))

        y_labels = []
        y_positions = []
        bar_colors = []
        bar_values = []

        pos_idx = 0
        for topic in topics:
            for polarity in ["pos", "neg"]:
                row = [d for d in subset if d["topic"] == topic and d["polarity"] == polarity][0]
                y_labels.append(f"{clean_topic(topic)} ({polarity})")
                y_positions.append(pos_idx)
                bar_colors.append(COLORS[polarity])
                bar_values.append(row["cross_acc"])
                pos_idx += 1
            pos_idx += 0.3  # gap between topic groups

        y_positions = np.array(y_positions)
        ax.barh(y_positions, bar_values, color=bar_colors, height=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlim(0.5, 1.0)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Cross-group pairwise accuracy")
        ax.set_title(exp)
        ax.invert_yaxis()

    fig.suptitle("Cross-group pairwise accuracy: on-target vs off-target", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_022828_on_target_cross_acc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ASSETS_DIR / 'plot_022828_on_target_cross_acc.png'}")


# ---------- Plot 2: Summary 2x2 grid ----------

def plot_summary():
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    experiments = ["exp1b", "exp1c"]
    polarities = ["pos", "neg"]

    for row_idx, exp in enumerate(experiments):
        subset = [d for d in data if d["experiment"] == exp]

        # Left column: Pearson r (on vs off)
        ax_r = axes[row_idx, 0]
        for pol_idx, pol in enumerate(polarities):
            pol_data = [d for d in subset if d["polarity"] == pol]
            on_r_mean = np.mean([d["on_r"] for d in pol_data])
            off_r_mean = np.mean([d["off_r"] for d in pol_data])
            x = np.array([0, 1]) + pol_idx * 0.35
            ax_r.bar(x, [on_r_mean, off_r_mean], width=0.3, color=COLORS[pol], label=pol if row_idx == 0 else None)

        ax_r.set_xticks([0.175, 1.175])
        ax_r.set_xticklabels(["On-target r", "Off-target r"])
        ax_r.set_ylim(0, 1.0)
        ax_r.set_ylabel("Pearson r")
        ax_r.set_title(f"{exp} — Pearson r")

        # Right column: Pairwise accuracy (on, off, cross)
        ax_acc = axes[row_idx, 1]
        for pol_idx, pol in enumerate(polarities):
            pol_data = [d for d in subset if d["polarity"] == pol]
            on_acc_mean = np.mean([d["on_acc"] for d in pol_data])
            off_acc_mean = np.mean([d["off_acc"] for d in pol_data])
            cross_acc_mean = np.mean([d["cross_acc"] for d in pol_data])
            x = np.array([0, 1, 2]) + pol_idx * 0.35
            ax_acc.bar(x, [on_acc_mean, off_acc_mean, cross_acc_mean], width=0.3, color=COLORS[pol],
                       label=pol if row_idx == 0 else None)

        ax_acc.set_xticks([0.175, 1.175, 2.175])
        ax_acc.set_xticklabels(["On-target", "Off-target", "Cross-group"])
        ax_acc.set_ylim(0.0, 1.0)
        ax_acc.set_ylabel("Pairwise accuracy")
        ax_acc.set_title(f"{exp} — Pairwise accuracy")
        ax_acc.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Legend from first row
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=11)

    fig.suptitle("On-target utility fitting: summary", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(ASSETS_DIR / "plot_022828_on_target_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ASSETS_DIR / 'plot_022828_on_target_summary.png'}")


if __name__ == "__main__":
    plot_cross_acc()
    plot_summary()
