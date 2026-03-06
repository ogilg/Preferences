"""Plot ICL transfer results: dot plot showing P(choose target) across conditions."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("experiments/icl_transfer/assets")

# Data from experiments
AXES = [
    ("math > harmful", "math_over_harmful_request"),
    ("harmful > math", "harmful_request_over_math"),
    ("fiction > harmful", "fiction_over_harmful_request"),
    ("harmful > fiction", "harmful_request_over_fiction"),
    ("math > fiction", "math_over_fiction"),
    ("fiction > math", "fiction_over_math"),
]

BASELINES = {
    "math_over_harmful_request": 0.778,
    "harmful_request_over_math": 0.222,
    "fiction_over_harmful_request": 0.662,
    "harmful_request_over_fiction": 0.338,
    "math_over_fiction": 0.636,
    "fiction_over_math": 0.364,
}


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    by_axis = {}
    for c in data["conditions"]:
        axis = c["axis"]
        if axis not in by_axis:
            by_axis[axis] = []
        by_axis[axis].append(c["p_preferred"])
    return {k: (np.mean(v), np.std(v)) for k, v in by_axis.items()}


def main():
    rev_k1 = load_results(OUTPUT_DIR / "phase1_revealed_k1_results.json")
    rev_k3 = load_results(OUTPUT_DIR / "phase1_revealed_k3_results.json")
    stated_k1 = load_results(OUTPUT_DIR / "phase2_stated_k1_results.json")

    conditions = [
        ("Baseline", None, "#888888", "s", 8),
        ("Revealed K=1", rev_k1, "#4C72B0", "o", 7),
        ("Revealed K=3", rev_k3, "#4C72B0", "D", 7),
        ("Stated K=1", stated_k1, "#DD8452", "o", 7),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    y_positions = {}  # axis_key -> y position
    y_labels_list = []  # (y, label) pairs
    y = 0

    # Group axes by topic pair — top to bottom
    groups = [
        ("math vs harmful", [0, 1]),
        ("fiction vs harmful", [2, 3]),
        ("math vs fiction", [4, 5]),
    ]

    for gi, (group_name, indices) in enumerate(groups):
        for idx in indices:
            label, axis_key = AXES[idx]
            y_positions[axis_key] = y
            y_labels_list.append((y, label))

            baseline = BASELINES[axis_key]

            # Shade region between baseline and chance to show "natural preference"
            # Light connector lines from baseline to each condition
            for cond_name, cond_data, color, marker, ms in conditions:
                if cond_data is None:
                    ax.plot(baseline, y, marker="s", color="#888888", ms=8,
                            zorder=5, markeredgecolor="white", markeredgewidth=0.5)
                else:
                    mean, std = cond_data[axis_key]
                    # Thin line from baseline to condition
                    ax.plot([baseline, mean], [y, y], color=color, alpha=0.3,
                            linewidth=1, zorder=2)
                    ax.plot(mean, y, marker=marker, color=color, ms=ms,
                            zorder=5, markeredgecolor="white", markeredgewidth=0.5)
                    # Error bar (std across 5 ICL context pairs)
                    ax.errorbar(mean, y, xerr=std, color=color, alpha=0.4,
                                linewidth=1.5, capsize=0, zorder=3)

            y += 1
        y += 0.5  # gap between groups

    # Chance line
    ax.axvline(0.5, color="black", linestyle=":", alpha=0.3, linewidth=1, zorder=1)
    ax.text(0.5, y - 0.3, "chance", ha="center", va="bottom", fontsize=8,
            color="#666666", style="italic")

    # Arrow annotation for the backfire
    backfire_y = y_positions["math_over_fiction"]
    ax.annotate("backfire", xy=(0.276, backfire_y), xytext=(0.15, backfire_y - 0.6),
                fontsize=8, color="#4C72B0", alpha=0.7,
                arrowprops=dict(arrowstyle="->", color="#4C72B0", alpha=0.5))

    yticks = [yl[0] for yl in y_labels_list]
    ylabels = [yl[1] for yl in y_labels_list]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.invert_yaxis()
    ax.set_xlabel("P(choose target topic)", fontsize=11)
    ax.set_title("ICL preference manipulation: stated vs revealed", fontsize=12, fontweight="bold")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#888888",
                    markersize=8, label="Baseline"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0",
                    markersize=7, label="Revealed K=1"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#4C72B0",
                    markersize=7, label="Revealed K=3"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#DD8452",
                    markersize=7, label="Stated K=1"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor="#cccccc")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Group labels on the right
    for gi, (group_name, indices) in enumerate(groups):
        group_y = np.mean([y_positions[AXES[idx][1]] for idx in indices])
        ax.text(1.05, group_y, group_name, ha="left", va="center", fontsize=9,
                color="#666666", transform=ax.get_yaxis_transform())

    plt.tight_layout()
    out_path = OUTPUT_DIR / "plot_030626_icl_transfer_dot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
