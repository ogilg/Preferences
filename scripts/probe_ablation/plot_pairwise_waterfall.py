"""Waterfall chart showing how much pairwise choice variance each stage captures."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("results/probes/ablation_sweep")

# From fair comparison (task-level 5-fold CV, L31)
CHANCE = 0.50
PROBE_ACC = 0.746
THURSTONIAN_ACC = 0.866

# Segments (above chance)
probe_above_chance = PROBE_ACC - CHANCE
thurstonian_gap = THURSTONIAN_ACC - PROBE_ACC
noise = 1.0 - THURSTONIAN_ACC


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 3))

    segments = [
        ("Chance", CHANCE, "#d9d9d9"),
        ("Ridge probe", probe_above_chance, "#c0392b"),
        ("Thurstonian\n(not captured\nby probe)", thurstonian_gap, "#e8a0a0"),
        ("Measurement\nnoise", noise, "#f0f0f0"),
    ]

    left = 0
    for label, width, color in segments:
        bar = ax.barh(0, width, left=left, color=color, edgecolor="white", height=0.5)
        # Label inside if wide enough
        cx = left + width / 2
        if width > 0.06:
            ax.text(cx, 0, f"{width:.1%}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="black")
        left += width

    # Bracket annotations above the bar
    y_top = 0.35
    arrow_props = dict(arrowstyle="-", color="black", lw=1)

    # Probe accuracy bracket
    ax.annotate("", xy=(0, y_top), xytext=(PROBE_ACC, y_top), arrowprops=arrow_props)
    ax.plot([0, 0], [y_top - 0.03, y_top + 0.03], color="black", lw=1)
    ax.plot([PROBE_ACC, PROBE_ACC], [y_top - 0.03, y_top + 0.03], color="black", lw=1)
    ax.text(PROBE_ACC / 2, y_top + 0.06, f"Probe: {PROBE_ACC:.1%}", ha="center", fontsize=10)

    # Thurstonian ceiling bracket
    ax.annotate("", xy=(0, y_top + 0.18), xytext=(THURSTONIAN_ACC, y_top + 0.18), arrowprops=arrow_props)
    ax.plot([0, 0], [y_top + 0.15, y_top + 0.21], color="black", lw=1)
    ax.plot([THURSTONIAN_ACC, THURSTONIAN_ACC], [y_top + 0.15, y_top + 0.21], color="black", lw=1)
    ax.text(THURSTONIAN_ACC / 2, y_top + 0.24, f"Thurstonian ceiling: {THURSTONIAN_ACC:.1%}",
            ha="center", fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 0.85)
    ax.set_xlabel("Held-out pairwise accuracy")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks([])
    ax.set_title("How much of pairwise choice does the probe explain?")

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in segments]
    labels = [l.replace("\n", " ") for l, _, _ in segments]
    ax.legend(handles, labels, loc="lower right", fontsize=8, ncol=2)

    plt.tight_layout()

    date_str = datetime.now().strftime("%m%d%y")
    path = OUTPUT_DIR / f"plot_{date_str}_pairwise_waterfall.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
