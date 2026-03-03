"""Bar chart: probe r vs baseline utilities r for 4.2 (one-sided conflict + opposing prompts).

All tasks. Two grouped bars per condition, two panels (Pearson r, pairwise accuracy).
Style matches plot_exp1b_overview.py.

Usage:
    python docs/lw_post/plots/plot_section4_utility_bars.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

plt.style.use("seaborn-v0_8-whitegrid")

ASSETS_DIR = Path(__file__).parent.parent / "assets"

from scripts.utility_fitting.plot_harmful_comparison import (
    EXPERIMENTS,
    compute_per_condition,
    aggregate,
)

SUBSET = "all"
ERROR_KW = {"linewidth": 0.8, "alpha": 0.5}

SELECTED = {
    "Exp 1c\n(Crossed)": "One-sided\nconflict",
    "Exp 1d\n(Competing)": "Opposing\nprompts",
}


def main():
    probe_dir = REPO_ROOT / "results/probes/gemma3_10k_heldout_std_raw"
    probe_weights = np.load(probe_dir / "probes" / "probe_ridge_L31.npy")

    agg = {}
    for orig_key, label in SELECTED.items():
        records = compute_per_condition(EXPERIMENTS[orig_key], probe_weights)
        agg[label] = aggregate(records, SUBSET)

    labels = list(SELECTED.values())
    x = np.arange(len(labels))
    width = 0.3

    fig, (ax_r, ax_acc) = plt.subplots(1, 2, figsize=(8, 5))

    # Pearson r
    bars_bl_r = ax_r.bar(
        x - width / 2,
        [agg[n]["baseline_utils_r_mean"] for n in labels], width,
        yerr=[agg[n]["baseline_utils_r_se"] for n in labels],
        label="Baseline utilities", color="#B0B0B0", capsize=3, error_kw=ERROR_KW,
    )
    bars_pr_r = ax_r.bar(
        x + width / 2,
        [agg[n]["probe_r_mean"] for n in labels], width,
        yerr=[agg[n]["probe_r_se"] for n in labels],
        label="Probe", color="#6675B0", capsize=3, error_kw=ERROR_KW,
    )
    for bar in list(bars_bl_r) + list(bars_pr_r):
        val = bar.get_height()
        ax_r.text(bar.get_x() + bar.get_width() / 2, val + 0.04,
                  f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_r.set_title("Pearson r", fontsize=13)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels, fontsize=11)
    ax_r.set_ylim(0, 1.0)
    ax_r.set_ylabel("Pearson r", fontsize=11)
    ax_r.legend(loc="upper left", fontsize=9)

    # Pairwise accuracy
    bars_bl_acc = ax_acc.bar(
        x - width / 2,
        [agg[n]["baseline_utils_acc_mean"] for n in labels], width,
        yerr=[agg[n]["baseline_utils_acc_se"] for n in labels],
        label="Baseline utilities", color="#B0B0B0", capsize=3, error_kw=ERROR_KW,
    )
    bars_pr_acc = ax_acc.bar(
        x + width / 2,
        [agg[n]["probe_acc_mean"] for n in labels], width,
        yerr=[agg[n]["probe_acc_se"] for n in labels],
        label="Probe", color="#6675B0", capsize=3, error_kw=ERROR_KW,
    )
    for bar in list(bars_bl_acc) + list(bars_pr_acc):
        val = bar.get_height()
        ax_acc.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_acc.set_title("Pairwise accuracy", fontsize=13)
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(labels, fontsize=11)
    ax_acc.set_ylim(0.5, 1.0)
    ax_acc.set_ylabel("Pairwise accuracy", fontsize=11)
    ax_acc.axhline(y=0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_acc.legend(loc="upper left", fontsize=9)

    fig.suptitle("Probe vs baseline utilities", fontsize=14)
    fig.tight_layout()

    out = ASSETS_DIR / "plot_030226_s4_utility_bars_conflict_opposing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
