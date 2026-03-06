"""Generate scatter and bar plots for the OOD EOT experiment report.

Plot 1: 2x2 scatter grid of probe delta vs behavioral delta (exp1a-1d).
Plot 2: Grouped bar chart comparing prompt_last vs EOT probe performance.

Usage: python scripts/ood_eot/plot_eot_scatters.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.ood_eot.analyze_ground_truth_eot import recompute_experiment

ASSETS_DIR = REPO_ROOT / "experiments" / "ood_eot" / "assets"

EXPERIMENT_KEYS = ["exp1a", "exp1b", "exp1c", "exp1d"]
PANEL_TITLES = {
    "exp1a": "1a: Known categories",
    "exp1b": "1b: Novel topics",
    "exp1c": "1c: Topic in wrong shell",
    "exp1d": "1d: Competing valence",
}


def plot_scatter_4panel() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    for idx, key in enumerate(EXPERIMENT_KEYS):
        ax = axes_flat[idx]
        beh, probe, labels, per_point_gt = recompute_experiment(key)

        # Split into off-target and targeted
        off_target = per_point_gt == 0
        targeted_pos = per_point_gt > 0
        targeted_neg = per_point_gt < 0
        targeted = ~off_target

        # Plot off-target (grey)
        ax.scatter(
            beh[off_target], probe[off_target],
            c="#BDBDBD", alpha=0.3, s=12, edgecolors="none", zorder=1,
        )
        # Plot targeted positive (green)
        ax.scatter(
            beh[targeted_pos], probe[targeted_pos],
            c="#4CAF50", alpha=0.7, s=20, edgecolors="none", zorder=2,
        )
        # Plot targeted negative (red)
        ax.scatter(
            beh[targeted_neg], probe[targeted_neg],
            c="#E53935", alpha=0.7, s=20, edgecolors="none", zorder=2,
        )

        # Trend line: all data
        slope_all, intercept_all, r_all, _, _ = stats.linregress(beh, probe)
        x_range = np.array([beh.min(), beh.max()])
        ax.plot(
            x_range, slope_all * x_range + intercept_all,
            color="grey", linestyle="--", linewidth=1, zorder=3,
            label=f"All (r = {r_all:.2f})",
        )

        # Trend line: targeted only
        if targeted.sum() > 2:
            slope_t, intercept_t, r_t, _, _ = stats.linregress(beh[targeted], probe[targeted])
            x_range_t = np.array([beh[targeted].min(), beh[targeted].max()])
            ax.plot(
                x_range_t, slope_t * x_range_t + intercept_t,
                color="red", linestyle="-", linewidth=1.5, zorder=4,
                label=f"Targeted (r = {r_t:.2f})",
            )

        # Crosshairs
        ax.axhline(0, color="grey", linewidth=0.5, zorder=0)
        ax.axvline(0, color="grey", linewidth=0.5, zorder=0)

        ax.set_title(PANEL_TITLES[key], fontsize=11)
        ax.legend(fontsize=8, loc="lower right")

        # Axis labels only on edges
        row, col = divmod(idx, 2)
        if row == 1:
            ax.set_xlabel("Behavioral delta")
        if col == 0:
            ax.set_ylabel("Probe delta (EOT)")

    fig.suptitle("EOT probe: behavioral vs probe delta", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = ASSETS_DIR / "plot_030626_eot_scatter_4panel.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_comparison_bars() -> None:
    eot_path = REPO_ROOT / "experiments" / "ood_eot" / "ground_truth_results.json"
    pl_path = REPO_ROOT / "experiments" / "ood_system_prompts" / "ground_truth_results.json"

    with open(eot_path) as f:
        eot = json.load(f)
    with open(pl_path) as f:
        pl = json.load(f)

    metrics = [
        ("Beh\u2194Probe r (all)", "beh_probe_r_all"),
        ("Beh\u2194Probe r (targeted)", "beh_probe_r_on_target"),
        ("Probe\u2194GT r (targeted)", "probe_gt_r_on_target"),
    ]
    experiments = EXPERIMENT_KEYS

    n_groups = len(metrics)
    n_exps = len(experiments)
    bar_width = 0.08
    group_width = n_exps * 2 * bar_width + 0.15  # 2 bars per exp + gap

    fig, ax = plt.subplots(figsize=(10, 5))

    for g_idx, (metric_label, metric_key) in enumerate(metrics):
        group_center = g_idx * (group_width + 0.4)
        for e_idx, exp_key in enumerate(experiments):
            x_pl = group_center + e_idx * (2 * bar_width + 0.04)
            x_eot = x_pl + bar_width

            val_pl = pl[exp_key][metric_key]
            val_eot = eot[exp_key][metric_key]

            bar_pl = ax.bar(x_pl, val_pl, bar_width, color="#90CAF9", edgecolor="none")
            bar_eot = ax.bar(x_eot, val_eot, bar_width, color="#2196F3", edgecolor="none")

            ax.text(x_pl, val_pl + 0.01, f"{val_pl:.2f}", ha="center", va="bottom", fontsize=7)
            ax.text(x_eot, val_eot + 0.01, f"{val_eot:.2f}", ha="center", va="bottom", fontsize=7)

    # Build x-tick positions and labels
    tick_positions = []
    tick_labels_list = []
    for g_idx, (metric_label, _) in enumerate(metrics):
        group_center = g_idx * (group_width + 0.4)
        center = group_center + (n_exps - 1) * (2 * bar_width + 0.04) / 2 + bar_width / 2
        tick_positions.append(center)
        tick_labels_list.append(metric_label)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_list, fontsize=9)

    # Add experiment labels below bars
    for g_idx, (_, _) in enumerate(metrics):
        group_center = g_idx * (group_width + 0.4)
        for e_idx, exp_key in enumerate(experiments):
            x_mid = group_center + e_idx * (2 * bar_width + 0.04) + bar_width / 2
            ax.text(x_mid, -0.04, exp_key.replace("exp", ""), ha="center", va="top", fontsize=7, color="grey")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Correlation (r)")
    ax.set_title("prompt_last vs EOT probe: OOD tracking performance", fontsize=12)

    # Legend (use proxy artists)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#90CAF9", label="prompt_last"),
        Patch(facecolor="#2196F3", label="EOT (this experiment)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    fig.tight_layout()

    out_path = ASSETS_DIR / "plot_030626_eot_vs_promptlast_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating scatter plot (4 panel)...")
    plot_scatter_4panel()
    print("Generating comparison bar chart...")
    plot_comparison_bars()
    print("Done.")


if __name__ == "__main__":
    main()
