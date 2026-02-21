"""Generate scatter and summary plots for the OOD system prompt experiment."""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(REPO_ROOT, "experiments/ood_system_prompts/analysis_results_full.json")
ASSETS_DIR = os.path.join(REPO_ROOT, "experiments/ood_system_prompts/assets")

LAYERS = ["L31", "L43", "L55"]
LAYER_LABELS = ["L31", "L43", "L55"]

EXP_KEYS = ["exp1a", "exp1b", "exp1c", "exp1d", "exp2", "exp3"]
EXP_LABELS = ["1a\nCategory", "1b\nHidden", "1c\nCrossed", "1d\nCompeting", "2\nRoles", "3\nMinimal"]
EXP_TITLES = {
    "exp1a": "Exp 1a: Category preference (n=360)",
    "exp1b": "Exp 1b: Hidden preference (n=640)",
    "exp1c": "Exp 1c: Crossed preference (n=640)",
    "exp1d": "Exp 1d: Competing (n=1600)",
    "exp2": "Exp 2: Roles (n=1000)",
    "exp3": "Exp 3: Minimal pairs (n=100)",
}

COLORS_LAYERS = ["#e41a1c", "#377eb8", "#4daf4a"]  # red, blue, green (Set1)
COLORS_EXPS = plt.cm.tab10.colors


def get_exp_data(data, exp_key, layer_key):
    """Extract flat (behavioral_deltas, probe_deltas, r, sign_agreement, n) for an experiment/layer."""
    exp = data[exp_key]
    layer = exp[layer_key]
    if exp_key == "exp1d":
        layer = layer["full"]
    return (
        np.array(layer["behavioral_deltas"]),
        np.array(layer["probe_deltas"]),
        layer["pearson_r"],
        layer["sign_agreement"],
        layer["n"],
    )


def plot_summary_pearson_r(data, save_path):
    x = np.arange(len(EXP_KEYS))
    n_layers = len(LAYERS)
    width = 0.22

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    for i, (layer, color) in enumerate(zip(LAYERS, COLORS_LAYERS)):
        r_values = []
        sign_values = []
        for exp_key in EXP_KEYS:
            _, _, r, sign, _ = get_exp_data(data, exp_key, layer)
            r_values.append(r)
            sign_values.append(sign * 100)

        offset = (i - 1) * width
        bars = ax1.bar(x + offset, r_values, width=width, color=color, alpha=0.85, label=f"{layer} Pearson r", zorder=3)

    # Sign agreement: plot as markers connected by lines, one per layer
    for i, (layer, color) in enumerate(zip(LAYERS, COLORS_LAYERS)):
        sign_values = []
        for exp_key in EXP_KEYS:
            _, _, _, sign, _ = get_exp_data(data, exp_key, layer)
            sign_values.append(sign * 100)
        offset = (i - 1) * width
        ax2.plot(
            x + offset, sign_values,
            marker="D", markersize=5, color=color, alpha=0.65,
            linestyle="none", label=f"{layer} Sign %",
        )

    ax2.axhline(50, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label="50% chance")

    ax1.set_xticks(x)
    ax1.set_xticklabels(EXP_LABELS, fontsize=10)
    ax1.set_xlabel("Experiment", fontsize=11)
    ax1.set_ylabel("Pearson r", fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    ax2.set_ylabel("Sign agreement (%)", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.set_yticks(np.arange(0, 110, 10))

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8, ncol=2)

    ax1.set_title("OOD System Prompts: Pearson r and Sign Agreement by Experiment and Layer", fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_scatter_L31(data, save_path):
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    for idx, exp_key in enumerate(EXP_KEYS):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])

        bd, pd, r, sign, n = get_exp_data(data, exp_key, "L31")

        ax.scatter(bd, pd, alpha=0.25, s=12, color="#377eb8", zorder=2, rasterized=True)

        # Linear regression line
        slope, intercept, *_ = stats.linregress(bd, pd)
        x_range = np.linspace(bd.min(), bd.max(), 200)
        ax.plot(x_range, slope * x_range + intercept, color="#e41a1c", linewidth=1.5, zorder=3)

        # Reference lines
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="-", alpha=0.5, zorder=1)
        ax.axvline(0, color="gray", linewidth=0.6, linestyle="-", alpha=0.5, zorder=1)

        # Set x-axis to behavioral delta range [-1, 1]
        ax.set_xlim(-1.05, 1.05)
        # Set y-axis to probe delta range with padding
        y_lim = max(abs(pd).max(), 1) * 1.15
        ax.set_ylim(-y_lim, y_lim)

        ax.set_title(EXP_TITLES[exp_key], fontsize=9, pad=4)
        ax.set_xlabel("Behavioral delta (Δp_choose)", fontsize=8)
        ax.set_ylabel("Probe delta (Δprobe score)", fontsize=8)
        ax.tick_params(labelsize=7)

        # Annotate r and sign agreement
        ax.text(
            0.03, 0.97,
            f"r = {r:.3f}\nsign = {sign*100:.1f}%",
            transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    fig.suptitle("OOD System Prompts: Behavioral vs Probe Deltas at L31", fontsize=13, y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_layer_comparison(data, save_path):
    layer_x = [31, 43, 55]

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (exp_key, label) in enumerate(zip(EXP_KEYS, EXP_LABELS)):
        color = COLORS_EXPS[idx]
        sign_vals = []
        for layer in LAYERS:
            _, _, _, sign, _ = get_exp_data(data, exp_key, layer)
            sign_vals.append(sign * 100)
        ax.plot(layer_x, sign_vals, marker="o", markersize=7, color=color, linewidth=1.8,
                label=label.replace("\n", " "))

    ax.axhline(50, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, label="50% (chance)")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Sign agreement (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 110, 10))
    ax.set_xticks(layer_x)
    ax.set_xticklabels(LAYER_LABELS)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("OOD System Prompts: Sign Agreement by Layer and Experiment", fontsize=12, pad=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    with open(DATA_PATH) as f:
        data = json.load(f)

    plot_summary_pearson_r(
        data,
        os.path.join(ASSETS_DIR, "plot_022126_summary_pearson_r.png"),
    )
    plot_scatter_L31(
        data,
        os.path.join(ASSETS_DIR, "plot_022126_scatter_L31.png"),
    )
    plot_layer_comparison(
        data,
        os.path.join(ASSETS_DIR, "plot_022126_layer_comparison.png"),
    )


if __name__ == "__main__":
    main()
