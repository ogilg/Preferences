"""Plot utility fitting results from multilayer_results.json."""

from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")

DATA_PATH = Path("experiments/ood_system_prompts/utility_fitting/multilayer_results.json")
ASSETS_DIR = Path("experiments/ood_system_prompts/utility_fitting/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH) as f:
    raw = json.load(f)

df = pd.DataFrame(raw)

# Normalize experiment names: strip layer suffix for grouping
df["exp_base"] = df["experiment"].str.replace(r"_L\d+", "", regex=True)
# For mra_experiments, layer is already in the 'layer' column
# For exp1{b,c,d}_L{31,43,55}, extract layer from experiment name
df["layer"] = df.apply(
    lambda row: int(row["experiment"].split("_L")[1]) if "_L" in row["experiment"] else row["layer"],
    axis=1,
)

COLORS = {"exp1b": "#2196F3", "exp1c": "#FF9800", "exp1d": "#4CAF50", "mra_experiments": "#9C27B0"}
EXP_LABELS = {"exp1b": "Exp 1b", "exp1c": "Exp 1c", "exp1d": "Exp 1d", "mra_experiments": "MRA"}


# =============================================================================
# Plot 1: Overview bar chart — mean r by method, layer 31 only
# =============================================================================
def plot_overview_barplot():
    sub = df[(df["layer"] == 31) & (df["exp_base"].isin(["exp1b", "exp1c", "exp1d"]))].copy()
    methods = [
        ("cond_probe_r", "Condition probe"),
        ("bl_probe_r", "Baseline probe"),
        ("bl_utils_r", "Baseline utilities"),
    ]
    exps = ["exp1b", "exp1c", "exp1d"]
    method_colors = ["#2196F3", "#FF9800", "#9E9E9E"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(exps))
    width = 0.25

    for i, (col, label) in enumerate(methods):
        means = []
        ses = []
        for exp in exps:
            vals = sub[sub["exp_base"] == exp][col].values
            means.append(np.mean(vals))
            ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.bar(x + i * width, means, width, yerr=ses, label=label,
               color=method_colors[i], capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean Pearson r")
    ax.set_xticks(x + width)
    ax.set_xticklabels([EXP_LABELS[e] for e in exps])
    ax.legend(frameon=True)
    ax.set_title("Prediction accuracy by method (Layer 31)")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_022727_overview_barplot.png", dpi=200)
    plt.close(fig)
    print("Saved overview_barplot")


# =============================================================================
# Plot 2: Layer comparison — condition probe r across layers
# =============================================================================
def plot_layer_comparison():
    exps = ["exp1b", "exp1c", "exp1d"]
    layers = [31, 43, 55]

    fig, ax = plt.subplots(figsize=(7, 5))
    for exp in exps:
        means = []
        for layer in layers:
            vals = df[(df["exp_base"] == exp) & (df["layer"] == layer)]["cond_probe_r"].values
            means.append(np.mean(vals))
        ax.plot(layers, means, "o-", label=EXP_LABELS[exp], color=COLORS[exp], linewidth=2, markersize=7)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean condition probe r")
    ax.set_xticks(layers)
    ax.legend(frameon=True)
    ax.set_title("Condition probe accuracy across layers")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_022727_layer_comparison.png", dpi=200)
    plt.close(fig)
    print("Saved layer_comparison")


# =============================================================================
# Plot 3: Exp1b per-condition horizontal bar chart (layer 31)
# =============================================================================
def plot_exp1b_conditions():
    sub = df[(df["exp_base"] == "exp1b") & (df["layer"] == 31)].copy()
    sub = sub.sort_values("cond_probe_r", ascending=True)

    # Extract topic from condition name for coloring
    topics = sub["condition"].str.replace(r"_(pos|neg)_persona", "", regex=True).values
    unique_topics = sorted(set(topics))
    topic_cmap = plt.cm.Set2(np.linspace(0, 1, len(unique_topics)))
    topic_colors = {t: topic_cmap[i] for i, t in enumerate(unique_topics)}
    bar_colors = [topic_colors[t] for t in topics]

    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(sub))
    ax.barh(y_pos, sub["cond_probe_r"].values, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.scatter(sub["bl_probe_r"].values, y_pos, marker="d", color="black", s=30, zorder=5,
               label="Baseline probe r")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["condition"].values, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pearson r")
    ax.legend(loc="lower right", frameon=True)
    ax.set_title("Exp 1b: Condition probe r by condition (Layer 31)")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_022727_exp1b_conditions.png", dpi=200)
    plt.close(fig)
    print("Saved exp1b_conditions")


# =============================================================================
# Plot 4: Exp1d shellpos vs topicpos scatter
# =============================================================================
def plot_exp1d_scatter():
    sub = df[(df["exp_base"] == "exp1d") & (df["layer"] == 31)].copy()

    # Parse pair names: e.g. compete_ancient_history_coding_shellpos -> ancient_history_coding
    sub["pair"] = sub["condition"].str.replace("compete_", "").str.replace(r"_(shellpos|topicpos)$", "", regex=True)
    sub["pos_type"] = sub["condition"].str.extract(r"(shellpos|topicpos)$")[0]

    pairs = sub.pivot(index="pair", columns="pos_type", values="cond_probe_r")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pairs["shellpos"], pairs["topicpos"], s=60, color=COLORS["exp1d"], zorder=5)

    for pair_name, row in pairs.iterrows():
        ax.annotate(pair_name, (row["shellpos"], row["topicpos"]),
                    fontsize=7, textcoords="offset points", xytext=(5, 5))

    lims = [0, 1]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Shell-positive condition probe r")
    ax.set_ylabel("Topic-positive condition probe r")
    ax.set_title("Exp 1d: Shell-positive vs Topic-positive (Layer 31)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_022727_exp1d_shellpos_vs_topicpos.png", dpi=200)
    plt.close(fig)
    print("Saved exp1d_shellpos_vs_topicpos")


# =============================================================================
# Plot 5: Condition probe r vs baseline probe r scatter (all, layer 31)
# =============================================================================
def plot_cond_vs_baseline():
    sub = df[df["layer"] == 31].copy()

    fig, ax = plt.subplots(figsize=(7, 7))
    for exp in ["exp1b", "exp1c", "exp1d", "mra_experiments"]:
        mask = sub["exp_base"] == exp
        ax.scatter(sub.loc[mask, "bl_probe_r"], sub.loc[mask, "cond_probe_r"],
                   s=40, color=COLORS[exp], label=EXP_LABELS[exp], alpha=0.8, zorder=5)

    lims = [-0.5, 1]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Baseline probe r")
    ax.set_ylabel("Condition probe r")
    ax.legend(frameon=True)
    ax.set_title("Condition probe vs baseline probe (Layer 31)")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_022727_cond_vs_baseline_scatter.png", dpi=200)
    plt.close(fig)
    print("Saved cond_vs_baseline_scatter")


if __name__ == "__main__":
    plot_overview_barplot()
    plot_layer_comparison()
    plot_exp1b_conditions()
    plot_exp1d_scatter()
    plot_cond_vs_baseline()
    print(f"\nAll plots saved to {ASSETS_DIR}")
