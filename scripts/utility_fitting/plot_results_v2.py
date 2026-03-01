"""Improved plots for utility fitting results."""

from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")

DATA_PATH = Path("experiments/ood_system_prompts/utility_fitting/analysis_results.json")
ASSETS_DIR = Path("experiments/ood_system_prompts/utility_fitting/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH) as f:
    raw = json.load(f)

df = pd.DataFrame(raw)


# =============================================================================
# Plot 1: Exp1b conditions — horizontal bars colored by polarity
# =============================================================================
def plot_exp1b_conditions_v2():
    sub = df[df["experiment"] == "exp1b_hidden"].copy()

    # Extract topic and polarity
    sub["topic"] = sub["condition"].str.replace(r"_(pos|neg)_persona$", "", regex=True)
    sub["polarity"] = sub["condition"].str.extract(r"_(pos|neg)_persona$")[0]

    # Sort by topic name then polarity (neg before pos within each topic)
    sub = sub.sort_values(["topic", "polarity"], ascending=[True, True])
    # Reverse so first topic is at the top
    sub = sub.iloc[::-1].reset_index(drop=True)

    # Clean labels
    labels = []
    for _, row in sub.iterrows():
        topic_clean = row["topic"].replace("_", " ").title()
        pol = "neg" if row["polarity"] == "neg" else "pos"
        labels.append(f"{topic_clean} ({pol})")

    # Colors
    color_neg = "#E57373"  # warm red/coral
    color_pos = "#4DB6AC"  # cool blue/teal
    bar_colors = [color_neg if p == "neg" else color_pos for p in sub["polarity"].values]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(sub))
    ax.barh(y_pos, sub["cond_probe_r"].values, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.7)

    # Baseline probe r as black diamonds
    ax.scatter(sub["bl_probe_r"].values, y_pos, marker="D", color="black", s=25, zorder=5,
               label="Baseline probe r")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pearson r", fontsize=11)
    ax.set_title("Exp 1b: Condition probe r by persona condition", fontsize=13)

    # Legend with color patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_neg, label="Negative persona"),
        Patch(facecolor=color_pos, label="Positive persona"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
                   markersize=6, label="Baseline probe r"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9)

    fig.tight_layout()
    out = ASSETS_DIR / "plot_022828_exp1b_conditions_v2.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


# =============================================================================
# Plot 2: Exp1d competing — grouped horizontal bars
# =============================================================================
def plot_exp1d_competing_overview():
    sub = df[df["experiment"] == "exp1d_competing"].copy()

    # Parse pair and condition type
    sub["pair_raw"] = sub["condition"].str.replace("compete_", "").str.replace(r"_(shellpos|topicpos)$", "", regex=True)
    sub["pos_type"] = sub["condition"].str.extract(r"(shellpos|topicpos)$")[0]

    # Build pivot for cond_probe_r and bl_probe_r
    pivot_cond = sub.pivot(index="pair_raw", columns="pos_type", values="cond_probe_r")
    pivot_bl = sub.pivot(index="pair_raw", columns="pos_type", values="bl_probe_r")

    # Clean pair labels: "ancient_history_coding" -> "Ancient History vs Coding"
    def clean_pair(pair_raw: str) -> str:
        # The pairs are topic_shell where topic and shell can be multi-word
        # Known shells: coding, math, fiction
        # Known topics: ancient_history, astronomy, cats, cheese, classical_music, cooking, gardening, rainy_weather
        shells = {"coding", "math", "fiction"}
        parts = pair_raw.split("_")
        # Find the shell (last word or last part)
        shell = parts[-1]
        if shell in shells:
            topic_parts = parts[:-1]
        else:
            # Shouldn't happen based on data
            topic_parts = parts[:-1]
            shell = parts[-1]
        topic = " ".join(topic_parts).title()
        shell_clean = shell.title()
        return f"{topic} vs {shell_clean}"

    pair_labels = {p: clean_pair(p) for p in pivot_cond.index}

    # Sort alphabetically by pair label
    sorted_pairs = sorted(pivot_cond.index, key=lambda p: pair_labels[p])
    # Reverse for top-to-bottom reading
    sorted_pairs = sorted_pairs[::-1]

    color_topic = "#5C6BC0"  # indigo
    color_shell = "#FF8A65"  # orange

    n_pairs = len(sorted_pairs)
    bar_height = 0.35
    fig, ax = plt.subplots(figsize=(10, 8))

    y_base = np.arange(n_pairs)

    # Topic-positive bars (upper bar in each group)
    topicpos_vals = [pivot_cond.loc[p, "topicpos"] for p in sorted_pairs]
    shellpos_vals = [pivot_cond.loc[p, "shellpos"] for p in sorted_pairs]

    ax.barh(y_base + bar_height / 2, topicpos_vals, bar_height, color=color_topic,
            edgecolor="white", linewidth=0.5, label="Topic-positive (love topic, hate shell)")
    ax.barh(y_base - bar_height / 2, shellpos_vals, bar_height, color=color_shell,
            edgecolor="white", linewidth=0.5, label="Shell-positive (love shell, hate topic)")

    # Baseline markers
    bl_topicpos = [pivot_bl.loc[p, "topicpos"] for p in sorted_pairs]
    bl_shellpos = [pivot_bl.loc[p, "shellpos"] for p in sorted_pairs]
    ax.scatter(bl_topicpos, y_base + bar_height / 2, marker="D", color="black", s=20, zorder=5)
    ax.scatter(bl_shellpos, y_base - bar_height / 2, marker="o", facecolors="none",
               edgecolors="black", s=25, zorder=5, linewidths=1.2)

    # Aggregate means
    mean_topicpos = np.mean(topicpos_vals)
    mean_shellpos = np.mean(shellpos_vals)

    # Add aggregate row at top
    agg_y = n_pairs + 0.7
    ax.barh(agg_y + bar_height / 2, mean_topicpos, bar_height, color=color_topic,
            edgecolor="black", linewidth=1.0, alpha=0.85)
    ax.barh(agg_y - bar_height / 2, mean_shellpos, bar_height, color=color_shell,
            edgecolor="black", linewidth=1.0, alpha=0.85)
    # Separator line
    ax.axhline(y=n_pairs + 0.1, color="gray", linewidth=0.8, linestyle="--")

    # Y-axis labels
    all_y = list(y_base) + [agg_y]
    all_labels = [pair_labels[p] for p in sorted_pairs] + ["Aggregate mean"]
    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labels, fontsize=9)
    # Bold the aggregate label
    ax.get_yticklabels()[-1].set_fontweight("bold")

    ax.set_xlim(0, 1)
    ax.set_xlabel("Condition probe Pearson r", fontsize=11)
    ax.set_title("Exp 1d: Topic vs shell preferences in competing conditions", fontsize=13)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_topic, label="Topic-positive"),
        Patch(facecolor=color_shell, label="Shell-positive"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
                   markersize=6, label="Baseline (topic-pos)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor="black", markersize=6, markeredgewidth=1.2,
                   label="Baseline (shell-pos)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9)

    fig.tight_layout()
    out = ASSETS_DIR / "plot_022828_exp1d_competing_overview.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_exp1b_conditions_v2()
    plot_exp1d_competing_overview()
    print(f"\nAll plots saved to {ASSETS_DIR}")
