"""V3 plots: side-by-side Pearson r and pairwise accuracy panels."""

from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

plt.style.use("seaborn-v0_8-whitegrid")

DATA_PATH = Path("experiments/ood_system_prompts/utility_fitting/analysis_results.json")
ASSETS_DIR = Path("experiments/ood_system_prompts/utility_fitting/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH) as f:
    raw = json.load(f)

df = pd.DataFrame(raw)


# =============================================================================
# Plot 1: Exp1b — two horizontal bar panels (Pearson r | Pairwise accuracy)
# =============================================================================
def plot_exp1b_v3():
    sub = df[df["experiment"] == "exp1b_hidden"].copy()

    sub["topic"] = sub["condition"].str.replace(r"_(pos|neg)_persona$", "", regex=True)
    sub["polarity"] = sub["condition"].str.extract(r"_(pos|neg)_persona$")[0]

    # Sort: alphabetical by topic, neg before pos within each
    sub = sub.sort_values(["topic", "polarity"], ascending=[True, True])
    sub = sub.iloc[::-1].reset_index(drop=True)

    labels = []
    for _, row in sub.iterrows():
        topic_clean = row["topic"].replace("_", " ").title()
        pol = "neg" if row["polarity"] == "neg" else "pos"
        labels.append(f"{topic_clean} ({pol})")

    color_neg = "#E57373"
    color_pos = "#4DB6AC"
    bar_colors = [color_neg if p == "neg" else color_pos for p in sub["polarity"].values]

    fig, (ax_r, ax_acc) = plt.subplots(1, 2, sharey=True, figsize=(14, 8))
    y_pos = np.arange(len(sub))

    # Left panel: Pearson r
    ax_r.barh(y_pos, sub["cond_probe_r"].values, color=bar_colors,
              edgecolor="white", linewidth=0.5, height=0.7)
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(labels, fontsize=9)
    ax_r.set_xlim(0, 1)
    ax_r.set_xlabel("Pearson r", fontsize=11)
    ax_r.set_title("Pearson r", fontsize=13)

    # Right panel: Pairwise accuracy
    ax_acc.barh(y_pos, sub["cond_probe_acc"].values, color=bar_colors,
                edgecolor="white", linewidth=0.5, height=0.7)
    ax_acc.set_xlim(0.5, 1.0)
    ax_acc.set_xlabel("Pairwise accuracy", fontsize=11)
    ax_acc.set_title("Pairwise accuracy", fontsize=13)
    ax_acc.axvline(x=0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # Legend on right panel
    legend_elements = [
        Patch(facecolor=color_neg, label="Negative persona"),
        Patch(facecolor=color_pos, label="Positive persona"),
    ]
    ax_acc.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9)

    fig.suptitle("Exp 1b: Condition probe performance by persona condition", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = ASSETS_DIR / "plot_022828_exp1b_conditions_v3.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


# =============================================================================
# Plot 2: Exp1d — two horizontal bar panels (Pearson r | Pairwise accuracy)
# =============================================================================
def plot_exp1d_v3():
    sub = df[df["experiment"] == "exp1d_competing"].copy()

    sub["pair_raw"] = sub["condition"].str.replace("compete_", "").str.replace(
        r"_(shellpos|topicpos)$", "", regex=True
    )
    sub["pos_type"] = sub["condition"].str.extract(r"(shellpos|topicpos)$")[0]

    pivot_r = sub.pivot(index="pair_raw", columns="pos_type", values="cond_probe_r")
    pivot_acc = sub.pivot(index="pair_raw", columns="pos_type", values="cond_probe_acc")

    shells = {"coding", "math", "fiction"}

    def clean_pair(pair_raw: str) -> str:
        parts = pair_raw.split("_")
        shell = parts[-1]
        if shell in shells:
            topic_parts = parts[:-1]
        else:
            topic_parts = parts[:-1]
            shell = parts[-1]
        topic = " ".join(topic_parts).title()
        return f"{topic} vs {shell.title()}"

    pair_labels = {p: clean_pair(p) for p in pivot_r.index}
    sorted_pairs = sorted(pivot_r.index, key=lambda p: pair_labels[p])[::-1]

    color_topic = "#5C6BC0"
    color_shell = "#FF8A65"

    n_pairs = len(sorted_pairs)
    bar_height = 0.35

    fig, (ax_r, ax_acc) = plt.subplots(1, 2, sharey=True, figsize=(14, 8))
    y_base = np.arange(n_pairs)

    # Aggregate means
    topicpos_r = [pivot_r.loc[p, "topicpos"] for p in sorted_pairs]
    shellpos_r = [pivot_r.loc[p, "shellpos"] for p in sorted_pairs]
    topicpos_acc = [pivot_acc.loc[p, "topicpos"] for p in sorted_pairs]
    shellpos_acc = [pivot_acc.loc[p, "shellpos"] for p in sorted_pairs]

    agg_y = n_pairs + 0.7

    for ax, tp_vals, sp_vals, xlabel, title, xlim in [
        (ax_r, topicpos_r, shellpos_r, "Pearson r", "Pearson r", (0, 1)),
        (ax_acc, topicpos_acc, shellpos_acc, "Pairwise accuracy", "Pairwise accuracy", (0.5, 1.0)),
    ]:
        ax.barh(y_base + bar_height / 2, tp_vals, bar_height, color=color_topic,
                edgecolor="white", linewidth=0.5)
        ax.barh(y_base - bar_height / 2, sp_vals, bar_height, color=color_shell,
                edgecolor="white", linewidth=0.5)

        # Aggregate row
        ax.barh(agg_y + bar_height / 2, np.mean(tp_vals), bar_height, color=color_topic,
                edgecolor="black", linewidth=1.0, alpha=0.85)
        ax.barh(agg_y - bar_height / 2, np.mean(sp_vals), bar_height, color=color_shell,
                edgecolor="black", linewidth=1.0, alpha=0.85)
        ax.axhline(y=n_pairs + 0.1, color="gray", linewidth=0.8, linestyle="--")

        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(title, fontsize=13)

        if xlim[0] == 0.5:
            ax.axvline(x=0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # Y-axis labels on left panel only
    all_y = list(y_base) + [agg_y]
    all_labels = [pair_labels[p] for p in sorted_pairs] + ["Aggregate mean"]
    ax_r.set_yticks(all_y)
    ax_r.set_yticklabels(all_labels, fontsize=9)
    ax_r.get_yticklabels()[-1].set_fontweight("bold")

    legend_elements = [
        Patch(facecolor=color_topic, label="Topic-positive"),
        Patch(facecolor=color_shell, label="Shell-positive"),
    ]
    ax_acc.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9)

    fig.suptitle("Exp 1d: Topic vs shell preferences in competing conditions", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = ASSETS_DIR / "plot_022828_exp1d_competing_v3.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


# =============================================================================
# Plot 3: Overview — two grouped bar panels across experiments
# =============================================================================
def plot_overview_v3():
    # Compute per-experiment means and SEs for cond_probe_r and cond_probe_acc
    experiments = ["exp1b_hidden", "exp1c_crossed", "exp1d_competing"]
    exp_labels = ["Exp 1b\n(Hidden)", "Exp 1c\n(Crossed)", "Exp 1d\n(Competing)"]

    means_r, ses_r = [], []
    means_acc, ses_acc = [], []

    for exp in experiments:
        sub = df[df["experiment"] == exp]
        vals_r = sub["cond_probe_r"].values
        vals_acc = sub["cond_probe_acc"].values
        means_r.append(np.mean(vals_r))
        ses_r.append(np.std(vals_r, ddof=1) / np.sqrt(len(vals_r)))
        means_acc.append(np.mean(vals_acc))
        ses_acc.append(np.std(vals_acc, ddof=1) / np.sqrt(len(vals_acc)))

    color = "#5C6BC0"
    x = np.arange(len(experiments))
    bar_width = 0.5

    fig, (ax_r, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Pearson r
    ax_r.bar(x, means_r, bar_width, yerr=ses_r, color=color, edgecolor="white",
             linewidth=0.5, capsize=4, error_kw={"linewidth": 1.2})
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(exp_labels, fontsize=10)
    ax_r.set_ylim(0, 1)
    ax_r.set_ylabel("Pearson r", fontsize=11)
    ax_r.set_title("Pearson r", fontsize=13)

    # Right: Pairwise accuracy
    ax_acc.bar(x, means_acc, bar_width, yerr=ses_acc, color=color, edgecolor="white",
               linewidth=0.5, capsize=4, error_kw={"linewidth": 1.2})
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(exp_labels, fontsize=10)
    ax_acc.set_ylim(0.5, 1.0)
    ax_acc.set_ylabel("Pairwise accuracy", fontsize=11)
    ax_acc.set_title("Pairwise accuracy", fontsize=13)
    ax_acc.axhline(y=0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    fig.suptitle("Condition probe performance across experiments", fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = ASSETS_DIR / "plot_022828_overview_v3.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_exp1b_v3()
    plot_exp1d_v3()
    plot_overview_v3()
    print(f"\nAll plots saved to {ASSETS_DIR}")
