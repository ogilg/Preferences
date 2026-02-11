"""Behavioral validation plot for competing preferences experiment.

Shows choice rates under baseline, topic+, and format+ conditions for each pair.
Demonstrates that:
1. Both competing conditions affect behavior (rates change from baseline)
2. Format+ preserves more preference than topic+ (task type matters more than subject)
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("experiments/competing_preferences/results")
ASSETS_DIR = Path("docs/logs/assets/competing_preferences")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
DATE_STR = datetime.now().strftime("%m%d%y")


def load_behavioral() -> list[dict]:
    with open(RESULTS_DIR / "behavioral_competing.json") as f:
        return json.load(f)


def plot_behavioral_choice_rates(behavioral: list[dict], filename: str):
    """Grouped bar chart: baseline vs topic+ vs format+ choice rates per pair."""
    pair_ids = sorted(set(b["pair_id"] for b in behavioral))

    baselines = []
    topic_rates = []
    format_rates = []
    labels = []

    for pid in pair_ids:
        topic_b = [b for b in behavioral if b["pair_id"] == pid and b["favored_dim"] == "topic"][0]
        format_b = [b for b in behavioral if b["pair_id"] == pid and b["favored_dim"] == "shell"][0]

        baselines.append(topic_b["baseline_rate"])
        topic_rates.append(topic_b["manipulation_rate"])
        format_rates.append(format_b["manipulation_rate"])
        # Label: "cheese x math" style
        labels.append(f"{topic_b['target_topic']}\n× {topic_b['category_shell']}")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(pair_ids))
    width = 0.25

    ax.bar(x - width, baselines, width, label="Baseline (no prompt)", color="#9E9E9E", alpha=0.7, edgecolor="k", linewidth=0.5)
    ax.bar(x, topic_rates, width, label="\"Love subject, hate task type\"", color="#FF7043", alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.bar(x + width, format_rates, width, label="\"Love task type, hate subject\"", color="#42A5F5", alpha=0.8, edgecolor="k", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Choice rate (fraction choosing target task)", fontsize=11)
    ax.set_title("Both conditions suppress choice, but task type matters more than subject\n(identical content words in both prompts)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")

    # Add mean lines
    mean_topic = np.mean(topic_rates)
    mean_format = np.mean(format_rates)
    ax.axhline(mean_topic, color="#FF7043", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(mean_format, color="#42A5F5", linestyle="--", alpha=0.4, linewidth=1)

    # Annotate means
    ax.text(len(pair_ids) - 0.5, mean_topic + 0.02, f"mean = {mean_topic:.2f}", fontsize=8, color="#FF7043", ha="right")
    ax.text(len(pair_ids) - 0.5, mean_format + 0.02, f"mean = {mean_format:.2f}", fontsize=8, color="#42A5F5", ha="right")

    plt.tight_layout()
    path = ASSETS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return baselines, topic_rates, format_rates


def plot_format_vs_topic_dominance(behavioral: list[dict], filename: str):
    """Paired dot plot showing format+ rate vs topic+ rate per pair.

    Clearly shows: for most pairs, task type preference preserves more choice
    than subject preference. Lines connect the same pair.
    """
    pair_ids = sorted(set(b["pair_id"] for b in behavioral))

    topic_rates = []
    format_rates = []
    labels = []

    for pid in pair_ids:
        topic_b = [b for b in behavioral if b["pair_id"] == pid and b["favored_dim"] == "topic"][0]
        format_b = [b for b in behavioral if b["pair_id"] == pid and b["favored_dim"] == "shell"][0]
        topic_rates.append(topic_b["manipulation_rate"])
        format_rates.append(format_b["manipulation_rate"])
        labels.append(pid.replace("_", " "))

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot connected pairs
    for i, (tr, fr, label) in enumerate(zip(topic_rates, format_rates, labels)):
        color = "#42A5F5" if fr > tr else "#FF7043"
        ax.plot([tr, fr], [i, i], color=color, linewidth=1.5, alpha=0.6, zorder=1)

    ax.scatter(topic_rates, range(len(pair_ids)), s=60, color="#FF7043", label="\"Love subject, hate task type\"", zorder=5, edgecolors="k", linewidth=0.5)
    ax.scatter(format_rates, range(len(pair_ids)), s=60, color="#42A5F5", label="\"Love task type, hate subject\"", zorder=5, edgecolors="k", linewidth=0.5)

    ax.set_yticks(range(len(pair_ids)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Choice rate for target task", fontsize=11)
    ax.set_title("Task type dominates subject in driving preference\n(9/12 pairs: blue right of red)", fontsize=12)
    ax.invert_yaxis()
    ax.legend(fontsize=9, loc="lower right")
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)

    # Add t-test annotation
    t_stat, p_val = stats.ttest_rel(format_rates, topic_rates)
    n_format_wins = sum(1 for f, t in zip(format_rates, topic_rates) if f > t)
    ax.text(0.98, 0.02, f"Format > topic: {n_format_wins}/12 pairs\npaired t={t_stat:.2f}, p={p_val:.3f}",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = ASSETS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    behavioral = load_behavioral()

    print("=" * 60)
    print("BEHAVIORAL VALIDATION")
    print("=" * 60)

    pair_ids = sorted(set(b["pair_id"] for b in behavioral))

    topic_rates = []
    format_rates = []
    baselines = []

    for pid in pair_ids:
        topic_b = [b for b in behavioral if b["pair_id"] == pid and b["favored_dim"] == "topic"][0]
        format_b = [b for b in behavioral if b["pair_id"] == pid and b["favored_dim"] == "shell"][0]
        baselines.append(topic_b["baseline_rate"])
        topic_rates.append(topic_b["manipulation_rate"])
        format_rates.append(format_b["manipulation_rate"])

    print(f"\nMean baseline choice rate: {np.mean(baselines):.3f}")
    print(f"Mean choice rate under 'love subject': {np.mean(topic_rates):.3f}")
    print(f"Mean choice rate under 'love task type': {np.mean(format_rates):.3f}")

    t_stat, p_val = stats.ttest_rel(format_rates, topic_rates)
    print(f"\nFormat vs topic paired t-test: t={t_stat:.2f}, p={p_val:.4f}")
    n_format_wins = sum(1 for f, t in zip(format_rates, topic_rates) if f > t)
    print(f"Format > topic in {n_format_wins}/12 pairs")

    # Ratio
    mean_topic = np.mean(topic_rates)
    mean_format = np.mean(format_rates)
    if mean_topic > 0:
        print(f"Ratio (format / topic): {mean_format / mean_topic:.1f}x")

    plot_behavioral_choice_rates(behavioral, f"plot_{DATE_STR}_behavioral_rates.png")
    plot_format_vs_topic_dominance(behavioral, f"plot_{DATE_STR}_format_vs_topic.png")


if __name__ == "__main__":
    main()
