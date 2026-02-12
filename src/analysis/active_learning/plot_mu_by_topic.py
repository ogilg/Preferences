"""Per-topic analysis of active learning results: mu, sigma, refusal rate by topic.

Merges ranked_tasks with topic classifications from topic_classification output.

Usage:
    python -m src.analysis.active_learning.plot_mu_by_topic \
        --experiment-id gemma3_500_completion_preference \
        --topics-json src/analysis/topic_classification/output/topics.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kruskal

from src.analysis.active_learning.utils import load_ranked_tasks, plot_output_path
from src.analysis.topic_classification.classify import MODELS

TOPIC_COLORS = {
    "fiction": "#E07B54",
    "persuasive_writing": "#5BA37E",
    "content_generation": "#6B8FBF",
    "knowledge_qa": "#C4A642",
    "coding": "#8B6BB0",
    "math": "#4C72B0",
    "summarization": "#CC6699",
    "harmful_request": "#C44E52",
    "sensitive_creative": "#D4A574",
    "model_manipulation": "#7B4F8A",
    "security_legal": "#B05050",
    "other": "#999999",
}


def merge_tasks_with_topics(
    tasks: list[dict], topics_cache: dict, model: str,
) -> list[dict]:
    """Add 'topic' field to each task from topic classification cache."""
    merged = []
    for t in tasks:
        tid = t["task_id"]
        if tid in topics_cache and model in topics_cache[tid]:
            t_copy = dict(t)
            t_copy["topic"] = topics_cache[tid][model]["primary"]
            merged.append(t_copy)
    return merged


def print_topic_stats(tasks: list[dict]) -> None:
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_topic[t["topic"]].append(t)

    print(f"\n{'Topic':<22} {'n':>5} {'Mean μ':>8} {'Std μ':>8} {'SEM μ':>8} {'Mean σ':>8} {'Ref Rate':>9}")
    print("-" * 80)

    rows = []
    for topic, ts in by_topic.items():
        mus = np.array([t["mu"] for t in ts])
        sigmas = np.array([t["sigma"] for t in ts])
        refusal_rates = np.array([t["refusal_rate"] for t in ts])
        rows.append({
            "topic": topic,
            "n": len(ts),
            "mean_mu": mus.mean(),
            "std_mu": mus.std(),
            "sem_mu": mus.std() / np.sqrt(len(mus)),
            "mean_sigma": sigmas.mean(),
            "mean_refusal": refusal_rates.mean(),
        })

    rows.sort(key=lambda r: r["mean_mu"], reverse=True)
    for r in rows:
        print(
            f"{r['topic']:<22} {r['n']:>5} {r['mean_mu']:>+8.3f} {r['std_mu']:>8.3f} "
            f"{r['sem_mu']:>8.3f} {r['mean_sigma']:>8.3f} {r['mean_refusal']:>8.1%}"
        )

    # Kruskal-Wallis test across topics
    groups = [np.array([t["mu"] for t in ts]) for ts in by_topic.values() if len(ts) >= 5]
    if len(groups) >= 2:
        h_stat, p_val = kruskal(*groups)
        print(f"\nKruskal-Wallis H={h_stat:.2f}, p={p_val:.3g} (across {len(groups)} topics)")


def plot_topic_analysis(tasks: list[dict], output_path: Path, title: str) -> None:
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_topic[t["topic"]].append(t)

    # Sort topics by mean mu
    topic_order = sorted(by_topic.keys(), key=lambda tp: np.mean([t["mu"] for t in by_topic[tp]]), reverse=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Panel 1: Mean mu by topic (bar chart) ---
    ax1 = axes[0, 0]
    means = [np.mean([t["mu"] for t in by_topic[tp]]) for tp in topic_order]
    sems = [np.std([t["mu"] for t in by_topic[tp]]) / np.sqrt(len(by_topic[tp])) for tp in topic_order]
    ns = [len(by_topic[tp]) for tp in topic_order]
    colors = [TOPIC_COLORS.get(tp, "#999999") for tp in topic_order]

    bars = ax1.bar(range(len(topic_order)), means, yerr=sems, capsize=4, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_xticks(range(len(topic_order)))
    ax1.set_xticklabels(topic_order, rotation=45, ha="right", fontsize=9)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Mean Utility (μ)")
    ax1.set_title("Mean Preference by Topic")
    for bar, n, mean in zip(bars, ns, means):
        y_off = 0.1 if mean >= 0 else -0.15
        ax1.text(bar.get_x() + bar.get_width() / 2, mean + y_off, f"n={n}",
                 ha="center", va="bottom" if mean >= 0 else "top", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # --- Panel 2: Violin plot of mu distribution per topic ---
    ax2 = axes[0, 1]
    violin_data = [[t["mu"] for t in by_topic[tp]] for tp in topic_order]
    parts = ax2.violinplot(violin_data, positions=range(len(topic_order)), showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    ax2.set_xticks(range(len(topic_order)))
    ax2.set_xticklabels(topic_order, rotation=45, ha="right", fontsize=9)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Utility (μ)")
    ax2.set_title("Preference Distribution by Topic")
    ax2.grid(axis="y", alpha=0.3)

    # --- Panel 3: Mean sigma by topic ---
    ax3 = axes[1, 0]
    sigma_means = [np.mean([t["sigma"] for t in by_topic[tp]]) for tp in topic_order]
    sigma_sems = [np.std([t["sigma"] for t in by_topic[tp]]) / np.sqrt(len(by_topic[tp])) for tp in topic_order]

    bars3 = ax3.bar(range(len(topic_order)), sigma_means, yerr=sigma_sems, capsize=4,
                    color=colors, edgecolor="black", alpha=0.85)
    ax3.set_xticks(range(len(topic_order)))
    ax3.set_xticklabels(topic_order, rotation=45, ha="right", fontsize=9)
    ax3.set_ylabel("Mean Uncertainty (σ)")
    ax3.set_title("Mean Uncertainty by Topic")
    ax3.grid(axis="y", alpha=0.3)

    # --- Panel 4: Mean refusal rate by topic ---
    ax4 = axes[1, 1]
    refusal_means = [np.mean([t["refusal_rate"] for t in by_topic[tp]]) * 100 for tp in topic_order]

    bars4 = ax4.bar(range(len(topic_order)), refusal_means, color=colors, edgecolor="black", alpha=0.85)
    ax4.set_xticks(range(len(topic_order)))
    ax4.set_xticklabels(topic_order, rotation=45, ha="right", fontsize=9)
    ax4.set_ylabel("Mean Refusal Rate (%)")
    ax4.set_title("Mean Refusal Rate by Topic")
    ax4.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-topic analysis of active learning results")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--topics-json", type=str, required=True,
                        help="Path to topic classification cache JSON")
    parser.add_argument("--model", type=str, default=MODELS[0],
                        help="Classifier model to use for topic labels")
    args = parser.parse_args()

    tasks = load_ranked_tasks(args.experiment_id, args.run_name)
    print(f"Loaded {len(tasks)} ranked tasks")

    with open(args.topics_json) as f:
        topics_cache = json.load(f)
    print(f"Loaded {len(topics_cache)} topic classifications")

    merged = merge_tasks_with_topics(tasks, topics_cache, args.model)
    print(f"Merged: {len(merged)}/{len(tasks)} tasks have topic labels")

    if not merged:
        print("No tasks with topic labels — run topic classification first.")
        return

    print_topic_stats(merged)

    display_name = f"{args.experiment_id} ({args.run_name})" if args.run_name else args.experiment_id
    output = plot_output_path(args.experiment_id, "mu_by_topic", args.run_name)
    plot_topic_analysis(merged, output, f"Per-Topic Preference Analysis\n{display_name}")


if __name__ == "__main__":
    main()
