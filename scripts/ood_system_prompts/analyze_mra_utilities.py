"""Per-topic utility analysis for MRA personas.

Loads Thurstonian utilities across all 3 splits (A, B, C), zero-centers,
computes per-topic deltas vs noprompt baseline, and generates plots.

Usage: python -m scripts.ood_system_prompts.analyze_mra_utilities
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.measurement.storage.loading import load_run_utilities

REPO_ROOT = Path(__file__).parent.parent.parent
TOPICS_PATH = REPO_ROOT / "data" / "topics" / "topics.json"
ASSETS = REPO_ROOT / "experiments" / "probe_generalization" / "multi_role_ablation" / "assets"

# Persona definitions: (results_dir, sys_hash)
MRA_EXP2 = REPO_ROOT / "results" / "experiments" / "mra_exp2" / "pre_task_active_learning"
MRA_EXP3 = REPO_ROOT / "results" / "experiments" / "mra_exp3" / "pre_task_active_learning"

PERSONAS = {
    # exp2 personas
    "noprompt": (MRA_EXP2, ""),
    "villain": (MRA_EXP2, "syse8f24ac6"),
    "aesthete": (MRA_EXP2, "sys021d8ca1"),
    "midwest": (MRA_EXP2, "sys5d504504"),
    # exp3 personas (evil)
    "provocateur": (MRA_EXP3, "sysf4d93514"),
    "trickster": (MRA_EXP3, "sys09a42edc"),
    "autocrat": (MRA_EXP3, "sys1c18219a"),
    "sadist": (MRA_EXP3, "sys39e01d59"),
}

# Grouped for reporting
ORIGINAL_PERSONAS = ["villain", "aesthete", "midwest"]
EVIL_PERSONAS = ["villain", "provocateur", "trickster", "autocrat", "sadist"]

SPLITS = ["a", "b", "c"]


def _find_run_dir(results_dir: Path, sys_suffix: str, split: str) -> Path:
    prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"
    if sys_suffix:
        prefix += f"_{sys_suffix}"
    dirname = f"{prefix}_mra_exp2_split_{split}_*"

    matches = list(results_dir.glob(dirname))
    assert len(matches) == 1, f"Expected 1 match for {dirname} in {results_dir}, got {len(matches)}"
    return matches[0]


def load_persona_utilities(persona_key: str) -> dict[str, float]:
    """Load and concatenate utilities across all splits for a persona, then zero-center."""
    results_dir, sys_suffix = PERSONAS[persona_key]
    all_utils: dict[str, float] = {}
    for split in SPLITS:
        run_dir = _find_run_dir(results_dir, sys_suffix, split)
        utilities, task_ids = load_run_utilities(run_dir)
        for tid, mu in zip(task_ids, utilities):
            all_utils[tid] = mu

    # Zero-center
    mean_val = np.mean(list(all_utils.values()))
    return {tid: mu - mean_val for tid, mu in all_utils.items()}


def load_topics() -> dict[str, str]:
    """Load primary topic for each task_id."""
    with open(TOPICS_PATH) as f:
        raw = json.load(f)
    topics = {}
    for tid, classifiers in raw.items():
        classification = classifiers["anthropic/claude-sonnet-4.5"]
        topics[tid] = classification["primary"]
    return topics


def compute_per_topic_stats(
    persona_utils: dict[str, float],
    baseline_utils: dict[str, float],
    topics: dict[str, str],
) -> pd.DataFrame:
    """Compute per-topic mean utility, std, and delta from baseline."""
    shared_ids = sorted(set(persona_utils) & set(baseline_utils) & set(topics))

    rows = []
    for tid in shared_ids:
        rows.append({
            "task_id": tid,
            "topic": topics[tid],
            "persona_mu": persona_utils[tid],
            "baseline_mu": baseline_utils[tid],
            "delta": persona_utils[tid] - baseline_utils[tid],
        })
    df = pd.DataFrame(rows)

    stats = df.groupby("topic").agg(
        persona_mean=("persona_mu", "mean"),
        persona_std=("persona_mu", "std"),
        baseline_mean=("baseline_mu", "mean"),
        baseline_std=("baseline_mu", "std"),
        delta_mean=("delta", "mean"),
        delta_std=("delta", "std"),
        n=("task_id", "count"),
        corr=("persona_mu", lambda x: np.corrcoef(
            x.values,
            df.loc[x.index, "baseline_mu"].values,
        )[0, 1]),
    ).reset_index()
    return stats


def plot_topic_deltas(stats: pd.DataFrame, persona_name: str):
    stats_sorted = stats.sort_values("delta_mean")
    colors = ["#d62728" if d < 0 else "#2ca02c" for d in stats_sorted["delta_mean"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(stats_sorted)), stats_sorted["delta_mean"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(stats_sorted)))
    ax.set_yticklabels(stats_sorted["topic"])
    ax.set_xlabel("Mean Δu (persona − noprompt)")
    ax.set_title(f"{persona_name}: per-topic utility shift vs baseline")
    ax.axvline(0, color="black", linewidth=0.5)

    for i, (_, row) in enumerate(stats_sorted.iterrows()):
        offset = 0.3 if row["delta_mean"] >= 0 else -0.3
        ha = "left" if row["delta_mean"] >= 0 else "right"
        ax.text(row["delta_mean"] + offset, i, f"{row['delta_mean']:+.1f}", va="center", ha=ha, fontsize=8)

    plt.tight_layout()
    path = ASSETS / f"plot_030226_mra_{persona_name.lower()}_topic_delta.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_topic_absolute(stats: pd.DataFrame, persona_name: str):
    stats_sorted = stats.sort_values("baseline_mean")

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(stats_sorted))
    height = 0.35
    ax.barh(y - height / 2, stats_sorted["baseline_mean"], height, label="noprompt", color="#1f77b4", alpha=0.7)
    ax.barh(y + height / 2, stats_sorted["persona_mean"], height, label=persona_name, color="#ff7f0e", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(stats_sorted["topic"])
    ax.set_xlabel("Mean zero-centered utility")
    ax.set_title(f"{persona_name}: absolute per-topic utility vs baseline")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend()

    for i, (_, row) in enumerate(stats_sorted.iterrows()):
        for val, y_off in [(row["baseline_mean"], -height / 2), (row["persona_mean"], height / 2)]:
            offset = 0.2 if val >= 0 else -0.2
            ha = "left" if val >= 0 else "right"
            ax.text(val + offset, i + y_off, f"{val:.1f}", va="center", ha=ha, fontsize=7)

    plt.tight_layout()
    path = ASSETS / f"plot_030226_mra_{persona_name.lower()}_topic_absolute.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_within_topic_std(stats: pd.DataFrame, persona_name: str):
    stats_sorted = stats.sort_values("topic")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(stats_sorted))
    width = 0.35
    ax.bar(x - width / 2, stats_sorted["baseline_std"], width, label="noprompt", color="#1f77b4", alpha=0.7)
    ax.bar(x + width / 2, stats_sorted["persona_std"], width, label=persona_name, color="#ff7f0e", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(stats_sorted["topic"], rotation=45, ha="right")
    ax.set_ylabel("Within-topic std(utility)")
    ax.set_title(f"{persona_name}: within-topic utility spread")
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = ASSETS / f"plot_030226_mra_{persona_name.lower()}_within_topic_std.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_within_topic_corr(stats: pd.DataFrame, persona_name: str):
    stats_sorted = stats.sort_values("corr")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d62728" if c < 0 else "#1f77b4" for c in stats_sorted["corr"]]
    ax.barh(range(len(stats_sorted)), stats_sorted["corr"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(stats_sorted)))
    ax.set_yticklabels(stats_sorted["topic"])
    ax.set_xlabel("Within-topic Pearson r (persona vs noprompt)")
    ax.set_title(f"{persona_name}: within-topic correlation with baseline")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(-1, 1)

    for i, (_, row) in enumerate(stats_sorted.iterrows()):
        ax.text(row["corr"] + 0.03, i, f"{row['corr']:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    path = ASSETS / f"plot_030226_mra_{persona_name.lower()}_within_topic_corr.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_evil_comparison(all_stats: dict[str, pd.DataFrame]):
    """Cross-persona comparison of per-topic deltas for all evil personas."""
    topics = sorted(all_stats["villain"]["topic"].unique())
    personas = EVIL_PERSONAS
    n_topics = len(topics)
    n_personas = len(personas)

    fig, ax = plt.subplots(figsize=(14, 8))
    y = np.arange(n_topics)
    height = 0.8 / n_personas
    colors = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]

    for i, persona in enumerate(personas):
        stats = all_stats[persona].set_index("topic")
        deltas = [stats.loc[t, "delta_mean"] if t in stats.index else 0 for t in topics]
        ax.barh(y + i * height - 0.4 + height / 2, deltas, height, label=persona, color=colors[i], alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(topics)
    ax.set_xlabel("Mean Δu (persona − noprompt)")
    ax.set_title("Evil personas: per-topic utility shifts compared")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend(loc="lower right")

    plt.tight_layout()
    path = ASSETS / "plot_030226_mra_evil_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    topics = load_topics()

    print("Loading noprompt utilities...")
    baseline = load_persona_utilities("noprompt")
    print(f"  {len(baseline)} tasks")

    all_personas = ORIGINAL_PERSONAS + [p for p in EVIL_PERSONAS if p not in ORIGINAL_PERSONAS]
    all_stats: dict[str, pd.DataFrame] = {}

    for persona_key in all_personas:
        persona_name = persona_key.capitalize()
        print(f"\nLoading {persona_name} utilities...")
        persona_utils = load_persona_utilities(persona_key)
        print(f"  {len(persona_utils)} tasks")

        stats = compute_per_topic_stats(persona_utils, baseline, topics)
        all_stats[persona_key] = stats
        print(f"\n{persona_name} per-topic stats:")
        print(stats[["topic", "delta_mean", "persona_std", "baseline_std", "corr", "n"]].to_string(index=False))

        # Overall correlation
        shared = sorted(set(persona_utils) & set(baseline))
        p_vals = [persona_utils[t] for t in shared]
        b_vals = [baseline[t] for t in shared]
        r = np.corrcoef(p_vals, b_vals)[0, 1]
        print(f"\nOverall correlation (Pearson r): {r:.3f}, R²: {r**2:.3f}")

        plot_topic_absolute(stats, persona_name)
        plot_topic_deltas(stats, persona_name)
        plot_within_topic_std(stats, persona_name)
        plot_within_topic_corr(stats, persona_name)

    # Evil persona comparison
    plot_evil_comparison(all_stats)


if __name__ == "__main__":
    main()
