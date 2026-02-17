"""Analyze persona OOD pilot results: per-task deltas vs baseline, topic breakdowns, summary plot."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update({"font.size": 11})

RESULTS_PATH = Path("experiments/probe_generalization/persona_ood/pilot_results.json")
TOPICS_PATH = Path("src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json")
PLOT_PATH = Path("experiments/probe_generalization/persona_ood/assets/plot_021626_pilot_deltas.png")

PERSONA_DISPLAY = {
    "retired_diplomat": "Retired Diplomat",
    "overwhelmed_phd_student": "Overwhelmed PhD",
    "victorian_librarian": "Victorian Librarian",
    "street_artist": "Street Artist",
    "emergency_room_nurse": "ER Nurse",
}


def load_data():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    with open(TOPICS_PATH) as f:
        topics_raw = json.load(f)

    # Extract primary topic per task (use first model's classification)
    topics = {}
    for task_id, model_dict in topics_raw.items():
        first_model = next(iter(model_dict.values()))
        topics[task_id] = first_model["primary"]

    return results, topics


def compute_deltas(results: dict) -> pd.DataFrame:
    baseline_rates = results["baseline"]["task_rates"]
    task_ids = sorted(baseline_rates.keys())
    personas = [k for k in results if k != "baseline"]

    rows = []
    for persona in personas:
        persona_rates = results[persona]["task_rates"]
        for task_id in task_ids:
            bl = baseline_rates[task_id]["p_choose"]
            ps = persona_rates[task_id]["p_choose"]
            rows.append({
                "persona": persona,
                "task_id": task_id,
                "p_baseline": bl,
                "p_persona": ps,
                "delta": ps - bl,
            })

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, topics: dict) -> None:
    df = df.copy()
    df["abs_delta"] = df["delta"].abs()
    df["topic"] = df["task_id"].map(topics)

    personas = sorted(df["persona"].unique())
    print("=" * 80)
    print("PERSONA OOD PILOT — SUMMARY STATISTICS (vs baseline)")
    print("=" * 80)

    for persona in personas:
        sub = df[df["persona"] == persona]
        print(f"\n--- {PERSONA_DISPLAY[persona]} ({persona}) ---")
        print(f"  Mean delta:        {sub['delta'].mean():+.4f}")
        print(f"  Std delta:         {sub['delta'].std():.4f}")
        print(f"  Mean |delta|:      {sub['abs_delta'].mean():.4f}")
        print(f"  Max |delta|:       {sub['abs_delta'].max():.4f}")
        print(f"  Tasks |delta|>0.1: {(sub['abs_delta'] > 0.1).sum()} / {len(sub)}")
        print(f"  Tasks |delta|>0.2: {(sub['abs_delta'] > 0.2).sum()} / {len(sub)}")

        # Top 5 most shifted tasks
        top5 = sub.nlargest(5, "abs_delta")
        print("  Top 5 shifted tasks:")
        for _, row in top5.iterrows():
            topic = topics.get(row["task_id"], "?")
            print(
                f"    {row['task_id']:40s}  delta={row['delta']:+.3f}  "
                f"(bl={row['p_baseline']:.2f} -> ps={row['p_persona']:.2f})  [{topic}]"
            )

    # Cross-persona summary
    print("\n" + "=" * 80)
    print("CROSS-PERSONA COMPARISON")
    print("=" * 80)
    summary_rows = []
    for persona in personas:
        sub = df[df["persona"] == persona]
        summary_rows.append({
            "persona": PERSONA_DISPLAY[persona],
            "mean_delta": sub["delta"].mean(),
            "mean_abs_delta": sub["abs_delta"].mean(),
            "max_abs_delta": sub["abs_delta"].max(),
            "n_gt_0.1": (sub["abs_delta"] > 0.1).sum(),
            "n_gt_0.2": (sub["abs_delta"] > 0.2).sum(),
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False, float_format="%.4f"))

    # Topic breakdown
    print("\n" + "=" * 80)
    print("MEAN |DELTA| BY TOPIC CATEGORY")
    print("=" * 80)
    pivot = df.pivot_table(
        values="abs_delta", index="topic", columns="persona", aggfunc="mean"
    )
    pivot.columns = [PERSONA_DISPLAY[c] for c in pivot.columns]
    pivot["mean_across"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean_across", ascending=False)
    print(pivot.to_string(float_format="%.4f"))

    # Task count per topic
    print("\n" + "=" * 80)
    print("TASK COUNT PER TOPIC")
    print("=" * 80)
    topic_counts = df[df["persona"] == personas[0]].groupby("topic").size()
    print(topic_counts.sort_values(ascending=False).to_string())


def make_plot(df: pd.DataFrame, topics: dict) -> None:
    df = df.copy()
    df["abs_delta"] = df["delta"].abs()
    df["topic"] = df["task_id"].map(topics)
    df["persona_display"] = df["persona"].map(PERSONA_DISPLAY)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1.4]})

    # Panel 1: violin/box plot of deltas per persona
    ax1 = axes[0]
    order = [PERSONA_DISPLAY[p] for p in PERSONA_DISPLAY]
    sns.violinplot(
        data=df,
        x="persona_display",
        y="delta",
        order=order,
        ax=ax1,
        inner=None,
        color="lightblue",
        alpha=0.5,
        cut=0,
    )
    sns.stripplot(
        data=df,
        x="persona_display",
        y="delta",
        order=order,
        ax=ax1,
        color="black",
        alpha=0.3,
        size=3,
        jitter=True,
    )
    # Overlay box plot summary
    sns.boxplot(
        data=df,
        x="persona_display",
        y="delta",
        order=order,
        ax=ax1,
        showfliers=False,
        boxprops=dict(facecolor="none", edgecolor="black"),
        whiskerprops=dict(color="black"),
        medianprops=dict(color="red", linewidth=2),
        capprops=dict(color="black"),
        width=0.3,
    )
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("")
    ax1.set_ylabel("Delta (persona - baseline)")
    ax1.set_title("Distribution of P(choose) deltas vs baseline")
    ax1.tick_params(axis="x", rotation=25)
    for label in ax1.get_xticklabels():
        label.set_ha("right")

    # Panel 2: heatmap of mean |delta| by topic x persona
    ax2 = axes[1]
    pivot = df.pivot_table(
        values="abs_delta", index="topic", columns="persona", aggfunc="mean"
    )
    pivot.columns = [PERSONA_DISPLAY[c] for c in pivot.columns]
    # Sort topics by overall mean
    pivot["_sort"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_sort", ascending=True)
    pivot = pivot.drop(columns="_sort")

    # Add task counts as row labels
    topic_counts = df[df["persona"] == df["persona"].iloc[0]].groupby("topic").size()
    new_index = [f"{t} (n={topic_counts.get(t, 0)})" for t in pivot.index]
    pivot.index = new_index

    sns.heatmap(
        pivot,
        ax=ax2,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "Mean |delta|"},
    )
    ax2.set_title("Mean |delta| by topic category and persona")
    ax2.set_ylabel("")
    ax2.set_xlabel("")

    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {PLOT_PATH}")
    plt.close(fig)


def main():
    results, topics = load_data()
    df = compute_deltas(results)
    print_summary(df, topics)
    make_plot(df, topics)


if __name__ == "__main__":
    main()
