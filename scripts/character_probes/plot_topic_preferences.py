"""Plot per-topic mean utilities for each character-trained persona.

Produces:
1. Heatmap of absolute topic utilities across all personas
2. Deviation heatmap (each persona minus the cross-persona mean)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results/experiments/character_probes")
TOPICS_PATH = Path("data/topics/topics_v1_12groups.json")
ASSETS_DIR = Path("experiments/character_probes/assets")

PERSONAS = [
    "goodness", "humor", "impulsiveness", "loving", "mathematical",
    "nonchalance", "poeticism", "remorse", "sarcasm", "sycophancy",
]

TOPIC_DISPLAY = {
    "math": "Math",
    "fiction": "Fiction",
    "persuasive_writing": "Persuasive Writing",
    "content_generation": "Content Generation",
    "coding": "Coding",
    "knowledge_qa": "Knowledge QA",
    "summarization": "Summarization",
    "sensitive_creative": "Sensitive Creative",
    "security_legal": "Security & Legal",
    "model_manipulation": "Model Manipulation",
    "harmful_request": "Harmful Request",
    "other": "Other",
}

PERSONA_DISPLAY = {
    "goodness": "Goodness",
    "humor": "Humor",
    "impulsiveness": "Impulsive",
    "loving": "Loving",
    "mathematical": "Mathematical",
    "nonchalance": "Nonchalant",
    "poeticism": "Poetic",
    "remorse": "Remorseful",
    "sarcasm": "Sarcastic",
    "sycophancy": "Sycophantic",
}


def load_topics() -> dict[str, str]:
    with open(TOPICS_PATH) as f:
        raw = json.load(f)
    return {
        task_id: list(models.values())[0]["primary"]
        for task_id, models in raw.items()
    }


def load_utilities(persona: str, split: str = "a") -> pd.DataFrame:
    split_dirs = list(
        (RESULTS_DIR / persona / "pre_task_active_learning").glob(f"*split_{split}_*")
    )
    assert len(split_dirs) == 1, f"Expected 1 split dir for {persona}/{split}, got {len(split_dirs)}"
    csvs = list(split_dirs[0].glob("thurstonian_*.csv"))
    assert len(csvs) == 1, f"Expected 1 CSV for {persona}/{split}, got {len(csvs)}"
    return pd.read_csv(csvs[0])


def compute_topic_means(persona: str, topics: dict[str, str]) -> pd.DataFrame:
    df = load_utilities(persona, split="a")
    df["topic"] = df["task_id"].map(topics)
    df = df.dropna(subset=["topic"])
    grouped = df.groupby("topic")["mu"].agg(["mean", "std", "count"]).reset_index()
    grouped.columns = ["topic", "mean_mu", "std_mu", "count"]
    grouped["sem"] = grouped["std_mu"] / np.sqrt(grouped["count"])
    return grouped


def plot_heatmap(
    pivot: pd.DataFrame,
    topic_order: list[str],
    title: str,
    cbar_label: str,
    path: Path,
    fmt: str = ".1f",
    topic_counts: dict[str, int] | None = None,
):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    vals = pivot.values
    vmax = np.nanmax(np.abs(vals))

    im = ax.imshow(vals, cmap=plt.cm.RdBu, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(PERSONAS)))
    ax.set_xticklabels(
        [PERSONA_DISPLAY[p] for p in PERSONAS],
        rotation=45, ha="right", fontsize=10,
    )

    ylabels = []
    for t in topic_order:
        name = TOPIC_DISPLAY.get(t, t)
        if topic_counts and t in topic_counts:
            name += f"  (n={topic_counts[t]})"
        ylabels.append(name)

    ax.set_yticks(range(len(topic_order)))
    ax.set_yticklabels(ylabels, fontsize=10)

    for i in range(len(topic_order)):
        for j in range(len(PERSONAS)):
            val = vals[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def main():
    topics = load_topics()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    all_means = {}
    for persona in PERSONAS:
        all_means[persona] = compute_topic_means(persona, topics)

    combined = pd.concat(
        [df.assign(persona=p) for p, df in all_means.items()],
        ignore_index=True,
    )
    topic_order = (
        combined.groupby("topic")["mean_mu"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    pivot = combined.pivot_table(
        index="topic", columns="persona", values="mean_mu"
    ).reindex(topic_order)
    pivot = pivot[[p for p in PERSONAS]]

    # Topic counts (from any persona — same tasks)
    sample = all_means[PERSONAS[0]]
    topic_counts = dict(zip(sample["topic"], sample["count"].astype(int)))

    # --- Plot 1: Absolute topic utilities ---
    plot_heatmap(
        pivot, topic_order,
        title="Which topics does each persona prefer to complete?",
        cbar_label="Mean utility (μ)  — higher = more preferred",
        path=ASSETS_DIR / "plot_031126_topic_preference_heatmap.png",
        topic_counts=topic_counts,
    )

    # --- Plot 2: Deviation from cross-persona mean ---
    base_profile = pivot.mean(axis=1)
    deviation = pivot.sub(base_profile, axis=0)

    plot_heatmap(
        deviation, topic_order,
        title="How does each persona differ from the average preference profile?",
        cbar_label="Deviation from cross-persona mean (Δμ)",
        path=ASSETS_DIR / "plot_031126_topic_preference_deviation.png",
        fmt="+.1f",
        topic_counts=topic_counts,
    )


if __name__ == "__main__":
    main()
