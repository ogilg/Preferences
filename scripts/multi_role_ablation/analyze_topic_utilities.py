import json
import pandas as pd
from pathlib import Path

from src.measurement.storage.loading import load_aligned_utilities

PERSONA_DIRS = {
    "no_prompt": Path("results/experiments/mra_persona1_noprompt/mra_persona1_noprompt/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0"),
    "villain": Path("results/experiments/mra_persona2_villain/mra_persona2_villain/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0"),
    "midwest": Path("results/experiments/mra_persona3_midwest/mra_persona3_midwest/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0"),
    "aesthete": Path("results/experiments/mra_persona4_aesthete/mra_persona4_aesthete/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0"),
}

TOPICS_PATH = "data/topics/topics.json"
TASK_IDS_PATH = "experiments/probe_generalization/multi_role_ablation/task_ids_all.txt"

# Load task IDs
with open(TASK_IDS_PATH) as f:
    task_ids = set(line.strip() for line in f if line.strip())

# Load utilities
df = load_aligned_utilities(PERSONA_DIRS)
df = df[df.index.isin(task_ids)]
print(f"Tasks with all 4 personas: {len(df)}")
print(f"\n=== Overall correlations ===")
print(df.corr().round(3))

# Load topics
with open(TOPICS_PATH) as f:
    topics_raw = json.load(f)

topic_map = {}
for tid, models in topics_raw.items():
    for model_name, cats in models.items():
        topic_map[tid] = cats["primary"]

df["topic"] = df.index.map(lambda x: topic_map.get(x, "unknown"))
print(f"\nTasks with topic labels: {(df['topic'] != 'unknown').sum()}")
print(f"Tasks missing topic: {(df['topic'] == 'unknown').sum()}")

# Per-topic mean utilities
print(f"\n=== Mean utility by topic (sorted by no_prompt) ===")
topic_means = df.groupby("topic")[["no_prompt", "villain", "midwest", "aesthete"]].mean()
topic_counts = df.groupby("topic").size().rename("n")
topic_summary = topic_means.join(topic_counts).sort_values("no_prompt", ascending=False)
print(topic_summary.round(2).to_string())

# Per-topic: where do personas DISAGREE most?
print(f"\n=== Topic-level disagreements (largest rank changes) ===")
# For each topic, compute villain_rank - no_prompt_rank
topic_ranks = topic_means.rank(ascending=False)
topic_ranks.columns = [f"{c}_rank" for c in topic_ranks.columns]
disagreements = pd.DataFrame({
    "no_prompt_mean": topic_means["no_prompt"],
    "villain_mean": topic_means["villain"],
    "midwest_mean": topic_means["midwest"],
    "aesthete_mean": topic_means["aesthete"],
    "villain_minus_noprompt": topic_means["villain"] - topic_means["no_prompt"],
    "aesthete_minus_noprompt": topic_means["aesthete"] - topic_means["no_prompt"],
    "midwest_minus_noprompt": topic_means["midwest"] - topic_means["no_prompt"],
    "n": topic_counts,
})
print(disagreements.sort_values("villain_minus_noprompt", ascending=False).round(2).to_string())

# Overall: how much variance is there across personas?
print(f"\n=== Per-topic std across personas (where do personas disagree most?) ===")
topic_std = topic_means.std(axis=1).sort_values(ascending=False)
for topic, std in topic_std.items():
    means = topic_means.loc[topic]
    n = topic_counts[topic]
    print(f"  {topic:30s} std={std:.2f}  n={n:4d}  means: {means.round(2).to_dict()}")

# Task-level: show some tasks with largest persona disagreements
print(f"\n=== Individual tasks with largest villain vs no_prompt disagreement ===")
df["villain_minus_noprompt"] = df["villain"] - df["no_prompt"]
extremes = pd.concat([
    df.nlargest(5, "villain_minus_noprompt")[["no_prompt", "villain", "midwest", "aesthete", "villain_minus_noprompt", "topic"]],
    df.nsmallest(5, "villain_minus_noprompt")[["no_prompt", "villain", "midwest", "aesthete", "villain_minus_noprompt", "topic"]],
])
print(extremes.round(2).to_string())

# Within-topic correlations
print(f"\n=== Within-topic correlations (no_prompt vs villain) ===")
for topic in topic_means.index:
    subset = df[df["topic"] == topic]
    if len(subset) >= 10:
        r = subset["no_prompt"].corr(subset["villain"])
        print(f"  {topic:30s} n={len(subset):4d}  r={r:.3f}")
