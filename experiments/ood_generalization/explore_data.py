"""Explore Thurstonian utilities and topic classifications to select target tasks."""
import json
import pandas as pd
from collections import Counter
from pathlib import Path

THURSTONIAN_PATH = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_a1ebd06e.csv")
TOPICS_PATH = Path("src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json")

# Load data
df = pd.read_csv(THURSTONIAN_PATH)
with open(TOPICS_PATH) as f:
    topics_raw = json.load(f)

# Extract primary topics
topics = {
    tid: list(models.values())[0]["primary"]
    for tid, models in topics_raw.items()
}

# Merge
df["topic"] = df["task_id"].map(topics)
classified = df[df["topic"].notna()]
unclassified = df[df["topic"].isna()]

print(f"Total tasks: {len(df)}")
print(f"With topic classification: {len(classified)}")
print(f"Without: {len(unclassified)}")
print()

# Topic distribution
topic_counts = Counter(classified["topic"])
print("=== Topic Distribution ===")
for topic, count in topic_counts.most_common():
    subset = classified[classified["topic"] == topic]
    print(f"  {topic:25s}: n={count:3d}  mu=[{subset['mu'].min():.1f}, {subset['mu'].max():.1f}]  mean={subset['mu'].mean():.1f}")

print()
print("=== Sample Tasks per Topic (median utility) ===")
for topic in sorted(topic_counts.keys()):
    subset = classified[classified["topic"] == topic].sort_values("mu")
    if len(subset) < 5:
        continue
    # Pick tasks at 25th, 50th, 75th percentile
    indices = [len(subset)//4, len(subset)//2, 3*len(subset)//4]
    print(f"\n--- {topic} (n={len(subset)}) ---")
    for i in indices:
        row = subset.iloc[i]
        print(f"  {row['task_id']:40s}  mu={row['mu']:.2f}  sigma={row['sigma']:.2f}")

# Also show overall utility distribution
print()
print("=== Overall Utility Distribution ===")
print(df["mu"].describe())
