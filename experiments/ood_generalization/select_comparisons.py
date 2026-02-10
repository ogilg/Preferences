"""Select comparison tasks for each target task.

For each target, select ~50 comparison tasks that are:
- Close in utility (within ±3 of target mu for tight signal)
- From different topics than the target
- Diverse across topics (max 8 per non-target category)
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

THURSTONIAN_PATH = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_a1ebd06e.csv")
TOPICS_PATH = Path("src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json")
TARGETS_PATH = Path("experiments/ood_generalization/target_tasks.json")
# Tasks with existing activations
ACTIVATIONS_PATH = Path("activations/gemma_3_27b/completions_with_activations.json")

df = pd.read_csv(THURSTONIAN_PATH)
with open(TOPICS_PATH) as f:
    topics_raw = json.load(f)
topics = {
    tid: list(models.values())[0]["primary"]
    for tid, models in topics_raw.items()
}
df["topic"] = df["task_id"].map(topics)

with open(TARGETS_PATH) as f:
    targets = json.load(f)

# Get task IDs that have activations (for later probe scoring)
with open(ACTIVATIONS_PATH) as f:
    completions = json.load(f)
activation_ids = {c["task_id"] for c in completions}

target_ids = {t["task_id"] for t in targets}
rng = np.random.default_rng(42)

UTILITY_WINDOW = 3.0
MAX_PER_CATEGORY = 8
N_COMPARISONS = 50

all_comparisons = {}

for target in targets:
    tid = target["task_id"]
    tmu = target["mu"]
    ttopic = target["topic"]

    # Filter: different topic, close in utility, has topic classification, has activations
    candidates = df[
        (df["topic"].notna()) &
        (df["topic"] != ttopic) &
        (~df["task_id"].isin(target_ids)) &
        (df["task_id"].isin(activation_ids)) &
        (abs(df["mu"] - tmu) <= UTILITY_WINDOW)
    ].copy()

    # If not enough in window, widen
    if len(candidates) < N_COMPARISONS:
        candidates = df[
            (df["topic"].notna()) &
            (df["topic"] != ttopic) &
            (~df["task_id"].isin(target_ids)) &
            (df["task_id"].isin(activation_ids)) &
            (abs(df["mu"] - tmu) <= 5.0)
        ].copy()

    # Sample with category cap
    selected = []
    topic_counts = {}
    # Sort by closeness to target utility
    candidates["dist"] = abs(candidates["mu"] - tmu)
    candidates = candidates.sort_values("dist")

    for _, row in candidates.iterrows():
        cat = row["topic"]
        if topic_counts.get(cat, 0) >= MAX_PER_CATEGORY:
            continue
        selected.append(row["task_id"])
        topic_counts[cat] = topic_counts.get(cat, 0) + 1
        if len(selected) >= N_COMPARISONS:
            break

    all_comparisons[tid] = selected
    topic_summary = {k: v for k, v in sorted(topic_counts.items(), key=lambda x: -x[1])}
    print(f"{ttopic:20s} ({tid}): {len(selected)} comparisons, topics: {topic_summary}")

output = Path("experiments/ood_generalization/comparison_tasks.json")
with open(output, "w") as f:
    json.dump(all_comparisons, f, indent=2)
print(f"\nSaved to {output}")
