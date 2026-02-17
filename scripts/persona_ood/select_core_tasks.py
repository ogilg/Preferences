"""Select ~100 core tasks stratified by topic and utility for persona OOD v2.

Force-includes the 10 Part B target tasks, then stratified-samples ~90 more.
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src.probes.data_loading import load_thurstonian_scores

RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
TOPICS_PATH = Path("src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json")
OUTPUT_PATH = Path("experiments/probe_generalization/persona_ood/core_tasks.json")

FORCE_INCLUDE = {
    "wildchat_39653",   # organ_enthusiast target
    "alpaca_7766",      # horror_fanatic target
    "wildchat_14416",   # chess_programming_lover target
    "wildchat_11393",   # spongebob_superfan target
    "alpaca_2494",      # polynomial_enthusiast target
    "wildchat_63216",   # dune_lore_master target
    "alpaca_14046",     # sql_devotee target
    "alpaca_12314",     # witch_trials_scholar target
    "wildchat_48235",   # doctor_who_fan target
    "alpaca_10324",     # wildlife_conservation_storyteller target
}

TARGET_N = 100

# Load data
scores = load_thurstonian_scores(RUN_DIR)
print(f"Loaded {len(scores)} Thurstonian scores")

with open(TOPICS_PATH) as f:
    topics_raw = json.load(f)

topics: dict[str, str] = {}
for task_id, model_dict in topics_raw.items():
    for model_name, classification in model_dict.items():
        topics[task_id] = classification["primary"]
        break

print(f"Loaded {len(topics)} topic classifications")

task_ids_with_both = set(scores.keys()) & set(topics.keys())
print(f"Tasks with both score and topic: {len(task_ids_with_both)}")

# Verify force-includes exist
missing = FORCE_INCLUDE - task_ids_with_both
if missing:
    raise ValueError(f"Force-include tasks not found: {missing}")

# Start with force-included tasks
selected_ids = set(FORCE_INCLUDE)
remaining_budget = TARGET_N - len(selected_ids)

# Group remaining tasks by topic (excluding force-included)
topic_groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
for tid in task_ids_with_both:
    if tid not in selected_ids:
        topic_groups[topics[tid]].append((tid, scores[tid]))

print(f"\nForce-included: {len(selected_ids)} tasks")
print(f"Remaining budget: {remaining_budget}")

# Count how many force-included per topic
force_topics = defaultdict(int)
for tid in FORCE_INCLUDE:
    force_topics[topics[tid]] += 1
print(f"Force-included by topic: {dict(force_topics)}")

# Stratified selection of remaining ~90
rng = np.random.RandomState(42)
total_available = sum(len(tasks) for tasks in topic_groups.values())

for topic, tasks in topic_groups.items():
    n_for_topic = max(1, round(len(tasks) / total_available * remaining_budget))
    n_for_topic = min(n_for_topic, len(tasks))

    tasks_sorted = sorted(tasks, key=lambda x: x[1])
    tercile_size = len(tasks_sorted) // 3

    if tercile_size == 0:
        selected_ids.update(tid for tid, _ in tasks_sorted[:n_for_topic])
        continue

    per_tercile = n_for_topic // 3
    remainder = n_for_topic - 3 * per_tercile

    terciles = [
        tasks_sorted[:tercile_size],
        tasks_sorted[tercile_size:2*tercile_size],
        tasks_sorted[2*tercile_size:],
    ]

    for i, tercile in enumerate(terciles):
        n_sample = per_tercile + (1 if i < remainder else 0)
        n_sample = min(n_sample, len(tercile))
        indices = rng.choice(len(tercile), size=n_sample, replace=False)
        for idx in indices:
            selected_ids.add(tercile[idx][0])

print(f"\nSelected {len(selected_ids)} tasks")

# Verify distribution
selected_topics = defaultdict(int)
selected_mus = []
for tid in selected_ids:
    selected_topics[topics[tid]] += 1
    selected_mus.append(scores[tid])

print(f"\nSelected task distribution:")
for topic, count in sorted(selected_topics.items(), key=lambda x: -x[1]):
    print(f"  {topic}: {count}")

print(f"\nUtility range: [{min(selected_mus):.2f}, {max(selected_mus):.2f}]")
print(f"Utility mean: {np.mean(selected_mus):.2f}, std: {np.std(selected_mus):.2f}")

# Save
output = {
    "task_ids": sorted(selected_ids),
    "n_tasks": len(selected_ids),
    "selection_method": "10 force-included Part B targets + stratified sample by topic and utility tercile",
    "seed": 42,
}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {OUTPUT_PATH}")
