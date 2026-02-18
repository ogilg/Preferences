"""Sample 50 tasks for phase 3: stratified by topic, excluding phase 2 tasks.

Pool constraints:
- Must be in topics_v2.json (has topic labels)
- Must be in activations_prompt_last.npz (has no-prompt activations)
- Must NOT be in phase 2 core_tasks.json (the old 101)
"""

import json
from pathlib import Path

import numpy as np

TOPICS_PATH = Path("src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json")
ACTIVATIONS_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
EXCLUDE_PATH = Path("experiments/probe_generalization/persona_ood/core_tasks.json")
OUTPUT_PATH = Path("experiments/probe_generalization/persona_ood/phase3/core_tasks.json")

SEED = 42
N_TASKS = 50


def main():
    with open(TOPICS_PATH) as f:
        topics_data = json.load(f)
    topic_task_ids = set(topics_data.keys())
    print(f"Tasks with topic labels: {len(topic_task_ids)}")

    # Load activation task IDs (just need the task_ids array)
    act_data = np.load(ACTIVATIONS_PATH, allow_pickle=True)
    act_task_ids = set(act_data["task_ids"].tolist())
    print(f"Tasks with activations: {len(act_task_ids)}")

    with open(EXCLUDE_PATH) as f:
        exclude_data = json.load(f)
    exclude_ids = set(exclude_data["task_ids"])
    # Also exclude anchor tasks if present
    if "anchor_task_ids" in exclude_data:
        exclude_ids |= set(exclude_data["anchor_task_ids"])
    print(f"Tasks to exclude (phase 2): {len(exclude_ids)}")

    # Pool: intersection of topics and activations, minus exclusions
    pool = topic_task_ids & act_task_ids - exclude_ids
    print(f"Eligible pool: {len(pool)}")

    # Get topic for each task (use primary topic from first model)
    task_topics = {}
    for tid in pool:
        entry = topics_data[tid]
        model_key = list(entry.keys())[0]
        task_topics[tid] = entry[model_key]["primary"]

    # Group by topic
    topic_groups: dict[str, list[str]] = {}
    for tid, topic in task_topics.items():
        topic_groups.setdefault(topic, []).append(tid)

    print("\nTopic distribution in pool:")
    for topic, tids in sorted(topic_groups.items(), key=lambda x: -len(x[1])):
        print(f"  {topic}: {len(tids)}")

    # Stratified sampling: proportional to pool size, minimum 1 per topic
    rng = np.random.RandomState(SEED)
    total_pool = sum(len(tids) for tids in topic_groups.values())

    # Allocate proportionally
    allocations: dict[str, int] = {}
    for topic, tids in topic_groups.items():
        allocations[topic] = max(1, round(len(tids) / total_pool * N_TASKS))

    # Adjust to hit exactly N_TASKS
    while sum(allocations.values()) > N_TASKS:
        # Remove from largest allocation
        largest = max(allocations, key=lambda t: allocations[t])
        allocations[largest] -= 1
    while sum(allocations.values()) < N_TASKS:
        # Add to largest pool that's underrepresented
        largest = max(topic_groups, key=lambda t: len(topic_groups[t]) / max(allocations.get(t, 1), 1))
        allocations[largest] = allocations.get(largest, 0) + 1

    print(f"\nAllocations (total={sum(allocations.values())}):")
    selected = []
    for topic in sorted(allocations.keys()):
        n = allocations[topic]
        candidates = topic_groups[topic]
        sampled = rng.choice(candidates, size=min(n, len(candidates)), replace=False).tolist()
        selected.extend(sampled)
        print(f"  {topic}: {n} (from {len(candidates)})")

    selected.sort()
    print(f"\nSelected {len(selected)} tasks")

    # Verify no overlap with excluded
    overlap = set(selected) & exclude_ids
    assert len(overlap) == 0, f"Overlap with excluded tasks: {overlap}"

    # Verify all in activations
    missing_act = set(selected) - act_task_ids
    assert len(missing_act) == 0, f"Missing activations: {missing_act}"

    output = {
        "task_ids": selected,
        "n_tasks": len(selected),
        "selection_method": "stratified sample by topic, excluding phase 2 tasks",
        "seed": SEED,
        "topic_allocations": allocations,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    # Print topic distribution of selected tasks
    selected_topics = {tid: task_topics[tid] for tid in selected}
    topic_counts: dict[str, int] = {}
    for t in selected_topics.values():
        topic_counts[t] = topic_counts.get(t, 0) + 1
    print("\nFinal topic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count}")


if __name__ == "__main__":
    main()
