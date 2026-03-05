"""Select 50 heldout tasks for exp3 v8.

Pool constraints:
- Must have activations (activations_prompt_last.npz)
- Must have topic labels (topics.json)
- Must NOT be in probe training set (gemma3_10k_run1 Thurstonian CSV)
- Must NOT be in v7 task set
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ACTIVATIONS_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
TOPICS_PATH = Path("data/topics/topics.json")
THURSTONIAN_CSV = Path(
    "results/experiments/main_probes/gemma3_10k_run1/"
    "pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/"
    "thurstonian_80fa9dc8.csv"
)
V7_TASKS_PATH = Path("configs/ood/tasks/minimal_pairs_v7_tasks.json")
OUTPUT_PATH = Path("configs/ood/tasks/minimal_pairs_v8_tasks.json")

SEED = 42
N_TASKS = 50


def main():
    # Load activation task IDs
    act_data = np.load(ACTIVATIONS_PATH, allow_pickle=True)
    act_task_ids = set(act_data["task_ids"].tolist())
    print(f"Tasks with activations: {len(act_task_ids)}")

    # Load topic labels
    with open(TOPICS_PATH) as f:
        topics_data = json.load(f)
    topic_task_ids = set(topics_data.keys())
    print(f"Tasks with topic labels: {len(topic_task_ids)}")

    # Load probe training task IDs to exclude
    train_df = pd.read_csv(THURSTONIAN_CSV)
    train_task_ids = set(train_df["task_id"].tolist())
    print(f"Probe training tasks to exclude: {len(train_task_ids)}")

    # Load v7 task IDs to exclude
    with open(V7_TASKS_PATH) as f:
        v7_data = json.load(f)
    v7_task_ids = set(v7_data["task_ids"])
    print(f"V7 tasks to exclude: {len(v7_task_ids)}")

    # Pool: has activations & topics, minus training & v7
    exclude_ids = train_task_ids | v7_task_ids
    pool = (act_task_ids & topic_task_ids) - exclude_ids
    print(f"Eligible pool: {len(pool)}")

    # Get primary topic and origin for each task
    task_topics = {}
    for tid in pool:
        entry = topics_data[tid]
        model_key = list(entry.keys())[0]
        task_topics[tid] = entry[model_key]["primary"]

    # Detect origin from task ID prefix
    def get_origin(tid: str) -> str:
        if tid.startswith("alpaca_"):
            return "alpaca"
        if tid.startswith("wildchat_"):
            return "wildchat"
        if tid.startswith("competition_math_"):
            return "math"
        if tid.startswith("stresstest_"):
            return "stresstest"
        if tid.startswith("bailbench_"):
            return "bailbench"
        raise ValueError(f"Unknown origin for {tid}")

    # Group by origin
    origin_groups: dict[str, list[str]] = {}
    for tid in pool:
        origin = get_origin(tid)
        origin_groups.setdefault(origin, []).append(tid)

    print("\nOrigin distribution in pool:")
    for origin, tids in sorted(origin_groups.items(), key=lambda x: -len(x[1])):
        print(f"  {origin}: {len(tids)}")

    # Equal allocation per origin (floor division, extras to largest pools)
    rng = np.random.RandomState(SEED)
    n_origins = len(origin_groups)
    base = N_TASKS // n_origins
    extras = N_TASKS - base * n_origins

    sorted_origins = sorted(origin_groups.keys(), key=lambda o: -len(origin_groups[o]))
    allocations: dict[str, int] = {}
    for i, origin in enumerate(sorted_origins):
        allocations[origin] = base + (1 if i < extras else 0)
        allocations[origin] = min(allocations[origin], len(origin_groups[origin]))

    print(f"\nAllocations (total={sum(allocations.values())}):")
    selected = []
    for origin in sorted(allocations.keys()):
        n = allocations[origin]
        candidates = origin_groups[origin]
        sampled = rng.choice(candidates, size=min(n, len(candidates)), replace=False).tolist()
        selected.extend(sampled)
        print(f"  {origin}: {n} (from {len(candidates)})")

    selected.sort()
    print(f"\nSelected {len(selected)} tasks")

    # Verify no overlap
    overlap = set(selected) & exclude_ids
    assert len(overlap) == 0, f"Overlap with excluded tasks: {overlap}"

    output = {
        "task_ids": selected,
        "n_tasks": len(selected),
        "selection_method": "stratified by origin, excluding probe training + v7 tasks",
        "seed": SEED,
        "origin_allocations": allocations,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    # Print final distributions
    print("\nFinal origin distribution:")
    origin_counts: dict[str, int] = {}
    for tid in selected:
        o = get_origin(tid)
        origin_counts[o] = origin_counts.get(o, 0) + 1
    for origin, count in sorted(origin_counts.items(), key=lambda x: -x[1]):
        print(f"  {origin}: {count}")

    selected_topics = {tid: task_topics[tid] for tid in selected}
    topic_counts: dict[str, int] = {}
    for t in selected_topics.values():
        topic_counts[t] = topic_counts.get(t, 0) + 1
    print("\nFinal topic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count}")


if __name__ == "__main__":
    main()
