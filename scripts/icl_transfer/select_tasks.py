"""Select 50 tasks balanced across topics for ICL transfer experiment.

Outputs a JSON file compatible with custom_tasks_file in the runner config.

Run: python -m scripts.icl_transfer.select_tasks
"""

import json
import random
from collections import Counter
from pathlib import Path

from src.task_data import load_filtered_tasks, OriginDataset

N_TASKS = 50
SEED = 42
TOPICS_PATH = "data/topics/topics.json"
OUTPUT_PATH = Path("configs/icl_transfer/icl_50_tasks.json")


def get_topic(task_id: str, topics: dict) -> str:
    for _, cats in topics[task_id].items():
        return cats["primary"]


def main():
    with open(TOPICS_PATH) as f:
        topics = json.load(f)

    all_tasks = load_filtered_tasks(
        n=None,
        origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH,
                 OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST],
    )
    task_lookup = {t.id: t for t in all_tasks}
    available_ids = set(task_lookup.keys()) & set(topics.keys())
    print(f"Tasks with topics: {len(available_ids)}")

    # Group by topic
    by_topic: dict[str, list[str]] = {}
    for tid in available_ids:
        by_topic.setdefault(get_topic(tid, topics), []).append(tid)

    rng = random.Random(SEED)
    for t in by_topic:
        by_topic[t].sort()
        rng.shuffle(by_topic[t])

    # Round-robin across topics
    selected: list[str] = []
    cycle = sorted(by_topic.keys())
    idx = {t: 0 for t in cycle}
    while len(selected) < N_TASKS:
        added_any = False
        for t in cycle:
            if idx[t] < len(by_topic[t]) and len(selected) < N_TASKS:
                selected.append(by_topic[t][idx[t]])
                idx[t] += 1
                added_any = True
        if not added_any:
            break

    topic_dist = Counter(get_topic(tid, topics) for tid in selected)
    print(f"Selected {len(selected)} tasks:")
    for t, c in topic_dist.most_common():
        print(f"  {t}: {c}")

    # Write in custom_tasks_file format
    output = [
        {"task_id": tid, "prompt": task_lookup[tid].prompt, "topic": get_topic(tid, topics)}
        for tid in selected
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
