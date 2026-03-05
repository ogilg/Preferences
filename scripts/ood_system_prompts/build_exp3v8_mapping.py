"""Build pairwise mapping for exp3 v8.

50 tasks → 1225 unique pairs × 121 conditions (120 + baseline) = 148,225 triples.
"""

import json
from itertools import combinations
from pathlib import Path

TASKS_PATH = Path("configs/ood/tasks/minimal_pairs_v8_tasks.json")
PROMPTS_PATH = Path("configs/ood/prompts/minimal_pairs_v8.json")
OUTPUT_PATH = Path("configs/ood/mappings/minimal_pairs_v8.json")


def main():
    with open(TASKS_PATH) as f:
        task_ids = json.load(f)["task_ids"]

    with open(PROMPTS_PATH) as f:
        prompts_data = json.load(f)

    condition_ids = ["baseline"] + [c["condition_id"] for c in prompts_data["conditions"]]

    # All unique ordered pairs
    task_pairs = list(combinations(sorted(task_ids), 2))
    print(f"Tasks: {len(task_ids)}")
    print(f"Task pairs: {len(task_pairs)}")
    print(f"Conditions: {len(condition_ids)}")

    pairs = []
    for cid in condition_ids:
        for task_a, task_b in task_pairs:
            pairs.append({
                "condition_id": cid,
                "task_a": task_a,
                "task_b": task_b,
            })

    print(f"Total triples: {len(pairs)}")

    output = {"pairs": pairs}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
