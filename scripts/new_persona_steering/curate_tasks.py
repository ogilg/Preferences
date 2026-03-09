"""Curate task set and pairs for persona steering v2.

Selects 10 tasks per category (5 categories), then samples 10 cross-category
pairs per pair type (10 pair types = 5 choose 2), totalling 100 fixed pairs.
"""

import json
import itertools
from pathlib import Path

import numpy as np

from dotenv import load_dotenv
from src.task_data import load_tasks, OriginDataset

load_dotenv()

TOPICS_PATH = Path("data/topics/topics.json")
OUTPUT_DIR = Path("experiments/new_persona_steering/artifacts")
SEED = 42
N_PER_CATEGORY = 3

# Map experiment categories to topic classification primary labels
CATEGORY_TOPIC_MAP = {
    "harmful": "harmful_request",
    "creative": "fiction",
    "math": "math",
    "value_conflict": "value_conflict",
    "knowledge_qa": "knowledge_qa",
}


def main():
    with open(TOPICS_PATH) as f:
        topics_data = json.load(f)

    # Load all tasks from all origins
    all_origins = [
        OriginDataset.WILDCHAT,
        OriginDataset.ALPACA,
        OriginDataset.MATH,
        OriginDataset.BAILBENCH,
        OriginDataset.STRESS_TEST,
    ]
    tasks = load_tasks(n=50000, origins=all_origins, seed=None)
    print(f"Loaded {len(tasks)} total tasks")

    # Build id -> task lookup
    task_by_id = {t.id: t for t in tasks}

    # Get primary topic for each task that has a classification
    task_topics: dict[str, str] = {}
    for tid in task_by_id:
        if tid in topics_data:
            entry = topics_data[tid]
            model_key = list(entry.keys())[0]
            task_topics[tid] = entry[model_key]["primary"]

    print(f"Tasks with topic labels: {len(task_topics)}")

    # Sample tasks per category
    rng = np.random.RandomState(SEED)
    selected: dict[str, list[str]] = {}

    for category, topic_label in CATEGORY_TOPIC_MAP.items():
        candidates = [tid for tid, topic in task_topics.items() if topic == topic_label]
        print(f"{category} ({topic_label}): {len(candidates)} candidates")
        assert len(candidates) >= N_PER_CATEGORY, (
            f"Not enough {category} tasks: {len(candidates)} < {N_PER_CATEGORY}"
        )
        sampled = rng.choice(candidates, size=N_PER_CATEGORY, replace=False).tolist()
        selected[category] = sorted(sampled)

    # Build task_set.json
    task_set = {
        "categories": {},
        "seed": SEED,
        "n_per_category": N_PER_CATEGORY,
    }
    for category, ids in selected.items():
        task_set["categories"][category] = {
            "topic": CATEGORY_TOPIC_MAP[category],
            "ids": ids,
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "task_set.json", "w") as f:
        json.dump(task_set, f, indent=2)
    print(f"\nSaved task_set.json ({sum(len(v) for v in selected.values())} tasks)")

    # Build pairs: 10 per pair type
    categories = list(CATEGORY_TOPIC_MAP.keys())
    pair_types = list(itertools.combinations(categories, 2))
    print(f"\n{len(pair_types)} pair types:")

    all_pairs = []
    for cat_a, cat_b in pair_types:
        ids_a = selected[cat_a]
        ids_b = selected[cat_b]
        # All possible cross-category pairings
        all_pairings = [(a, b) for a in ids_a for b in ids_b]
        for task_a, task_b in all_pairings:
            all_pairs.append({
                "task_a": task_a,
                "task_b": task_b,
                "category_a": cat_a,
                "category_b": cat_b,
            })
        print(f"  {cat_a} × {cat_b}: {len(all_pairings)} pairs")

    pairs_output = {
        "pairs": all_pairs,
        "n_pairs": len(all_pairs),
        "n_per_pair_type": N_PER_CATEGORY ** 2,
        "seed": SEED,
    }

    with open(OUTPUT_DIR / "task_pairs.json", "w") as f:
        json.dump(pairs_output, f, indent=2)
    print(f"\nSaved task_pairs.json ({len(all_pairs)} pairs)")

    # Print sample tasks for review
    print("\n--- Sample tasks per category ---")
    for category, ids in selected.items():
        print(f"\n{category}:")
        for tid in ids[:3]:
            task = task_by_id[tid]
            prompt_preview = task.prompt[:100].replace("\n", " ")
            print(f"  {tid}: {prompt_preview}...")


if __name__ == "__main__":
    main()
