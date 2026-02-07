"""Audit topic categories by classifying a sample and showing examples per category."""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.analysis.topic_classification.classify import (
    classify_tasks_batch,
    load_cache,
    save_cache,
)
from src.analysis.topic_classification.run_classification import (
    load_tasks_from_origins,
)

load_dotenv()

OUTPUT_DIR = Path(__file__).parent.parent / "src" / "analysis" / "topic_classification" / "output"
CATEGORIES_PATH = OUTPUT_DIR / "categories.json"
AUDIT_CACHE_PATH = OUTPUT_DIR / "audit_topics.json"

SEED = 42
N_TASKS = 5000
SAMPLE_SIZE = 300
MAX_PROMPT_CHARS = 200
EXAMPLES_PER_CATEGORY = 5


async def main():
    with open(CATEGORIES_PATH) as f:
        categories = json.load(f)

    tasks = load_tasks_from_origins(
        ["wildchat", "alpaca", "math", "bailbench", "stress_test"],
        n_tasks=N_TASKS,
        seed=SEED,
    )
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(tasks), size=min(SAMPLE_SIZE, len(tasks)), replace=False)
    sample = [tasks[i] for i in indices]

    print(f"Classifying {len(sample)} tasks into {len(categories)} categories...\n")

    cache = load_cache(AUDIT_CACHE_PATH)
    cache = await classify_tasks_batch(sample, categories, cache, max_concurrent=30)
    save_cache(cache, AUDIT_CACHE_PATH)

    # Group by primary category
    by_category: dict[str, list[dict]] = defaultdict(list)
    for task in sample:
        entry = cache.get(task["task_id"])
        if entry is None:
            by_category["UNCLASSIFIED"].append(task)
        else:
            by_category[entry["primary"]].append(task)

    # Print examples with secondary category shown
    for cat in sorted(by_category.keys(), key=lambda c: -len(by_category[c])):
        tasks_in_cat = by_category[cat]
        print(f"\n{'='*80}")
        print(f"  {cat.upper()} — {len(tasks_in_cat)} tasks")
        print(f"{'='*80}")
        for task in tasks_in_cat[:EXAMPLES_PER_CATEGORY]:
            prompt_preview = task["prompt"][:MAX_PROMPT_CHARS].replace("\n", " ")
            if len(task["prompt"]) > MAX_PROMPT_CHARS:
                prompt_preview += "..."
            entry = cache.get(task["task_id"], {})
            secondary = entry.get("secondary", "?")
            sec_label = f" (2nd: {secondary})" if secondary != cat else ""
            print(f"  [{task['task_id'][:12]}]{sec_label} {prompt_preview}")
        if len(tasks_in_cat) > EXAMPLES_PER_CATEGORY:
            print(f"  ... and {len(tasks_in_cat) - EXAMPLES_PER_CATEGORY} more")

    # Print secondary category co-occurrence
    print(f"\n{'='*80}")
    print("  SECONDARY CATEGORY BREAKDOWN")
    print(f"{'='*80}")
    for cat in sorted(by_category.keys(), key=lambda c: -len(by_category[c])):
        if cat == "UNCLASSIFIED":
            continue
        tasks_in_cat = by_category[cat]
        secondaries = Counter(
            cache[t["task_id"]]["secondary"]
            for t in tasks_in_cat
            if t["task_id"] in cache
        )
        sec_str = ", ".join(
            f"{s}:{n}" for s, n in secondaries.most_common(5)
        )
        print(f"  {cat:<30} -> {sec_str}")


if __name__ == "__main__":
    asyncio.run(main())
