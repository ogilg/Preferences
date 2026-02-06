"""Run topic classification on tasks.

Two modes:
1. From ranked_tasks JSON (tasks already exported from an experiment):
    python -m src.analysis.topic_classification.run_classification \
        --experiment-id gemma3_revealed_v1

2. From raw task data (all tasks in a dataset):
    python -m src.analysis.topic_classification.run_classification \
        --origins wildchat alpaca math bailbench stress_test --n-tasks 2000

Pass --reclassify-other to re-run discovery+classification on "other" tasks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.analysis.topic_classification.classify import (
    classify_tasks_batch,
    discover_categories,
    load_cache,
    save_cache,
)

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"
DISCOVERY_SAMPLE_SIZE = 300


def load_tasks_from_ranked(experiment_id: str, run_name: str | None = None) -> list[dict]:
    """Load tasks from a ranked_tasks JSON file."""
    from src.analysis.active_learning.utils import load_ranked_tasks
    ranked = load_ranked_tasks(experiment_id, run_name)
    return [{"task_id": t["task_id"], "prompt": t["prompt"]} for t in ranked]


def load_tasks_from_origins(origins: list[str], n_tasks: int, seed: int) -> list[dict]:
    """Load tasks directly from task data."""
    from src.task_data import load_tasks, parse_origins
    origin_enums = parse_origins(origins)
    tasks = load_tasks(n=n_tasks, origins=origin_enums, seed=seed, stratified=True)
    return [{"task_id": t.id, "prompt": t.prompt} for t in tasks]


def cache_path_for(experiment_id: str | None, label: str = "topics") -> Path:
    if experiment_id:
        return OUTPUT_DIR / experiment_id / f"{label}.json"
    return OUTPUT_DIR / f"{label}.json"


def categories_path_for(experiment_id: str | None) -> Path:
    if experiment_id:
        return OUTPUT_DIR / experiment_id / "categories.json"
    return OUTPUT_DIR / "categories.json"


async def run_discovery(tasks: list[dict], n_sample: int, seed: int) -> list[str]:
    """Sample tasks and discover categories."""
    rng = np.random.default_rng(seed)
    sample_size = min(n_sample, len(tasks))
    indices = rng.choice(len(tasks), size=sample_size, replace=False)
    sample_prompts = [tasks[i]["prompt"] for i in indices]

    print(f"\nDiscovering categories from {sample_size} sampled tasks...")
    categories = await discover_categories(sample_prompts)
    print(f"Discovered {len(categories)} categories:")
    for cat in categories:
        print(f"  - {cat}")

    return categories


async def run_classification(
    tasks: list[dict],
    categories: list[str],
    cache_path: Path,
    max_concurrent: int,
) -> dict[str, str]:
    """Classify all tasks into categories."""
    cache = load_cache(cache_path)
    print(f"\nClassifying {len(tasks)} tasks into {len(categories)} categories...")

    cache = await classify_tasks_batch(tasks, categories, cache, max_concurrent)
    save_cache(cache, cache_path)
    print(f"Cache saved to {cache_path}")

    return cache


def print_distribution(cache: dict[str, str], task_ids: set[str]) -> None:
    """Print category distribution for the given task IDs."""
    relevant = {tid: cat for tid, cat in cache.items() if tid in task_ids}
    counts = Counter(relevant.values())

    print(f"\nCategory distribution ({len(relevant)} tasks):")
    for cat, count in counts.most_common():
        pct = count / len(relevant) * 100
        print(f"  {cat:<30} {count:>5} ({pct:5.1f}%)")

    missing = task_ids - set(relevant.keys())
    if missing:
        print(f"  {'(unclassified)':<30} {len(missing):>5}")


async def reclassify_other(
    tasks: list[dict],
    cache: dict[str, str],
    original_categories: list[str],
    c_path: Path,
    cat_path: Path,
    discovery_sample_size: int,
    seed: int,
    max_concurrent: int,
) -> dict[str, str]:
    """Re-discover categories from 'other' tasks and reclassify them."""
    other_tasks = [t for t in tasks if cache.get(t["task_id"]) == "other"]
    if not other_tasks:
        print("\nNo 'other' tasks to reclassify.")
        return cache

    print(f"\n--- Reclassifying {len(other_tasks)} 'other' tasks ---")

    sub_categories = await run_discovery(other_tasks, discovery_sample_size, seed)

    # Avoid duplicating existing categories
    sub_categories = [c for c in sub_categories if c not in original_categories]
    if not sub_categories:
        print("No new categories discovered from 'other' tasks.")
        return cache

    print(f"\nNew sub-categories (not in original): {sub_categories}")

    # Remove 'other' entries so they get reclassified
    for t in other_tasks:
        cache.pop(t["task_id"], None)

    all_categories = original_categories + sub_categories
    cache = await classify_tasks_batch(other_tasks, all_categories, cache, max_concurrent)
    save_cache(cache, c_path)

    # Save updated categories
    with open(cat_path, "w") as f:
        json.dump(all_categories, f, indent=2)
    print(f"Updated categories saved to {cat_path}")

    return cache


async def main():
    parser = argparse.ArgumentParser(description="Run topic classification on tasks")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment-id", type=str, help="Load from ranked_tasks export")
    source.add_argument("--origins", nargs="+", help="Load from raw task data")

    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--n-tasks", type=int, default=5000, help="Max tasks to load from origins")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discovery-sample-size", type=int, default=DISCOVERY_SAMPLE_SIZE)
    parser.add_argument("--max-concurrent", type=int, default=30)
    parser.add_argument("--reclassify-other", action="store_true",
                        help="Re-run discovery on 'other' tasks")
    parser.add_argument("--categories-file", type=str, default=None,
                        help="Path to JSON file with pre-defined categories (skip discovery)")
    args = parser.parse_args()

    # Load tasks
    if args.experiment_id:
        tasks = load_tasks_from_ranked(args.experiment_id, args.run_name)
        label = args.experiment_id
    else:
        tasks = load_tasks_from_origins(args.origins, args.n_tasks, args.seed)
        label = None

    print(f"Loaded {len(tasks)} tasks")
    task_ids = {t["task_id"] for t in tasks}

    c_path = cache_path_for(label)
    cat_path = categories_path_for(label)

    # Discover or load categories
    if args.categories_file:
        with open(args.categories_file) as f:
            categories = json.load(f)
        print(f"Loaded {len(categories)} categories from {args.categories_file}")
    elif cat_path.exists():
        with open(cat_path) as f:
            categories = json.load(f)
        print(f"Loaded {len(categories)} existing categories from {cat_path}")
    else:
        categories = await run_discovery(tasks, args.discovery_sample_size, args.seed)
        cat_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cat_path, "w") as f:
            json.dump(categories, f, indent=2)
        print(f"Categories saved to {cat_path}")

    # Classify
    cache = await run_classification(tasks, categories, c_path, args.max_concurrent)
    print_distribution(cache, task_ids)

    # Reclassify "other" if requested
    if args.reclassify_other:
        cache = await reclassify_other(
            tasks, cache, categories, c_path, cat_path,
            args.discovery_sample_size, args.seed, args.max_concurrent,
        )
        print_distribution(cache, task_ids)


if __name__ == "__main__":
    asyncio.run(main())
