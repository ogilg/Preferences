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

from src.analysis.active_learning.utils import load_ranked_tasks
from src.analysis.topic_classification.classify import (
    MODELS,
    OUTPUT_DIR,
    Cache,
    TaskInput,
    classify_tasks_batch,
    discover_categories,
    load_cache,
    save_cache,
)
from src.measurement.storage.loading import get_activation_task_ids
from src.task_data import load_filtered_tasks, parse_origins

load_dotenv()

DISCOVERY_SAMPLE_SIZE = 300


def load_tasks_from_ranked(experiment_id: str, run_name: str | None = None) -> list[TaskInput]:
    ranked = load_ranked_tasks(experiment_id, run_name)
    return [{"task_id": t["task_id"], "prompt": t["prompt"]} for t in ranked]


def load_tasks_from_origins(
    origins: list[str],
    n_tasks: int,
    seed: int,
    activations_dir: str | None = None,
) -> list[TaskInput]:
    task_ids: set[str] | None = None
    if activations_dir is not None:
        task_ids = get_activation_task_ids(Path(activations_dir))
        print(f"Filtering to {len(task_ids)} tasks with activations from {activations_dir}")
    origin_enums = parse_origins(origins)
    tasks = load_filtered_tasks(
        n=n_tasks, origins=origin_enums, seed=seed, task_ids=task_ids, stratified=True,
    )
    return [{"task_id": t.id, "prompt": t.prompt} for t in tasks]


def cache_path_for(experiment_id: str | None, label: str = "topics") -> Path:
    if experiment_id:
        return OUTPUT_DIR / experiment_id / f"{label}.json"
    return OUTPUT_DIR / f"{label}.json"


def categories_path_for(experiment_id: str | None) -> Path:
    if experiment_id:
        return OUTPUT_DIR / experiment_id / "categories.json"
    return OUTPUT_DIR / "categories.json"


async def run_discovery(tasks: list[TaskInput], n_sample: int, seed: int) -> list[str]:
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
    tasks: list[TaskInput],
    categories: list[str],
    cache_path: Path,
    max_concurrent: int,
    category_descriptions: dict[str, str] | None = None,
) -> Cache:
    cache = load_cache(cache_path)
    print(f"\nClassifying {len(tasks)} tasks into {len(categories)} categories...")

    cache = await classify_tasks_batch(
        tasks, categories, cache, max_concurrent, category_descriptions,
    )
    save_cache(cache, cache_path)
    print(f"Cache saved to {cache_path}")

    return cache


def print_distribution(cache: Cache, task_ids: set[str]) -> None:
    relevant = {tid: entry for tid, entry in cache.items() if tid in task_ids}

    for model in MODELS:
        entries_with_model = {
            tid: e[model] for tid, e in relevant.items() if model in e
        }
        if not entries_with_model:
            continue
        counts = Counter(e["primary"] for e in entries_with_model.values())
        print(f"\n{model} distribution ({len(entries_with_model)} tasks):")
        for cat, count in counts.most_common():
            pct = count / len(entries_with_model) * 100
            print(f"  {cat:<30} {count:>5} ({pct:5.1f}%)")

    # Agreement rate (compare primary category across first two models)
    both = {
        tid: e for tid, e in relevant.items()
        if all(m in e for m in MODELS[:2])
    }
    if both and len(MODELS) >= 2:
        m1, m2 = MODELS[0], MODELS[1]
        agree = sum(1 for e in both.values() if e[m1]["primary"] == e[m2]["primary"])
        print(f"\nAgreement ({m1} vs {m2}): {agree}/{len(both)} ({agree/len(both)*100:.1f}%)")

    missing = task_ids - set(relevant.keys())
    if missing:
        print(f"  {'(unclassified)':<30} {len(missing):>5}")


async def reclassify_other(
    tasks: list[TaskInput],
    cache: Cache,
    original_categories: list[str],
    c_path: Path,
    cat_path: Path,
    discovery_sample_size: int,
    seed: int,
    max_concurrent: int,
) -> Cache:
    """Re-discover categories from 'other' tasks and reclassify them."""
    # Use first model's primary to identify 'other' tasks
    m1 = MODELS[0]
    other_tasks = [
        t for t in tasks
        if t["task_id"] in cache
        and m1 in cache[t["task_id"]]
        and cache[t["task_id"]][m1]["primary"] == "other"
    ]
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
    parser.add_argument("--max-concurrent", type=int, default=60)
    parser.add_argument("--reclassify-other", action="store_true",
                        help="Re-run discovery on 'other' tasks")
    parser.add_argument("--categories-file", type=str, default=None,
                        help="Path to JSON file with pre-defined categories (skip discovery)")
    parser.add_argument("--activations-dir", type=str, default=None,
                        help="Filter to tasks with activations in this directory")
    args = parser.parse_args()

    # Load tasks
    if args.experiment_id:
        tasks = load_tasks_from_ranked(args.experiment_id, args.run_name)
        label = args.experiment_id
    else:
        tasks = load_tasks_from_origins(
            args.origins, args.n_tasks, args.seed, args.activations_dir,
        )
        label = None

    print(f"Loaded {len(tasks)} tasks")
    task_ids = {t["task_id"] for t in tasks}

    c_path = cache_path_for(label)
    cat_path = categories_path_for(label)

    # Discover or load categories
    category_descriptions: dict[str, str] | None = None
    if args.categories_file:
        with open(args.categories_file) as f:
            cat_data = json.load(f)
        # Support both plain list and {categories, descriptions} format
        if isinstance(cat_data, list):
            categories = cat_data
        else:
            categories = cat_data["categories"]
            category_descriptions = cat_data.get("descriptions")
        print(f"Loaded {len(categories)} categories from {args.categories_file}")
        if category_descriptions:
            print(f"  with descriptions for {len(category_descriptions)} categories")
    elif cat_path.exists():
        with open(cat_path) as f:
            cat_data = json.load(f)
        if isinstance(cat_data, list):
            categories = cat_data
        else:
            categories = cat_data["categories"]
            category_descriptions = cat_data.get("descriptions")
        print(f"Loaded {len(categories)} existing categories from {cat_path}")
    else:
        categories = await run_discovery(tasks, args.discovery_sample_size, args.seed)
        cat_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cat_path, "w") as f:
            json.dump(categories, f, indent=2)
        print(f"Categories saved to {cat_path}")

    # Classify
    cache = await run_classification(
        tasks, categories, c_path, args.max_concurrent, category_descriptions,
    )
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
