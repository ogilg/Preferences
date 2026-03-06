"""Phase 0: Measure baseline pairwise revealed preferences for ICL transfer experiment.

Samples 50 tasks (balanced across topics), measures all pairs in both orderings
with 10 samples each.

Run: python scripts/icl_transfer/phase0_baseline.py
"""

import asyncio
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.measurement.elicitation import (
    measure_pre_task_revealed_async,
)
from src.measurement.elicitation.prompt_templates import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.measurement.storage.cache import MeasurementCache
from src.models import get_client, get_default_max_concurrent
from src.task_data import load_filtered_tasks, OriginDataset

# --- Config ---
MODEL = "google/gemma-3-27b-it"
N_TASKS = 50
SAMPLES_PER_DIRECTED_PAIR = 10
SEED = 42
TEMPLATE_PATH = "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"
TOPICS_PATH = "data/topics/topics.json"
OUTPUT_DIR = Path("experiments/icl_transfer/assets")


def get_topic(task_id: str, topics: dict) -> str:
    for _, cats in topics[task_id].items():
        return cats["primary"]


def select_task_ids(topics: dict, all_task_ids: set[str], n: int, seed: int) -> list[str]:
    """Sample n tasks that have topics, balanced across topics."""
    available = all_task_ids & set(topics.keys())
    rng = random.Random(seed)

    by_topic: dict[str, list[str]] = {}
    for tid in available:
        by_topic.setdefault(get_topic(tid, topics), []).append(tid)

    for t in by_topic:
        by_topic[t].sort()
        rng.shuffle(by_topic[t])

    # Round-robin across topics
    selected: list[str] = []
    cycle = sorted(by_topic.keys())
    idx = {t: 0 for t in cycle}
    while len(selected) < n:
        added_any = False
        for t in cycle:
            if idx[t] < len(by_topic[t]) and len(selected) < n:
                selected.append(by_topic[t][idx[t]])
                idx[t] += 1
                added_any = True
        if not added_any:
            break

    return selected


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(TOPICS_PATH) as f:
        topics = json.load(f)

    # Load all tasks from all origins, then select 50 with topics
    all_tasks = load_filtered_tasks(
        n=None,
        origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH,
                 OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST],
    )
    task_lookup = {t.id: t for t in all_tasks}
    print(f"Loaded {len(task_lookup)} tasks total")

    selected_ids = select_task_ids(topics, set(task_lookup.keys()), N_TASKS, SEED)
    topic_dist = Counter(get_topic(tid, topics) for tid in selected_ids)
    print(f"Selected {len(selected_ids)} tasks:")
    for t, c in topic_dist.most_common():
        print(f"  {t}: {c}")

    # Generate all directed pairs (both orderings)
    unordered = list(combinations(selected_ids, 2))
    canonical_pairs = [(task_lookup[a], task_lookup[b]) for a, b in unordered]
    reversed_pairs = [(task_lookup[b], task_lookup[a]) for a, b in unordered]

    # Repeat for samples
    canonical_all = canonical_pairs * SAMPLES_PER_DIRECTED_PAIR
    reversed_all = reversed_pairs * SAMPLES_PER_DIRECTED_PAIR

    total = len(canonical_all) + len(reversed_all)
    print(f"\nPairs: {len(unordered)} unordered × 2 orderings = {len(unordered) * 2} directed")
    print(f"Samples per directed pair: {SAMPLES_PER_DIRECTED_PAIR}")
    print(f"Total measurements: {total}")

    # Setup
    templates = load_templates_from_yaml(TEMPLATE_PATH)
    template = templates[0]
    builder = build_revealed_builder(template, response_format_name="completion")
    client = get_client(MODEL)
    semaphore = asyncio.Semaphore(get_default_max_concurrent())

    # Canonical ordering
    print(f"\n--- Canonical ordering ({len(canonical_all)} calls) ---")
    canon_batch = await measure_pre_task_revealed_async(
        client=client, pairs=canonical_all, builder=builder,
        semaphore=semaphore, temperature=1.0, seed=0,
    )
    print(f"  {len(canon_batch.successes)} successes, {len(canon_batch.failures)} failures")

    canon_cache = MeasurementCache(
        template=template, client=client,
        response_format="completion", order="canonical", seed=0,
    )
    canon_cache.append(canon_batch.successes)

    # Reversed ordering
    print(f"\n--- Reversed ordering ({len(reversed_all)} calls) ---")
    rev_batch = await measure_pre_task_revealed_async(
        client=client, pairs=reversed_all, builder=builder,
        semaphore=semaphore, temperature=1.0, seed=0,
    )
    print(f"  {len(rev_batch.successes)} successes, {len(rev_batch.failures)} failures")

    rev_cache = MeasurementCache(
        template=template, client=client,
        response_format="completion", order="canonical", seed=0,
    )
    rev_cache.append(rev_batch.successes)

    # Save task selection
    selection = {
        "seed": SEED,
        "n_tasks": N_TASKS,
        "samples_per_directed_pair": SAMPLES_PER_DIRECTED_PAIR,
        "task_ids": selected_ids,
        "topic_distribution": dict(topic_dist.most_common()),
    }
    selection_path = OUTPUT_DIR / "phase0_task_selection.json"
    with open(selection_path, "w") as f:
        json.dump(selection, f, indent=2)
    print(f"\nTask selection saved to {selection_path}")

    total_success = len(canon_batch.successes) + len(rev_batch.successes)
    total_fail = len(canon_batch.failures) + len(rev_batch.failures)
    print(f"\nDone. {total_success} successes, {total_fail} failures.")


if __name__ == "__main__":
    asyncio.run(main())
