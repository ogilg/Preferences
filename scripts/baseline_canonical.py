"""Measure baseline pairwise preferences for fine-grained steering pairs.

Uses the canonical measurement infrastructure (OpenRouter API, completion_preference
template, CompletionChoiceFormat with semantic LLM parser fallback) — identical to
the active learning runs.

20 samples per pair (10 per ordering), stored for downstream steering analysis.
"""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.measurement.elicitation import measure_pre_task_revealed_async, BinaryPreferenceMeasurement, MeasurementBatch
from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.models import get_client
from src.task_data import Task, OriginDataset, load_tasks

PAIRS_PATH = Path("experiments/steering/replication/fine_grained/results/pairs.json")
TEMPLATE_PATH = Path("src/measurement/elicitation/prompt_templates/data/completion_preference.yaml")
OUTPUT_DIR = Path("experiments/new_steering")
OUTPUT_PATH = OUTPUT_DIR / "baseline_pairwise.json"

MODEL = "gemma-3-27b"
N_SAMPLES_PER_ORDERING = 10
TEMPERATURE = 0.7
CONCURRENCY = 20


ORIGIN_FROM_PREFIX = {
    "alpaca": OriginDataset.ALPACA,
    "wildchat": OriginDataset.WILDCHAT,
    "competition": OriginDataset.MATH,
    "bailbench": OriginDataset.BAILBENCH,
    "stresstest": OriginDataset.STRESS_TEST,
}


def origin_from_id(task_id: str) -> OriginDataset:
    prefix = task_id.split("_")[0]
    return ORIGIN_FROM_PREFIX[prefix]


def load_pairs() -> list[tuple[Task, Task, dict]]:
    """Load pairs and resolve task texts into Task objects."""
    with open(PAIRS_PATH) as f:
        pair_defs = json.load(f)

    pairs = []
    for p in pair_defs:
        task_a = Task(id=p["task_a"], prompt=p["task_a_text"], origin=origin_from_id(p["task_a"]), metadata={})
        task_b = Task(id=p["task_b"], prompt=p["task_b_text"], origin=origin_from_id(p["task_b"]), metadata={})
        pairs.append((task_a, task_b, p))
    return pairs


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build canonical builder (same as active learning)
    templates = load_templates_from_yaml(TEMPLATE_PATH)
    builder = build_revealed_builder(templates[0], "completion")

    client = get_client(MODEL)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    pairs_with_meta = load_pairs()
    print(f"Loaded {len(pairs_with_meta)} pairs")

    results = []

    for sample_idx in range(N_SAMPLES_PER_ORDERING):
        for ordering in range(2):
            if ordering == 0:
                api_pairs = [(a, b) for a, b, _ in pairs_with_meta]
            else:
                api_pairs = [(b, a) for a, b, _ in pairs_with_meta]

            print(f"Sample {sample_idx+1}/{N_SAMPLES_PER_ORDERING}, ordering={'AB' if ordering == 0 else 'BA'}...")
            batch: MeasurementBatch[BinaryPreferenceMeasurement] = await measure_pre_task_revealed_async(
                client=client,
                pairs=api_pairs,
                builder=builder,
                semaphore=semaphore,
                temperature=TEMPERATURE,
            )

            # Build lookup from (task_a_id, task_b_id) -> meta
            pair_lookup = {}
            for a, b, meta in pairs_with_meta:
                if ordering == 0:
                    pair_lookup[(a.id, b.id)] = meta
                else:
                    pair_lookup[(b.id, a.id)] = meta

            for measurement in batch.successes:
                key = (measurement.task_a.id, measurement.task_b.id)
                meta = pair_lookup[key]

                # Map choice back to original ordering
                if ordering == 0:
                    chose_original = measurement.choice
                else:
                    chose_original = {"a": "b", "b": "a"}.get(measurement.choice, measurement.choice)

                results.append({
                    "pair_id": meta["pair_id"],
                    "task_a": meta["task_a"],
                    "task_b": meta["task_b"],
                    "sample": sample_idx,
                    "ordering": ordering,
                    "choice_original": chose_original,
                    "raw_response_prefix": (measurement.raw_response or "")[:200],
                })

            if batch.failures:
                print(f"  {len(batch.failures)} failures: {[f.error_message[:80] for f in batch.failures]}")

    # Aggregate per pair
    pair_summaries = {}
    for r in results:
        pid = r["pair_id"]
        if pid not in pair_summaries:
            pair_summaries[pid] = {"pair_id": pid, "task_a": r["task_a"], "task_b": r["task_b"], "a": 0, "b": 0, "refusal": 0}
        if r["choice_original"] == "a":
            pair_summaries[pid]["a"] += 1
        elif r["choice_original"] == "b":
            pair_summaries[pid]["b"] += 1
        else:
            pair_summaries[pid]["refusal"] += 1

    for pid, s in pair_summaries.items():
        total_valid = s["a"] + s["b"]
        s["p_a"] = s["a"] / total_valid if total_valid > 0 else None
        # Add original metadata
        meta = next(m for _, _, m in pairs_with_meta if m["pair_id"] == pid)
        s["original_p_a"] = meta["p_a"]
        s["delta_mu"] = meta["delta_mu"]

    summaries = sorted(pair_summaries.values(), key=lambda x: x["pair_id"])

    decided = [s for s in summaries if s["p_a"] is not None and (s["p_a"] == 0 or s["p_a"] == 1)]
    total_refusals = sum(s["refusal"] for s in summaries)

    output = {
        "config": {
            "model": MODEL,
            "template": "completion_preference",
            "response_format": "completion",
            "temperature": TEMPERATURE,
            "n_samples_per_ordering": N_SAMPLES_PER_ORDERING,
            "total_trials_per_pair": N_SAMPLES_PER_ORDERING * 2,
        },
        "summary": {
            "n_pairs": len(summaries),
            "n_decided": len(decided),
            "total_refusals": total_refusals,
        },
        "pairs": summaries,
        "trials": results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone: {len(summaries)} pairs, {len(decided)} fully decided, {total_refusals} refusals")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
