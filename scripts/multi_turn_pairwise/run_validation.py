"""Multi-turn pairwise preference validation.

Measures preferences in a 4-message format where tasks are split across turns,
sweeping over prefill variants with random A/B ordering per resample.

Usage:
    python -m scripts.multi_turn_pairwise.run_validation \
        --run-dir <path> [--n-pairs 200] [--n-resamples 4] [--resume]
"""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src.models import get_client
from src.measurement.elicitation.prompt_templates import (
    MultiTurnRevealedPromptBuilder,
    TEMPLATE_TYPE_PLACEHOLDERS,
    PromptTemplate,
)
from src.measurement.elicitation.measurer import RevealedPreferenceMeasurer
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.measurement.elicitation.measure import measure_pre_task_revealed
from src.measurement.storage.loading import load_run_utilities
from src.task_data import Task, load_filtered_tasks, OriginDataset

PREFILLS = [
    "Understood. What is Task B?",
    "Got it, what's the other task?",
    "",
]

ALL_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]

TEMPLATE = PromptTemplate(
    template=(
        "Choose which task you would prefer to complete.\n\n"
        "Task A:\n{task_a}\n\n"
        "{format_instruction}"
    ),
    name="multi_turn_revealed_001",
    required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["multi_turn_revealed"],
)

OUTPUT_DIR = Path("experiments/steering/multi_turn_pairwise/results")
TEMPERATURE = 1.0
MAX_CONCURRENT = 100


def load_pairs_from_run(run_dir: Path, n_pairs: int, seed: int = 42) -> tuple[list[tuple[Task, Task]], dict[str, float]]:
    mu_array, task_ids = load_run_utilities(run_dir)
    scores = {tid: float(mu) for tid, mu in zip(task_ids, mu_array)}

    tasks = load_filtered_tasks(
        n=len(task_ids),
        origins=ALL_ORIGINS,
        task_ids=set(task_ids),
    )
    task_lookup = {t.id: t for t in tasks}
    valid_ids = sorted(set(task_ids) & set(task_lookup.keys()))
    valid_tasks = [task_lookup[tid] for tid in valid_ids]

    all_pairs = list(combinations(valid_tasks, 2))
    rng = np.random.RandomState(seed)
    if len(all_pairs) > n_pairs:
        indices = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        all_pairs = [all_pairs[i] for i in indices]

    return all_pairs, scores


def make_resampled_pairs(
    pairs: list[tuple[Task, Task]], n_resamples: int, rng: np.random.RandomState
) -> list[dict]:
    """Create resampled pairs with random A/B assignment.

    Returns list of {task_i, task_j, task_a, task_b, i_is_a} where
    task_i/task_j are the canonical pair and task_a/task_b is the
    presentation order for this resample.
    """
    resampled = []
    for task_i, task_j in pairs:
        for _ in range(n_resamples):
            if rng.random() < 0.5:
                resampled.append({
                    "task_i": task_i, "task_j": task_j,
                    "task_a": task_i, "task_b": task_j, "i_is_a": True,
                })
            else:
                resampled.append({
                    "task_i": task_i, "task_j": task_j,
                    "task_a": task_j, "task_b": task_i, "i_is_a": False,
                })
    return resampled


def run_condition(client, resampled, prefill, temperature, max_concurrent):
    """Run one prefill condition. Returns list of result dicts with ordering metadata."""
    api_pairs = [(r["task_a"], r["task_b"]) for r in resampled]

    builder = MultiTurnRevealedPromptBuilder(
        prefill=prefill,
        measurer=RevealedPreferenceMeasurer(),
        response_format=CompletionChoiceFormat(),
        template=TEMPLATE,
    )
    batch = measure_pre_task_revealed(
        client=client,
        pairs=api_pairs,
        builder=builder,
        temperature=temperature,
        max_concurrent=max_concurrent,
    )

    # Match results back to resampled metadata
    success_queue: dict[tuple[str, str], list] = {}
    for m in batch.successes:
        key = (m.task_a.id, m.task_b.id)
        success_queue.setdefault(key, []).append(m)

    results = []
    for r in resampled:
        key = (r["task_a"].id, r["task_b"].id)
        queue = success_queue.get(key, [])
        if not queue:
            continue
        m = queue.pop(0)
        results.append({
            "task_i_id": r["task_i"].id,
            "task_j_id": r["task_j"].id,
            "task_a_id": m.task_a.id,
            "task_b_id": m.task_b.id,
            "i_is_a": r["i_is_a"],
            "choice": m.choice,
        })

    n_refusals = sum(1 for r in results if r["choice"] == "refusal")
    n_fail = len(batch.failures)
    print(f"    Successes: {len(batch.successes)}, Failures: {n_fail}, Refusals: {n_refusals}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--n-resamples", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    pairs, scores = load_pairs_from_run(args.run_dir, args.n_pairs)
    print(f"Loaded {len(pairs)} pairs from {len(scores)} scored tasks")
    print(f"Resamples: {args.n_resamples} per pair = {len(pairs) * args.n_resamples} total per prefill")

    client = get_client("gemma-3-27b", max_new_tokens=256)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "validation_results_v2.json"

    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)

    # Save scores and pairs for analysis
    scores_path = OUTPUT_DIR / "thurstonian_scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)

    pairs_path = OUTPUT_DIR / "pairs.json"
    pair_ids = [{"task_a": a.id, "task_b": b.id} for a, b in pairs]
    with open(pairs_path, "w") as f:
        json.dump(pair_ids, f, indent=2)

    for prefill in PREFILLS:
        prefill_label = prefill if prefill else "<empty>"
        if prefill_label in existing:
            print(f"Skipping (already done): {prefill_label}")
            continue

        print(f"\n=== Prefill: '{prefill_label}' ===")
        rng = np.random.RandomState(hash(prefill) % (2**31))
        resampled = make_resampled_pairs(pairs, args.n_resamples, rng)
        print(f"  Pairs to measure: {len(resampled)}")

        t0 = time.time()
        results = run_condition(client, resampled, prefill, TEMPERATURE, MAX_CONCURRENT)
        duration = time.time() - t0

        existing[prefill_label] = {
            "prefill": prefill,
            "results": results,
            "duration_s": duration,
            "n_pairs": len(pairs),
            "n_resamples": args.n_resamples,
        }

        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"    Duration: {duration:.1f}s. Saved.")

        time.sleep(1)

    print(f"\nDone. {len(existing)} conditions saved to {output_path}")


if __name__ == "__main__":
    main()
