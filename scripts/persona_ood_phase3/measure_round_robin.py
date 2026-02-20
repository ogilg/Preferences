"""Round-robin pairwise measurement for phase 3.

Full round-robin among 50 tasks with 4 resamples per condition.
50*49/2 = 1,225 unique pairs × 4 resamples = 4,900 pairs per condition.
21 conditions (20 personas + baseline) = 102,900 total pairs.

Usage:
    python scripts/persona_ood_phase3/measure_round_robin.py [--condition <name>]
"""

import argparse
import json
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

import os
os.environ.setdefault("VLLM_API_KEY", "dummy")

from src.models import get_client
from src.task_data import load_filtered_tasks, OriginDataset
from src.measurement.elicitation.prompt_templates import (
    PreTaskRevealedPromptBuilder,
    PromptTemplate,
    TEMPLATE_TYPE_PLACEHOLDERS,
)
from src.measurement.elicitation.measurer import RevealedPreferenceMeasurer
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.measurement.elicitation.measure import measure_pre_task_revealed

CORE_TASKS_PATH = Path("experiments/probe_generalization/persona_ood/phase3/core_tasks.json")
ORIGINAL_PERSONAS_PATH = Path("experiments/probe_generalization/persona_ood/v2_config.json")
ENRICHED_PERSONAS_PATH = Path("experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json")
OUTPUT_PATH = Path("experiments/probe_generalization/persona_ood/phase3/results.json")

N_RESAMPLES = 4
TEMPERATURE = 0.7
MAX_CONCURRENT = 150
BASELINE_PROMPT = "You are a helpful assistant."

ALL_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]

TEMPLATE = PromptTemplate(
    template=(
        "You will be given two tasks. Choose one and complete it.\n"
        "{format_instruction}\n\n"
        "Task A:\n{task_a}\n\n"
        "Task B:\n{task_b}"
    ),
    name="completion_preference",
    required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
)


def make_builder(system_prompt: str | None = None) -> PreTaskRevealedPromptBuilder:
    return PreTaskRevealedPromptBuilder(
        measurer=RevealedPreferenceMeasurer(),
        response_format=CompletionChoiceFormat(),
        template=TEMPLATE,
        system_prompt=system_prompt,
    )


def make_round_robin_pairs(tasks, n_resamples, rng):
    """Create round-robin pairs with random A/B assignment per resample."""
    all_pairs = list(combinations(range(len(tasks)), 2))
    pairs_with_meta = []
    for i, j in all_pairs:
        for _ in range(n_resamples):
            if rng.random() < 0.5:
                pairs_with_meta.append((tasks[i], tasks[j], tasks[i].id, tasks[j].id, True))
            else:
                pairs_with_meta.append((tasks[j], tasks[i], tasks[i].id, tasks[j].id, False))
    return pairs_with_meta


def measure_one_condition(client, pairs_with_meta, system_prompt, temperature, max_concurrent, seed):
    builder = make_builder(system_prompt)
    api_pairs = [(a, b) for a, b, _, _, _ in pairs_with_meta]

    batch = measure_pre_task_revealed(
        client=client,
        pairs=api_pairs,
        builder=builder,
        temperature=temperature,
        max_concurrent=max_concurrent,
        seed=seed,
    )

    # Match results back to metadata
    success_queue: dict[tuple[str, str], list] = defaultdict(list)
    for m in batch.successes:
        success_queue[(m.task_a.id, m.task_b.id)].append(m)

    results = []
    for task_a, task_b, id_i, id_j, i_is_a in pairs_with_meta:
        key = (task_a.id, task_b.id)
        queue = success_queue.get(key, [])
        if not queue:
            continue
        m = queue.pop(0)

        if m.choice == "refusal":
            results.append({
                "task_i": id_i,
                "task_j": id_j,
                "chose_i": None,
                "is_refusal": True,
            })
        else:
            chose_a = m.choice == "a"
            chose_i = chose_a if i_is_a else not chose_a
            results.append({
                "task_i": id_i,
                "task_j": id_j,
                "chose_i": chose_i,
                "is_refusal": False,
            })

    n_refusals = sum(1 for r in results if r["is_refusal"])
    n_success = len(batch.successes)
    n_fail = len(batch.failures)
    print(f"  Successes: {n_success}, Failures: {n_fail}, Refusals: {n_refusals}")
    return results


def compute_p_choose(results, task_ids):
    """Compute p_choose for each task from round-robin results.

    p_choose = fraction of pairings where this task was chosen.
    """
    task_wins: dict[str, int] = defaultdict(int)
    task_total: dict[str, int] = defaultdict(int)
    task_refusals: dict[str, int] = defaultdict(int)

    for r in results:
        ti, tj = r["task_i"], r["task_j"]
        if r["is_refusal"]:
            task_refusals[ti] = task_refusals.get(ti, 0) + 1
            task_refusals[tj] = task_refusals.get(tj, 0) + 1
        elif r["chose_i"] is not None:
            task_total[ti] += 1
            task_total[tj] += 1
            if r["chose_i"]:
                task_wins[ti] += 1
            else:
                task_wins[tj] += 1

    rates = {}
    for tid in task_ids:
        wins = task_wins.get(tid, 0)
        total = task_total.get(tid, 0)
        refusals = task_refusals.get(tid, 0)
        rates[tid] = {
            "p_choose": wins / total if total > 0 else float("nan"),
            "n_wins": wins,
            "n_total": total,
            "n_refusals": refusals,
        }
    return rates


def load_personas():
    """Load all 20 personas (10 original broad + 10 enriched)."""
    with open(ORIGINAL_PERSONAS_PATH) as f:
        v2_config = json.load(f)

    # Only part A (broad) personas from v2
    original = [
        {"name": p["name"], "system_prompt": p["system_prompt"], "group": "original"}
        for p in v2_config["personas"] if p["part"] == "A"
    ]

    with open(ENRICHED_PERSONAS_PATH) as f:
        enriched_prompts = json.load(f)

    enriched = [
        {"name": name, "system_prompt": prompt, "group": "enriched"}
        for name, prompt in enriched_prompts.items()
    ]

    return original + enriched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, help="Run only this condition")
    args = parser.parse_args()

    with open(CORE_TASKS_PATH) as f:
        core_data = json.load(f)
    task_ids = core_data["task_ids"]

    print(f"Loading {len(task_ids)} tasks...")
    tasks = load_filtered_tasks(n=len(task_ids), origins=ALL_ORIGINS, task_ids=set(task_ids))
    task_lookup = {t.id: t for t in tasks}
    tasks_ordered = [task_lookup[tid] for tid in task_ids]
    print(f"Loaded {len(tasks_ordered)} tasks")

    personas = load_personas()
    print(f"Loaded {len(personas)} personas")

    client = get_client("gemma-3-27b", max_new_tokens=256)

    # Build conditions list
    conditions = []
    if args.condition:
        if args.condition == "baseline":
            conditions.append(("baseline", BASELINE_PROMPT))
        else:
            p = next((p for p in personas if p["name"] == args.condition), None)
            if p is None:
                raise ValueError(f"Unknown condition: {args.condition}")
            conditions.append((p["name"], p["system_prompt"]))
    else:
        conditions.append(("baseline", BASELINE_PROMPT))
        for p in personas:
            conditions.append((p["name"], p["system_prompt"]))

    # Load existing results
    output_data = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            output_data = json.load(f)

    # Skip completed conditions
    conditions = [(name, sp) for name, sp in conditions if name not in output_data]
    print(f"Conditions to run: {len(conditions)} (skipping {len(output_data)} already done)")

    n_pairs_per = len(list(combinations(range(len(tasks_ordered)), 2))) * N_RESAMPLES
    print(f"Pairs per condition: {n_pairs_per}")

    for idx, (name, system_prompt) in enumerate(conditions):
        print(f"\n=== [{idx+1}/{len(conditions)}] {name} ===")
        rng = np.random.RandomState(hash(name) % (2**31))
        pairs_with_meta = make_round_robin_pairs(tasks_ordered, N_RESAMPLES, rng)
        print(f"  Pairs: {len(pairs_with_meta)}")

        t0 = time.time()
        results = measure_one_condition(
            client, pairs_with_meta, system_prompt, TEMPERATURE, MAX_CONCURRENT, None
        )
        task_rates = compute_p_choose(results, task_ids)
        duration = time.time() - t0

        output_data[name] = {
            "system_prompt": system_prompt,
            "task_rates": task_rates,
            "raw_results": results,
            "duration_s": duration,
            "n_pairs": len(pairs_with_meta),
            "n_resamples": N_RESAMPLES,
        }

        # Save after each condition
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Duration: {duration:.1f}s. Saved to {OUTPUT_PATH}")

        time.sleep(2)

    print(f"\nDone. Total conditions: {len(output_data)}")


if __name__ == "__main__":
    main()
