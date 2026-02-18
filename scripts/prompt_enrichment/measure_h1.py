"""Measure short vs rich prompt variants with stratified task sampling.

For each condition (baseline + 4 persona variants), draws a fresh stratified
sample of ~25 core tasks from the topic strata defined in the config, pairs
them against shared anchors, and measures P(choose core task).

Usage:
    python scripts/prompt_enrichment/measure_h1.py --config experiments/probe_generalization/persona_ood/prompt_enrichment/h1_config.json --output experiments/probe_generalization/persona_ood/prompt_enrichment/h1_results.json [--condition <name>]
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv

load_dotenv()

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


def stratified_sample(strata: dict, rng: np.random.RandomState, task_lookup: dict):
    """Draw a stratified sample from topic strata. Returns list of tasks."""
    sampled = []
    for topic, spec in strata.items():
        n = spec["n"]
        available = [tid for tid in spec["task_ids"] if tid in task_lookup]
        if len(available) < n:
            print(f"  Warning: {topic} has {len(available)} tasks, wanted {n}")
            n = len(available)
        chosen_ids = rng.choice(available, size=n, replace=False)
        sampled.extend([task_lookup[tid] for tid in chosen_ids])
        print(f"  {topic}: sampled {n} tasks")
    return sampled


def make_pairs(core_tasks, anchor_tasks, n_resamples, rng):
    pairs = []
    for core in core_tasks:
        for anchor in anchor_tasks:
            if core.id == anchor.id:
                continue
            for _ in range(n_resamples):
                is_core_a = rng.random() < 0.5
                if is_core_a:
                    pairs.append((core, anchor, core.id, anchor.id, True))
                else:
                    pairs.append((anchor, core, core.id, anchor.id, False))
    return pairs


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

    success_queue: dict[tuple[str, str], list] = defaultdict(list)
    for m in batch.successes:
        success_queue[(m.task_a.id, m.task_b.id)].append(m)

    results = []
    for task_a, task_b, core_id, anchor_id, is_core_a in pairs_with_meta:
        key = (task_a.id, task_b.id)
        queue = success_queue.get(key, [])
        if not queue:
            continue
        m = queue.pop(0)

        if m.choice == "refusal":
            results.append({
                "core_task_id": core_id,
                "anchor_task_id": anchor_id,
                "chose_core": None,
                "is_refusal": True,
            })
        else:
            chose_a = m.choice == "a"
            chose_core = chose_a if is_core_a else not chose_a
            results.append({
                "core_task_id": core_id,
                "anchor_task_id": anchor_id,
                "chose_core": chose_core,
                "is_refusal": False,
            })

    n_refusals = sum(1 for r in results if r["is_refusal"])
    print(f"  Successes: {len(batch.successes)}, Failures: {len(batch.failures)}, Refusals: {n_refusals}")
    return results


def compute_task_choice_rates(results):
    task_data: dict[str, list[bool]] = defaultdict(list)
    task_refusals: dict[str, int] = defaultdict(int)

    for r in results:
        tid = r["core_task_id"]
        if r["is_refusal"]:
            task_refusals[tid] += 1
        elif r["chose_core"] is not None:
            task_data[tid].append(r["chose_core"])

    rates = {}
    for tid in set(list(task_data.keys()) + list(task_refusals.keys())):
        choices = task_data.get(tid, [])
        n_chose = sum(choices)
        n_total = len(choices)
        refusals = task_refusals.get(tid, 0)
        rate = n_chose / n_total if n_total > 0 else float("nan")
        rates[tid] = {
            "p_choose": rate,
            "n_chose": n_chose,
            "n_total": n_total,
            "n_refusals": refusals,
        }
    return rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", type=str, help="Run only this condition (name or 'baseline')")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    strata = config["topic_strata"]
    anchor_task_ids = config["anchor_task_ids"]
    conditions_cfg = config["conditions"]
    n_resamples = config.get("n_resamples", 1)
    temperature = config.get("temperature", 0.7)
    max_concurrent = config.get("max_concurrent", 100)
    seed = config.get("seed", None)

    client = get_client("gemma-3-27b", max_new_tokens=256)

    # Collect all task IDs we might need
    all_pool_ids = set()
    for spec in strata.values():
        all_pool_ids.update(spec["task_ids"])
    all_pool_ids.update(anchor_task_ids)

    print("Loading tasks...")
    all_tasks = load_filtered_tasks(n=len(all_pool_ids), origins=ALL_ORIGINS, task_ids=all_pool_ids)
    task_lookup = {t.id: t for t in all_tasks}
    anchor_tasks = [task_lookup[tid] for tid in anchor_task_ids if tid in task_lookup]
    print(f"Loaded {len(task_lookup)} tasks total, {len(anchor_tasks)} anchors")

    # Load existing results
    output_data = {}
    if args.output.exists():
        with open(args.output) as f:
            output_data = json.load(f)

    # Build condition list
    baseline_prompt = config.get("baseline_prompt", "You are a helpful assistant.")
    conditions = []
    if args.condition:
        if args.condition == "baseline":
            conditions.append(("baseline", baseline_prompt))
        else:
            c = next((c for c in conditions_cfg if c["name"] == args.condition), None)
            if c is None:
                raise ValueError(f"Unknown condition: {args.condition}")
            conditions.append((c["name"], c["system_prompt"]))
    else:
        conditions.append(("baseline", baseline_prompt))
        for c in conditions_cfg:
            conditions.append((c["name"], c["system_prompt"]))

    conditions = [(name, sp) for name, sp in conditions if name not in output_data]
    print(f"Conditions to run: {len(conditions)}")

    # One stratified sample shared across all conditions
    sample_rng = np.random.RandomState(42)
    core_tasks = stratified_sample(strata, sample_rng, task_lookup)
    sampled_task_topics = {}
    for topic, spec in strata.items():
        for tid in spec["task_ids"]:
            if any(t.id == tid for t in core_tasks):
                sampled_task_topics[tid] = topic
    print(f"Shared core tasks: {len(core_tasks)}")

    for name, system_prompt in conditions:
        print(f"\n=== {name} ===")

        pair_rng = np.random.RandomState(hash(name) % (2**31))
        pairs_with_meta = make_pairs(core_tasks, anchor_tasks, n_resamples, pair_rng)
        print(f"  Pairs: {len(pairs_with_meta)}")

        t0 = time.time()
        results = measure_one_condition(
            client, pairs_with_meta, system_prompt, temperature, max_concurrent, seed
        )
        task_rates = compute_task_choice_rates(results)
        duration = time.time() - t0

        output_data[name] = {
            "system_prompt": system_prompt,
            "task_rates": task_rates,
            "sampled_tasks": sampled_task_topics,
            "duration_s": duration,
            "n_raw_results": len(results),
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Duration: {duration:.1f}s. Saved to {args.output}")

        time.sleep(5)

    print(f"\nDone. Total conditions: {len(output_data)}")


if __name__ == "__main__":
    main()
