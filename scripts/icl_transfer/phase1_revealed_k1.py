"""Phase 1: Revealed→Revealed ICL (K=1) across all 3 axes of the triad.

For each directed axis, tries 5 different ICL context pairs. Each condition
runs all eval pairs (both orderings, 10 samples each). All 30 conditions
run concurrently.

Pre-generated completions loaded from configs/icl_transfer/context_completions.json.

Run: python -m scripts.icl_transfer.phase1_revealed_k1
"""

import asyncio
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.measurement.elicitation.measure import measure_pre_task_revealed_async
from src.measurement.elicitation.prompt_templates.builders import PreTaskRevealedPromptBuilder
from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.models import get_client
from src.task_data import Task, OriginDataset

TEMPLATE_PATH = "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"
EVAL_TASKS_PATH = Path("configs/icl_transfer/icl_eval_15_tasks.json")
CONTEXT_TASKS_PATH = Path("configs/icl_transfer/icl_context_tasks.json")
COMPLETIONS_PATH = Path("configs/icl_transfer/context_completions.json")
OUTPUT_DIR = Path("experiments/icl_transfer/assets")

MODEL = "gemma-3-27b"
TEMPERATURE = 1.0
N_SAMPLES = 10
N_ICL_PAIRS_PER_AXIS = 5
SEED = 42
MAX_CONCURRENT = 50

AXES = [
    ("math", "harmful_request"),
    ("harmful_request", "math"),
    ("fiction", "harmful_request"),
    ("harmful_request", "fiction"),
    ("math", "fiction"),
    ("fiction", "math"),
]

ORIGIN_PREFIXES = {
    "wildchat": OriginDataset.WILDCHAT,
    "alpaca": OriginDataset.ALPACA,
    "competition_math": OriginDataset.MATH,
    "stresstest": OriginDataset.STRESS_TEST,
    "bailbench": OriginDataset.BAILBENCH,
}


def _infer_origin(task_id: str) -> OriginDataset:
    for prefix, origin in ORIGIN_PREFIXES.items():
        if task_id.startswith(prefix):
            return origin
    return OriginDataset.SYNTHETIC


def load_tasks(path: Path) -> dict[str, list[Task]]:
    with open(path) as f:
        raw = json.load(f)
    by_topic: dict[str, list[Task]] = defaultdict(list)
    for t in raw:
        task = Task(
            id=t["task_id"],
            prompt=t["prompt"],
            origin=_infer_origin(t["task_id"]),
            metadata={},
        )
        by_topic[t["topic"]].append(task)
    return dict(by_topic)


@dataclass
class Condition:
    axis: str
    preferred_topic: str
    other_topic: str
    icl_preferred_id: str
    icl_other_id: str
    builder: PreTaskRevealedPromptBuilder
    eval_pairs_canonical: list[tuple[Task, Task]]
    eval_pairs_reversed: list[tuple[Task, Task]]


def build_conditions(
    axes: list[tuple[str, str]],
    eval_tasks: dict[str, list[Task]],
    context_tasks: dict[str, list[Task]],
    completions: dict[str, dict],
    template,
    rng: random.Random,
) -> list[Condition]:
    """Build all conditions: 5 ICL pairs per axis."""
    base_builder = build_revealed_builder(template, response_format_name="completion")
    conditions = []

    for preferred_topic, other_topic in axes:
        # Pick 5 ICL pairs: one per preferred-topic context task, random other
        ctx_preferred_tasks = list(context_tasks[preferred_topic])
        ctx_other_tasks = list(context_tasks[other_topic])
        rng.shuffle(ctx_other_tasks)

        for i in range(N_ICL_PAIRS_PER_AXIS):
            ctx_pref = ctx_preferred_tasks[i]
            ctx_other = ctx_other_tasks[i % len(ctx_other_tasks)]

            # Build ICL user turn (pairwise prompt with preferred as Task A)
            icl_prompt = base_builder.build(ctx_pref, ctx_other)
            icl_user_msg = icl_prompt.messages[0]

            # Assistant turn: "Task A:" + pre-generated completion
            completion_text = completions[ctx_pref.id]["completion"]
            icl_assistant_msg = {
                "role": "assistant",
                "content": f"Task A:\n\n{completion_text}",
            }

            context_messages = [icl_user_msg, icl_assistant_msg]

            eval_builder = PreTaskRevealedPromptBuilder(
                measurer=base_builder.measurer,
                response_format=base_builder.response_format,
                template=template,
                context_messages=context_messages,
            )

            eval_preferred = eval_tasks[preferred_topic]
            eval_other = eval_tasks[other_topic]

            conditions.append(Condition(
                axis=f"{preferred_topic}_over_{other_topic}",
                preferred_topic=preferred_topic,
                other_topic=other_topic,
                icl_preferred_id=ctx_pref.id,
                icl_other_id=ctx_other.id,
                builder=eval_builder,
                eval_pairs_canonical=list(product(eval_preferred, eval_other)),
                eval_pairs_reversed=list(product(eval_other, eval_preferred)),
            ))

    return conditions


async def run_condition(
    client, condition: Condition, semaphore: asyncio.Semaphore
) -> dict:
    """Run one ICL condition: measure all eval pairs in both orderings."""
    all_measurements = []

    for ordering, pairs in [
        ("canonical", condition.eval_pairs_canonical),
        ("reversed", condition.eval_pairs_reversed),
    ]:
        expanded = pairs * N_SAMPLES
        batch = await measure_pre_task_revealed_async(
            client=client,
            pairs=expanded,
            builder=condition.builder,
            semaphore=semaphore,
            temperature=TEMPERATURE,
        )
        for m in batch.successes:
            all_measurements.append({
                "task_a": m.task_a.id,
                "task_b": m.task_b.id,
                "choice": m.choice,
                "ordering": ordering,
            })

    # Compute P(choose preferred topic)
    eval_preferred_ids = {t.id for t, _ in condition.eval_pairs_canonical}
    preferred_chosen = 0
    total_valid = 0
    for m in all_measurements:
        if m["choice"] == "refusal":
            continue
        total_valid += 1
        chosen_id = m["task_a"] if m["choice"] == "a" else m["task_b"]
        if chosen_id in eval_preferred_ids:
            preferred_chosen += 1

    p_preferred = preferred_chosen / total_valid if total_valid > 0 else 0

    return {
        "axis": condition.axis,
        "preferred_topic": condition.preferred_topic,
        "other_topic": condition.other_topic,
        "icl_preferred_id": condition.icl_preferred_id,
        "icl_other_id": condition.icl_other_id,
        "p_preferred": round(p_preferred, 4),
        "n_valid": total_valid,
        "n_refusals": sum(1 for m in all_measurements if m["choice"] == "refusal"),
        "measurements": all_measurements,
    }


async def main():
    templates = load_templates_from_yaml(TEMPLATE_PATH)
    template = templates[0]
    client = get_client(MODEL, max_new_tokens=256)
    rng = random.Random(SEED)

    eval_tasks = load_tasks(EVAL_TASKS_PATH)
    context_tasks = load_tasks(CONTEXT_TASKS_PATH)
    with open(COMPLETIONS_PATH) as f:
        completions = json.load(f)

    print(f"Eval tasks: {', '.join(f'{k}={len(v)}' for k, v in eval_tasks.items())}")
    print(f"Context tasks: {', '.join(f'{k}={len(v)}' for k, v in context_tasks.items())}")
    print(f"Completions: {len(completions)}")

    conditions = build_conditions(
        AXES, eval_tasks, context_tasks, completions, template, rng
    )
    print(f"\n{len(conditions)} conditions ({len(AXES)} axes × {N_ICL_PAIRS_PER_AXIS} ICL pairs)")
    print(f"Per condition: 25 pairs × 2 orderings × {N_SAMPLES} samples = {25 * 2 * N_SAMPLES} calls")
    print(f"Total: {len(conditions) * 25 * 2 * N_SAMPLES} API calls")

    # Run all conditions concurrently
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [run_condition(client, c, semaphore) for c in conditions]
    results = await asyncio.gather(*tasks)

    # Print results grouped by axis
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Axis':40s} {'ICL pair':50s} {'P(pref)':>8s} {'N':>5s}")
    print("-" * 105)
    for r in results:
        icl_pair = f"{r['icl_preferred_id']} vs {r['icl_other_id']}"
        print(f"  {r['axis']:40s} {icl_pair:50s} {r['p_preferred']:8.3f} {r['n_valid']:5d}")

    # Aggregate by axis
    print(f"\n{'Axis':40s} {'Mean P(pref)':>12s} {'Std':>8s} {'N_cond':>7s}")
    print("-" * 70)
    by_axis = defaultdict(list)
    for r in results:
        by_axis[r["axis"]].append(r["p_preferred"])
    for axis, ps in by_axis.items():
        import numpy as np
        mean_p = np.mean(ps)
        std_p = np.std(ps)
        print(f"  {axis:40s} {mean_p:12.3f} {std_p:8.3f} {len(ps):7d}")

    # Save summary
    summary = [{k: v for k, v in r.items() if k != "measurements"} for r in results]
    output_path = OUTPUT_DIR / "phase1_revealed_k1_results.json"
    with open(output_path, "w") as f:
        json.dump({"conditions": summary, "config": {
            "model": MODEL, "temperature": TEMPERATURE,
            "n_samples": N_SAMPLES, "k": 1, "seed": SEED,
            "n_icl_pairs_per_axis": N_ICL_PAIRS_PER_AXIS,
        }}, f, indent=2)
    print(f"\nSaved summary to {output_path}")

    # Save full measurements
    full_path = OUTPUT_DIR / "phase1_revealed_k1_measurements.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full measurements to {full_path}")


if __name__ == "__main__":
    asyncio.run(main())
