"""Phase 2: Stated→Revealed ICL (K=1) across all 3 axes of the triad.

ICL context contains stated preference ratings (high for preferred topic task,
low for other topic task). Measurement is revealed pairwise choices on eval pairs.

Run: python -m scripts.icl_transfer.phase2_stated_k1
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
from src.measurement.runners.runners import build_revealed_builder, build_stated_builder
from src.models import get_client
from src.task_data import Task, OriginDataset
from src.types import Message

REVEALED_TEMPLATE_PATH = "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"
STATED_TEMPLATE_PATH = "src/measurement/elicitation/prompt_templates/data/pre_task_stated_v3.yaml"
EVAL_TASKS_PATH = Path("configs/icl_transfer/icl_eval_15_tasks.json")
CONTEXT_TASKS_PATH = Path("configs/icl_transfer/icl_context_tasks.json")
OUTPUT_DIR = Path("experiments/icl_transfer/assets")

MODEL = "gemma-3-27b"
TEMPERATURE = 1.0
N_SAMPLES = 10
N_ICL_PAIRS_PER_AXIS = 5
SEED = 42
MAX_CONCURRENT = 50

# Stated ratings: max and min on 1-3 scale
RATING_HIGH = "3"
RATING_LOW = "1"

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
            id=t["task_id"], prompt=t["prompt"],
            origin=_infer_origin(t["task_id"]), metadata={},
        )
        by_topic[t["topic"]].append(task)
    return dict(by_topic)


def build_stated_icl_pair(
    stated_builder, preferred_task: Task, other_task: Task,
) -> list[Message]:
    """Build stated ICL example: high rating for preferred, low rating for other."""
    prompt_high = stated_builder.build(preferred_task)
    prompt_low = stated_builder.build(other_task)
    # Extract just the user turn (stated builder may prepend system prompt)
    high_user = prompt_high.messages[-1]  # last message is the user turn
    low_user = prompt_low.messages[-1]
    return [
        high_user,
        {"role": "assistant", "content": RATING_HIGH},
        low_user,
        {"role": "assistant", "content": RATING_LOW},
    ]


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
    axes, eval_tasks, context_tasks, revealed_template, stated_builder, rng,
) -> list[Condition]:
    base_revealed = build_revealed_builder(revealed_template, response_format_name="completion")
    conditions = []

    for preferred_topic, other_topic in axes:
        ctx_pref_list = list(context_tasks[preferred_topic])
        ctx_other_list = list(context_tasks[other_topic])
        rng.shuffle(ctx_other_list)

        for i in range(N_ICL_PAIRS_PER_AXIS):
            pref_task = ctx_pref_list[i]
            other_task = ctx_other_list[i % len(ctx_other_list)]

            context_messages = build_stated_icl_pair(stated_builder, pref_task, other_task)

            eval_builder = PreTaskRevealedPromptBuilder(
                measurer=base_revealed.measurer,
                response_format=base_revealed.response_format,
                template=revealed_template,
                context_messages=context_messages,
            )

            eval_preferred = eval_tasks[preferred_topic]
            eval_other = eval_tasks[other_topic]

            conditions.append(Condition(
                axis=f"{preferred_topic}_over_{other_topic}",
                preferred_topic=preferred_topic,
                other_topic=other_topic,
                icl_preferred_id=pref_task.id,
                icl_other_id=other_task.id,
                builder=eval_builder,
                eval_pairs_canonical=list(product(eval_preferred, eval_other)),
                eval_pairs_reversed=list(product(eval_other, eval_preferred)),
            ))

    return conditions


async def run_condition(client, condition: Condition, semaphore) -> dict:
    all_measurements = []
    for ordering, pairs in [
        ("canonical", condition.eval_pairs_canonical),
        ("reversed", condition.eval_pairs_reversed),
    ]:
        expanded = pairs * N_SAMPLES
        batch = await measure_pre_task_revealed_async(
            client=client, pairs=expanded, builder=condition.builder,
            semaphore=semaphore, temperature=TEMPERATURE,
        )
        for m in batch.successes:
            all_measurements.append({
                "task_a": m.task_a.id, "task_b": m.task_b.id,
                "choice": m.choice, "ordering": ordering,
            })

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
    revealed_templates = load_templates_from_yaml(REVEALED_TEMPLATE_PATH)
    revealed_template = revealed_templates[0]

    stated_templates = load_templates_from_yaml(STATED_TEMPLATE_PATH)
    # Use 1-3 scale, phrasing 1, no situating context (template 001)
    stated_template = stated_templates[0]
    stated_builder = build_stated_builder(stated_template, response_format_name="regex")

    client = get_client(MODEL, max_new_tokens=256)
    rng = random.Random(SEED)

    eval_tasks = load_tasks(EVAL_TASKS_PATH)
    context_tasks = load_tasks(CONTEXT_TASKS_PATH)

    # Print what the stated ICL looks like
    sample_pref = context_tasks["math"][0]
    sample_other = context_tasks["harmful_request"][0]
    sample_msgs = build_stated_icl_pair(stated_builder, sample_pref, sample_other)
    print("Sample stated ICL example:")
    for msg in sample_msgs:
        print(f"  [{msg['role']}]: {msg['content'][:100]}...")
    print()

    conditions = build_conditions(
        AXES, eval_tasks, context_tasks, revealed_template, stated_builder, rng,
    )
    n_calls = len(conditions) * 25 * 2 * N_SAMPLES
    print(f"{len(conditions)} conditions ({len(AXES)} axes × {N_ICL_PAIRS_PER_AXIS} cond, K=1 stated)")
    print(f"Total: {n_calls} API calls")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [run_condition(client, c, semaphore) for c in conditions]
    results = await asyncio.gather(*tasks)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import numpy as np
    by_axis = defaultdict(list)
    for r in results:
        by_axis[r["axis"]].append(r["p_preferred"])

    print(f"\n{'Axis':40s} {'Mean P(pref)':>12s} {'Std':>8s}")
    print("-" * 62)
    for axis, ps in by_axis.items():
        print(f"  {axis:40s} {np.mean(ps):12.3f} {np.std(ps):8.3f}")

    summary = [{k: v for k, v in r.items() if k != "measurements"} for r in results]
    with open(OUTPUT_DIR / "phase2_stated_k1_results.json", "w") as f:
        json.dump({"conditions": summary, "config": {
            "model": MODEL, "temperature": TEMPERATURE,
            "n_samples": N_SAMPLES, "k": 1, "type": "stated_to_revealed",
            "stated_template": stated_template.name, "seed": SEED,
        }}, f, indent=2)

    with open(OUTPUT_DIR / "phase2_stated_k1_measurements.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/phase2_stated_k1_*.json")


if __name__ == "__main__":
    asyncio.run(main())
