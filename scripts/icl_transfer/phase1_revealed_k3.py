"""Phase 1: Revealed→Revealed ICL (K=3) across all 3 axes of the triad.

Same as K=1 but with 3 ICL examples stacked in context. Each condition uses
3 distinct preferred-topic tasks paired with 3 other-topic tasks.

Run: python -m scripts.icl_transfer.phase1_revealed_k3
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
from src.types import Message

TEMPLATE_PATH = "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"
EVAL_TASKS_PATH = Path("configs/icl_transfer/icl_eval_15_tasks.json")
CONTEXT_TASKS_PATH = Path("configs/icl_transfer/icl_context_tasks.json")
COMPLETIONS_PATH = Path("configs/icl_transfer/context_completions.json")
OUTPUT_DIR = Path("experiments/icl_transfer/assets")

MODEL = "gemma-3-27b"
TEMPERATURE = 1.0
N_SAMPLES = 10
K = 3
N_CONDITIONS_PER_AXIS = 5
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
            id=t["task_id"], prompt=t["prompt"],
            origin=_infer_origin(t["task_id"]), metadata={},
        )
        by_topic[t["topic"]].append(task)
    return dict(by_topic)


@dataclass
class Condition:
    axis: str
    preferred_topic: str
    other_topic: str
    icl_preferred_ids: list[str]
    icl_other_ids: list[str]
    builder: PreTaskRevealedPromptBuilder
    eval_pairs_canonical: list[tuple[Task, Task]]
    eval_pairs_reversed: list[tuple[Task, Task]]


def build_one_revealed_icl_pair(
    base_builder, preferred_task: Task, other_task: Task, completions: dict
) -> list[Message]:
    """Build one revealed ICL example (user + assistant turns)."""
    icl_prompt = base_builder.build(preferred_task, other_task)
    icl_user = icl_prompt.messages[0]
    completion_text = completions[preferred_task.id]["completion"]
    icl_assistant: Message = {"role": "assistant", "content": f"Task A:\n\n{completion_text}"}
    return [icl_user, icl_assistant]


def build_conditions(
    axes, eval_tasks, context_tasks, completions, template, rng,
) -> list[Condition]:
    base_builder = build_revealed_builder(template, response_format_name="completion")
    conditions = []

    for preferred_topic, other_topic in axes:
        ctx_pref_list = list(context_tasks[preferred_topic])
        ctx_other_list = list(context_tasks[other_topic])
        rng.shuffle(ctx_pref_list)
        rng.shuffle(ctx_other_list)

        # 5 conditions, each with K=3 ICL examples
        # Rotate through context tasks to maximize diversity
        for cond_idx in range(N_CONDITIONS_PER_AXIS):
            context_messages: list[Message] = []
            pref_ids = []
            other_ids = []

            for k in range(K):
                idx = (cond_idx * K + k) % len(ctx_pref_list)
                other_idx = (cond_idx * K + k) % len(ctx_other_list)
                pref_task = ctx_pref_list[idx]
                other_task = ctx_other_list[other_idx]
                pref_ids.append(pref_task.id)
                other_ids.append(other_task.id)
                context_messages.extend(
                    build_one_revealed_icl_pair(base_builder, pref_task, other_task, completions)
                )

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
                icl_preferred_ids=pref_ids,
                icl_other_ids=other_ids,
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
        "icl_preferred_ids": condition.icl_preferred_ids,
        "icl_other_ids": condition.icl_other_ids,
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

    conditions = build_conditions(AXES, eval_tasks, context_tasks, completions, template, rng)
    n_calls = len(conditions) * 25 * 2 * N_SAMPLES
    print(f"{len(conditions)} conditions ({len(AXES)} axes × {N_CONDITIONS_PER_AXIS} cond, K={K})")
    print(f"Total: {n_calls} API calls")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [run_condition(client, c, semaphore) for c in conditions]
    results = await asyncio.gather(*tasks)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Aggregate by axis
    import numpy as np
    by_axis = defaultdict(list)
    for r in results:
        by_axis[r["axis"]].append(r["p_preferred"])

    print(f"\n{'Axis':40s} {'Mean P(pref)':>12s} {'Std':>8s}")
    print("-" * 62)
    for axis, ps in by_axis.items():
        print(f"  {axis:40s} {np.mean(ps):12.3f} {np.std(ps):8.3f}")

    summary = [{k: v for k, v in r.items() if k != "measurements"} for r in results]
    with open(OUTPUT_DIR / "phase1_revealed_k3_results.json", "w") as f:
        json.dump({"conditions": summary, "config": {
            "model": MODEL, "temperature": TEMPERATURE,
            "n_samples": N_SAMPLES, "k": K, "seed": SEED,
        }}, f, indent=2)

    with open(OUTPUT_DIR / "phase1_revealed_k3_measurements.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/phase1_revealed_k3_*.json")


if __name__ == "__main__":
    asyncio.run(main())
