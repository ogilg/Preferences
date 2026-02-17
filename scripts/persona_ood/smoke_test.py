"""Smoke test: 2 core tasks × 2 anchors × 1 resample × 2 conditions = 8 API calls."""

import asyncio
import json
from pathlib import Path

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


def main():
    with open("experiments/probe_generalization/persona_ood/pilot_config.json") as f:
        config = json.load(f)

    # Take just 2 core + 2 anchor
    task_ids = config["core_task_ids"][:2] + config["anchor_task_ids"][:2]
    tasks = load_filtered_tasks(
        n=len(task_ids),
        origins=ALL_ORIGINS,
        task_ids=set(task_ids),
    )
    task_lookup = {t.id: t for t in tasks}

    core_tasks = [task_lookup[tid] for tid in config["core_task_ids"][:2] if tid in task_lookup]
    anchor_tasks = [task_lookup[tid] for tid in config["anchor_task_ids"][:2] if tid in task_lookup]

    pairs = [(c, a) for c in core_tasks for a in anchor_tasks]
    print(f"Testing with {len(pairs)} pairs")

    client = get_client("gemma-3-27b", max_new_tokens=256)

    # Baseline
    print("\n--- Baseline ---")
    builder = PreTaskRevealedPromptBuilder(
        measurer=RevealedPreferenceMeasurer(),
        response_format=CompletionChoiceFormat(),
        template=TEMPLATE,
        system_prompt=None,
    )
    batch = measure_pre_task_revealed(client, pairs, builder, temperature=0.7, max_concurrent=10)
    print(f"Successes: {len(batch.successes)}, Failures: {len(batch.failures)}")
    for m in batch.successes:
        print(f"  {m.task_a.id} vs {m.task_b.id}: chose {'A' if m.choice == 'a' else 'B' if m.choice == 'b' else 'refusal'}")

    # With persona
    print("\n--- Retired Diplomat ---")
    persona_prompt = config["personas"][0]["system_prompt"]
    builder_persona = PreTaskRevealedPromptBuilder(
        measurer=RevealedPreferenceMeasurer(),
        response_format=CompletionChoiceFormat(),
        template=TEMPLATE,
        system_prompt=persona_prompt,
    )
    batch_persona = measure_pre_task_revealed(client, pairs, builder_persona, temperature=0.7, max_concurrent=10)
    print(f"Successes: {len(batch_persona.successes)}, Failures: {len(batch_persona.failures)}")
    for m in batch_persona.successes:
        print(f"  {m.task_a.id} vs {m.task_b.id}: chose {'A' if m.choice == 'a' else 'B' if m.choice == 'b' else 'refusal'}")

    if batch.failures:
        print("\nBaseline failures:")
        for f in batch.failures:
            print(f"  {f.error_message[:100]}")
    if batch_persona.failures:
        print("\nPersona failures:")
        for f in batch_persona.failures:
            print(f"  {f.error_message[:100]}")


if __name__ == "__main__":
    main()
