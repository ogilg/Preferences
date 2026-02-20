"""Quick pilot: test one condition (baseline) with just a few pairs."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
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

ALL_ORIGINS = [
    OriginDataset.WILDCHAT, OriginDataset.ALPACA,
    OriginDataset.MATH, OriginDataset.BAILBENCH,
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

CORE_TASKS_PATH = Path("experiments/probe_generalization/persona_ood/phase3/core_tasks.json")

with open(CORE_TASKS_PATH) as f:
    core_data = json.load(f)
task_ids = core_data["task_ids"][:5]  # Just 5 tasks for pilot

print(f"Loading {len(task_ids)} tasks...")
tasks = load_filtered_tasks(n=len(task_ids), origins=ALL_ORIGINS, task_ids=set(task_ids))
print(f"Loaded {len(tasks)} tasks")

client = get_client("gemma-3-27b", max_new_tokens=256)
print(f"Client: {type(client).__name__}, model: {client.model_name}")

builder = PreTaskRevealedPromptBuilder(
    measurer=RevealedPreferenceMeasurer(),
    response_format=CompletionChoiceFormat(),
    template=TEMPLATE,
    system_prompt="You are a helpful assistant.",
)

# Test with 3 pairs
pairs = [(tasks[0], tasks[1]), (tasks[1], tasks[2]), (tasks[2], tasks[3])]
print(f"Testing {len(pairs)} pairs...")

batch = measure_pre_task_revealed(
    client=client,
    pairs=pairs,
    builder=builder,
    temperature=0.7,
    max_concurrent=10,
    seed=None,
)

print(f"Successes: {len(batch.successes)}")
print(f"Failures: {len(batch.failures)}")
for m in batch.successes:
    print(f"  {m.task_a.id} vs {m.task_b.id}: chose {'A' if m.choice == 'a' else 'B'}")
for f in batch.failures:
    print(f"  FAIL: {f.error_message[:100]}")
print("Pilot test complete!")
