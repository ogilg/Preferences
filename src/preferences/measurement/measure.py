from __future__ import annotations

from typing import TYPE_CHECKING

from tqdm import tqdm

from src.models import GenerateRequest
from src.types import BinaryPreferenceMeasurement, PreferenceType, TaskScore
from src.preferences.measurement.measurer import BinaryPreferenceMeasurer
from src.preferences.measurement.response_format import RegexChoiceFormat
from src.preferences.templates.builders import BinaryPromptBuilder

if TYPE_CHECKING:
    from src.models import Model
    from src.task_data import Task
    from src.preferences.templates.builders import PromptBuilder
    from src.preferences.templates.template import PromptTemplate


def measure_binary_preferences(
    model: "Model",
    pairs: list[tuple["Task", "Task"]],
    builder: "PromptBuilder",
    temperature: float = 1.0,
    max_concurrent: int = 10,
) -> list[BinaryPreferenceMeasurement]:
    """Pairs that fail to parse are omitted from results."""
    # Build all prompts
    prompts = [builder.build(task_a, task_b) for task_a, task_b in pairs]

    # Create batch requests
    requests = [
        GenerateRequest(
            messages=prompt.messages,
            temperature=temperature,
            tools=getattr(prompt.response_format, "tools", None),
        )
        for prompt in prompts
    ]

    # Run in parallel with progress
    pbar = tqdm(total=len(requests), desc="  Requests", leave=False)
    responses = model.generate_batch(requests, max_concurrent, on_complete=pbar.update)
    pbar.close()

    # Parse responses
    measurements = []
    for prompt, response in zip(prompts, responses):
        if response.ok:
            try:
                parsed = prompt.measurer.parse(response.unwrap(), prompt)
                if parsed is not None and isinstance(parsed.result, BinaryPreferenceMeasurement):
                    measurements.append(parsed.result)
            except Exception:
                pass

    return measurements


def measure_ratings(
    model: "Model",
    tasks: list["Task"],
    builder: "PromptBuilder",
    temperature: float = 1.0,
    max_concurrent: int = 10,
) -> list[TaskScore]:
    """Tasks that fail to parse are omitted from results."""
    # Build all prompts
    prompts = [builder.build(task) for task in tasks]

    # Create batch requests
    requests = [
        GenerateRequest(
            messages=prompt.messages,
            temperature=temperature,
            tools=getattr(prompt.response_format, "tools", None),
        )
        for prompt in prompts
    ]

    # Run in parallel with progress
    pbar = tqdm(total=len(requests), desc="  Requests", leave=False)
    responses = model.generate_batch(requests, max_concurrent, on_complete=pbar.update)
    pbar.close()

    # Parse responses
    scores = []
    for prompt, response in zip(prompts, responses):
        if response.ok:
            try:
                parsed = prompt.measurer.parse(response.unwrap(), prompt)
                if parsed is not None and isinstance(parsed.result, TaskScore):
                    scores.append(parsed.result)
            except Exception:
                pass

    return scores


def measure_with_template(
    template: "PromptTemplate",
    model: "Model",
    pairs: list[tuple["Task", "Task"]],
    temperature: float = 1.0,
    max_concurrent: int = 10,
) -> list[BinaryPreferenceMeasurement]:
    """Convenience wrapper that extracts task labels from template tags."""
    task_a_label = template.tags_dict.get("task_a_label", "Task A")
    task_b_label = template.tags_dict.get("task_b_label", "Task B")

    response_format = RegexChoiceFormat(
        task_a_label=task_a_label,
        task_b_label=task_b_label,
    )
    builder = BinaryPromptBuilder(
        measurer=BinaryPreferenceMeasurer(),
        preference_type=PreferenceType.PRE_TASK_STATED,
        response_format=response_format,
        template=template,
    )

    return measure_binary_preferences(
        model=model,
        pairs=pairs,
        builder=builder,
        temperature=temperature,
        max_concurrent=max_concurrent,
    )
