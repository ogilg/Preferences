"""Preference measurement functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import BinaryPreferenceMeasurement, TaskScore, MeasurementResponse

if TYPE_CHECKING:
    from ..models import Model
    from ..task_data import Task
    from .prompt_builders import PromptBuilder


def _measure_single(
    model: "Model",
    builder: "PromptBuilder",
    args: tuple["Task", ...],
    temperature: float,
) -> MeasurementResponse | None:
    """Run a single measurement.

    Returns MeasurementResponse on success, None on failure.
    """
    prompt = builder.build(*args)
    tools = getattr(prompt.response_format, "tools", None)

    try:
        text = model.generate(
            prompt.messages,
            temperature=temperature,
            tools=tools,
        )
        return prompt.measurer.parse(text, prompt)
    except Exception:
        return None


def measure_binary_preferences(
    model: "Model",
    pairs: list[tuple["Task", "Task"]],
    builder: "PromptBuilder",
    temperature: float = 1.0,
) -> list[BinaryPreferenceMeasurement]:
    """Measure binary preferences for a list of task pairs.

    Args:
        model: Model to use for generation.
        pairs: List of (task_a, task_b) pairs to compare.
        builder: Prompt builder for binary comparisons.
        temperature: Sampling temperature.

    Returns:
        List of BinaryPreferenceMeasurement, one per successful measurement.
        Pairs that fail to parse are omitted.
    """
    measurements = []

    for task_a, task_b in pairs:
        response = _measure_single(model, builder, (task_a, task_b), temperature)
        if response is not None and isinstance(response.result, BinaryPreferenceMeasurement):
            measurements.append(response.result)

    return measurements


def measure_ratings(
    model: "Model",
    tasks: list["Task"],
    builder: "PromptBuilder",
    temperature: float = 1.0,
) -> list[TaskScore]:
    """Measure ratings for a list of tasks.

    Args:
        model: Model to use for generation.
        tasks: List of tasks to rate.
        builder: Prompt builder for rating measurements.
        temperature: Sampling temperature.

    Returns:
        List of TaskScore, one per successful measurement.
        Tasks that fail to parse are omitted.
    """
    scores = []

    for task in tasks:
        response = _measure_single(model, builder, (task,), temperature)
        if response is not None and isinstance(response.result, TaskScore):
            scores.append(response.result)

    return scores
