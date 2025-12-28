"""Preference measurement functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import GenerateRequest
from ..types import BinaryPreferenceMeasurement, TaskScore

if TYPE_CHECKING:
    from ..models import Model
    from ..task_data import Task
    from .prompt_builders import PromptBuilder


def measure_binary_preferences(
    model: "Model",
    pairs: list[tuple["Task", "Task"]],
    builder: "PromptBuilder",
    temperature: float = 1.0,
    max_concurrent: int = 10,
) -> list[BinaryPreferenceMeasurement]:
    """Measure binary preferences for a list of task pairs.

    Args:
        model: Model to use for generation.
        pairs: List of (task_a, task_b) pairs to compare.
        builder: Prompt builder for binary comparisons.
        temperature: Sampling temperature.
        max_concurrent: Maximum number of concurrent API calls.

    Returns:
        List of BinaryPreferenceMeasurement, one per successful measurement.
        Pairs that fail to parse are omitted.
    """
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

    # Run in parallel
    responses = model.generate_batch(requests, max_concurrent)

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
    """Measure ratings for a list of tasks.

    Args:
        model: Model to use for generation.
        tasks: List of tasks to rate.
        builder: Prompt builder for rating measurements.
        temperature: Sampling temperature.
        max_concurrent: Maximum number of concurrent API calls.

    Returns:
        List of TaskScore, one per successful measurement.
        Tasks that fail to parse are omitted.
    """
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

    # Run in parallel
    responses = model.generate_batch(requests, max_concurrent)

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
