"""Dataset-wide preference measurement orchestration."""

from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING, Any

from ..models import Model
from ..types import MeasurementResponse
from .config import DatasetMeasurementConfig, PairingStrategy

if TYPE_CHECKING:
    from ..task_data import Task
    from .prompt_builders import PromptBuilder


def measure_dataset_preferences(
    model: Model,
    tasks: list[Task],
    rating_builder: PromptBuilder | None = None,
    binary_builder: PromptBuilder | None = None,
    config: DatasetMeasurementConfig | None = None,
) -> dict[str, Any]:
    """Measure preferences across an entire dataset.

    Args:
        model: Model to use for generation.
        tasks: List of tasks to measure preferences for.
        rating_builder: Builder for per-task rating measurements.
            Required if "rating" in config.measurement_types.
        binary_builder: Builder for binary comparison measurements.
            Required if "binary" in config.measurement_types.
        config: Measurement configuration. Uses defaults if None.

    Returns:
        Dictionary containing:
            - config: The DatasetMeasurementConfig used
            - task_ratings: List of rating results per task (if enabled)
            - binary_comparisons: List of comparison results per pair (if enabled)

    Raises:
        ValueError: If required builders are missing for requested measurement types.
    """
    config = config or DatasetMeasurementConfig()
    _validate_builders(config, rating_builder, binary_builder)

    result: dict[str, Any] = {
        "config": config,
        "task_ratings": [],
        "binary_comparisons": [],
    }

    # Run rating measurements
    if "rating" in config.measurement_types and rating_builder is not None:
        result["task_ratings"] = _measure_all_ratings(
            model=model,
            tasks=tasks,
            builder=rating_builder,
            config=config,
        )

    # Run binary comparison measurements
    if "binary" in config.measurement_types and binary_builder is not None:
        pairs = _generate_pairs(tasks, config)
        result["binary_comparisons"] = _measure_all_comparisons(
            model=model,
            pairs=pairs,
            builder=binary_builder,
            config=config,
        )

    return result


def _validate_builders(
    config: DatasetMeasurementConfig,
    rating_builder: PromptBuilder | None,
    binary_builder: PromptBuilder | None,
) -> None:
    """Validate that required builders are provided."""
    if "rating" in config.measurement_types and rating_builder is None:
        raise ValueError("rating_builder required when 'rating' in measurement_types")
    if "binary" in config.measurement_types and binary_builder is None:
        raise ValueError("binary_builder required when 'binary' in measurement_types")


def _measure_all_ratings(
    model: Model,
    tasks: list[Task],
    builder: PromptBuilder,
    config: DatasetMeasurementConfig,
) -> list[dict[str, Any]]:
    """Measure ratings for all tasks with multiple samples each."""
    results = []
    for task in tasks:
        samples = _sample_measurement(
            model=model,
            builder=builder,
            config=config,
            args=(task,),
        )
        results.append({
            "task": task,
            "samples": samples,
        })
    return results


def _measure_all_comparisons(
    model: Model,
    pairs: list[tuple[Task, Task]],
    builder: PromptBuilder,
    config: DatasetMeasurementConfig,
) -> list[dict[str, Any]]:
    """Measure binary comparisons for all task pairs with multiple samples."""
    results = []
    for task_a, task_b in pairs:
        samples = _sample_measurement(
            model=model,
            builder=builder,
            config=config,
            args=(task_a, task_b),
        )
        results.append({
            "task_a": task_a,
            "task_b": task_b,
            "samples": samples,
        })
    return results


def _sample_measurement(
    model: Model,
    builder: PromptBuilder,
    config: DatasetMeasurementConfig,
    args: tuple[Task] | tuple[Task, Task],
) -> list[dict[str, Any]]:
    """Run a measurement N times and collect all samples."""
    samples = []
    for i in range(config.num_samples):
        prompt = builder.build(*args)
        text = model.generate(prompt.messages, temperature=config.temperature)
        response: MeasurementResponse = prompt.measurer.parse(text, prompt)
        samples.append({
            "sample_index": i,
            "response": response,
            "temperature": config.temperature,
        })
    return samples


def _generate_pairs(
    tasks: list[Task],
    config: DatasetMeasurementConfig,
) -> list[tuple[Task, Task]]:
    """Generate task pairs based on the configured strategy."""
    rng = random.Random(config.seed)

    if config.pairing_strategy == PairingStrategy.ALL_PAIRS:
        pairs = list(itertools.combinations(tasks, 2))

    elif config.pairing_strategy == PairingStrategy.ADJACENT_PAIRS:
        pairs = [(tasks[i], tasks[i + 1]) for i in range(len(tasks) - 1)]

    elif config.pairing_strategy == PairingStrategy.RANDOM_PAIRS:
        all_pairs = list(itertools.combinations(tasks, 2))
        max_pairs = config.max_pairs or len(all_pairs)
        pairs = rng.sample(all_pairs, min(max_pairs, len(all_pairs)))

    else:
        raise ValueError(f"Unknown pairing strategy: {config.pairing_strategy}")

    return pairs
