"""Shared utilities for concept vector measurement experiments.

Provides loading functions for completions and a synchronous measurement grid runner
for local model evaluation.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml

from src.measurement_storage import ExperimentStore
from src.measurement_storage.completions import TaskCompletion, _load_json, extract_completion_text
from src.running_measurements.progress import MultiExperimentProgress
from src.task_data import Task, OriginDataset


def load_concept_vector_completions(path: Path, condition: str) -> list[TaskCompletion]:
    """Load completions from a concept vector extraction directory."""
    completions_path = path / condition / "completions.json"
    data = _load_json(completions_path)
    return [
        TaskCompletion(
            task=Task(
                prompt=c["task_prompt"],
                origin=OriginDataset[c["origin"]],
                id=c["task_id"],
                metadata={},
            ),
            completion=extract_completion_text(c["completion"]),
        )
        for c in data
        if not c.get("truncated", False)
    ]


def load_neutral_completions(path: Path, origin_filter: str | None) -> list[TaskCompletion]:
    """Load completions from a standard completions file, optionally filtering by origin."""
    data = _load_json(path)
    completions = []
    for c in data:
        if origin_filter is not None and c.get("origin") != origin_filter:
            continue
        completions.append(
            TaskCompletion(
                task=Task(
                    prompt=c["task_prompt"],
                    origin=OriginDataset[c["origin"]],
                    id=c["task_id"],
                    metadata={},
                ),
                completion=c["completion"],
            )
        )
    return completions


def find_common_tasks(
    completion_sources: dict[str, list[TaskCompletion]],
) -> tuple[set[str], dict[str, list[TaskCompletion]]]:
    """Find common task IDs across all sources and filter each source to that set.

    Returns (common_ids, filtered_sources).
    """
    if not completion_sources:
        return set(), {}

    id_sets = [
        {tc.task.id for tc in completions}
        for completions in completion_sources.values()
    ]
    common_ids = set.intersection(*id_sets)

    filtered = {
        name: [tc for tc in completions if tc.task.id in common_ids]
        for name, completions in completion_sources.items()
    }
    return common_ids, filtered


def parse_stated_score(response: str, scale: tuple[int, int] = (1, 5)) -> float | None:
    """Parse a numeric score from a model response.

    Extracts the first number found that falls within the given scale.
    Returns None if no valid score found.
    """
    numbers = re.findall(r"-?(?:\d+\.?\d*|\.\d+)", response)
    for num_str in numbers:
        try:
            num = float(num_str)
            if scale[0] <= num <= scale[1]:
                return num
        except ValueError:
            continue
    return None


def run_measurement_grid(
    completions: dict[str, list[TaskCompletion]],
    conditions: dict[str, "MeasureFn"],
    seeds: list[int],
    exp_store: ExperimentStore,
    progress: MultiExperimentProgress,
    base_config: dict,
) -> dict[str, dict]:
    """Run measurements for all conditions × seeds synchronously.

    Args:
        completions: Dict of source_name -> list of TaskCompletion.
        conditions: Dict of condition_name -> measure function.
            Each measure function takes (TaskCompletion, seed) and returns (score | None, raw_response).
        seeds: List of generation seeds to use.
        exp_store: ExperimentStore to save results.
        progress: MultiExperimentProgress for progress display.
        base_config: Base configuration dict to include in saved configs.

    Returns:
        Dict of condition_name -> summary stats.
    """
    results_summary: dict[str, dict] = {}

    for condition_name, measure_fn in conditions.items():
        # Check if already complete
        if exp_store.exists("post_task_stated", condition_name):
            continue

        progress.set_status(condition_name, "running...")

        all_results = []
        successes = 0
        failures = 0

        # Get the completions for this condition
        # Condition names follow pattern: completion_{source}_...
        # We need to extract which completion source to use
        source_name = _extract_source_from_condition(condition_name)
        if source_name not in completions:
            # If can't determine source, use first available
            source_name = next(iter(completions))

        completion_list = completions[source_name]

        for seed in seeds:
            for tc in completion_list:
                score, raw = measure_fn(tc, seed)
                if score is not None:
                    all_results.append({
                        "task_id": tc.task.id,
                        "score": score,
                        "raw_response": raw,
                        "seed": seed,
                    })
                    successes += 1
                else:
                    failures += 1

                progress.update(condition_name, advance=1)

        # Save results
        run_config = {
            **base_config,
            "condition": condition_name,
            "n_results": len(all_results),
        }
        exp_store.save_stated("post_task_stated", condition_name, all_results, run_config)

        status = f"[green]{successes}✓[/green] [red]{failures}✗[/red]"
        progress.complete(condition_name, status=status)

        results_summary[condition_name] = {
            "successes": successes,
            "failures": failures,
            "total_runs": 1,
        }

    return results_summary


def _extract_source_from_condition(condition_name: str) -> str:
    """Extract completion source name from condition name.

    Expected patterns:
    - completion_{source}_layer{N}_coef{X} -> source
    - completion_{source}_context_{context} -> source
    """
    parts = condition_name.split("_")
    if len(parts) >= 2 and parts[0] == "completion":
        return parts[1]
    return condition_name


# Type alias for measure functions
from typing import Callable
MeasureFn = Callable[[TaskCompletion, int], tuple[float | None, str]]


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_steering_vector(concept_vectors_path: Path, layer: int, selector: str) -> np.ndarray:
    """Load steering vector for a given layer and selector."""
    selector_path = concept_vectors_path / "vectors" / selector / f"layer_{layer}.npy"
    if selector_path.exists():
        return np.load(selector_path)

    root_path = concept_vectors_path / "vectors" / f"layer_{layer}.npy"
    if root_path.exists():
        return np.load(root_path)

    raise FileNotFoundError(f"No steering vector found for layer {layer} at {concept_vectors_path}")
