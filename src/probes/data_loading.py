"""Shared measurement and activation loading utilities for probe training."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

from src.measurement.storage.loading import load_raw_scores, load_run_sigmas, load_run_utilities, load_yaml
from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, PreferenceType


def load_thurstonian_scores(run_dir: Path) -> dict[str, float]:
    """Load task_id -> mu mapping from Thurstonian fit."""
    mu_array, task_ids = load_run_utilities(run_dir)
    return dict(zip(task_ids, mu_array))


def load_thurstonian_scores_with_sigma(
    run_dir: Path,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """Load task_id -> mu and task_id -> sigma from Thurstonian fit.

    Returns (scores, sigmas) where sigmas is None if no thurstonian CSV exists.
    """
    scores = load_thurstonian_scores(run_dir)
    sigmas = load_run_sigmas(run_dir)
    return scores, sigmas


def load_pairwise_measurements(run_dir: Path) -> list[BinaryPreferenceMeasurement]:
    """Load measurements and reconstruct as BinaryPreferenceMeasurement objects."""
    measurements_path = run_dir / "measurements.yaml"
    raw = load_yaml(measurements_path)

    measurements = []
    for m in raw:
        task_a = Task(
            id=m["task_a"],
            prompt="",
            origin=OriginDataset[m["origin_a"]],
            metadata={},
        )
        task_b = Task(
            id=m["task_b"],
            prompt="",
            origin=OriginDataset[m["origin_b"]],
            metadata={},
        )
        measurements.append(BinaryPreferenceMeasurement(
            task_a=task_a,
            task_b=task_b,
            choice=m["choice"],
            preference_type=PreferenceType.POST_TASK_REVEALED,
        ))

    return measurements


def load_measurements_for_templates(
    experiment_dir: Path,
    templates: list[str],
    seeds: list[int],
) -> dict[str, float]:
    """Load measurements and average scores per task_id."""
    raw_measurements = []
    for template in templates:
        task_type = "pre_task" if template.startswith("pre_task") else "post_task"
        measurement_dir = experiment_dir / f"{task_type}_stated"
        raw_measurements.extend(load_raw_scores(measurement_dir, [template], seeds))

    # Average scores per task_id
    scores_by_task: dict[str, list[float]] = {}
    for task_id, score in raw_measurements:
        if task_id not in scores_by_task:
            scores_by_task[task_id] = []
        scores_by_task[task_id].append(score)

    return {tid: np.mean(scores) for tid, scores in scores_by_task.items()}


def filter_task_ids_by_datasets(
    task_ids: set[str],
    datasets: list[str] | None,
    origins_cache: dict[str, set[str]],
) -> set[str]:
    """Filter task IDs to only those in specified datasets. Returns all if datasets is None."""
    if not datasets:
        return task_ids
    target_ids = set()
    for dataset in datasets:
        target_ids.update(origins_cache.get(dataset.upper(), set()))
    return task_ids & target_ids


def expand_training_combinations(
    template_combos: list[list[str]],
    seed_combos: list[list[int]],
    layers: list[int],
    dataset_combos: list[list[str]] | None = None,
) -> list[dict]:
    """Returns list of dicts with keys: templates, datasets, seeds, layer."""
    ds_combos = dataset_combos or [None]

    return [
        {
            "templates": templates,
            "datasets": datasets,
            "seeds": seeds,
            "layer": layer,
        }
        for templates, seeds, datasets, layer in product(
            template_combos,
            seed_combos,
            ds_combos,
            layers,
        )
    ]


def collect_measurements_cache(
    experiment_dir: Path,
    combinations: list[dict],
) -> dict[tuple, dict[str, float]]:
    """Load and cache averaged measurements for all unique template/seed combinations."""
    cache: dict[tuple, dict[str, float]] = {}
    for combo in combinations:
        key = (tuple(combo["templates"]), tuple(combo["seeds"]))
        if key not in cache:
            cache[key] = load_measurements_for_templates(
                experiment_dir, combo["templates"], combo["seeds"]
            )
    return cache


def collect_needed_task_ids(
    measurements_cache: dict[tuple, dict[str, float]],
    combinations: list[dict],
    origins_cache: dict[str, set[str]],
) -> set[str]:
    """Collect union of all task IDs needed across all combinations."""
    all_task_ids: set[str] = set()

    for combo in combinations:
        key = (tuple(combo["templates"]), tuple(combo["seeds"]))
        task_ids = set(measurements_cache[key].keys())
        task_ids = filter_task_ids_by_datasets(task_ids, combo["datasets"], origins_cache)
        all_task_ids.update(task_ids)

    return all_task_ids
