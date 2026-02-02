"""Load measurement runs for correlation analysis."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from src.measurement.storage import (
    EXPERIMENTS_DIR,
    list_runs,
    load_run_utilities,
    RunConfig,
)


class MeasurementType(Enum):
    PRE_STATED = "pre_stated"
    POST_STATED = "post_stated"
    PRE_REVEALED = "pre_revealed"
    POST_REVEALED = "post_revealed"

    @property
    def experiment_subdir(self) -> str:
        """Subdirectory name within an experiment folder."""
        return {
            MeasurementType.PRE_STATED: "pre_task_stated",
            MeasurementType.POST_STATED: "post_task_stated",
            MeasurementType.PRE_REVEALED: "pre_task_revealed",
            MeasurementType.POST_REVEALED: "post_task_revealed",
        }[self]

    @property
    def short_name(self) -> str:
        return {
            MeasurementType.PRE_STATED: "pre_st",
            MeasurementType.POST_STATED: "post_st",
            MeasurementType.PRE_REVEALED: "pre_rev",
            MeasurementType.POST_REVEALED: "post_rev",
        }[self]

    @property
    def display_name(self) -> str:
        return {
            MeasurementType.PRE_STATED: "Pre-task Stated",
            MeasurementType.POST_STATED: "Post-task Stated",
            MeasurementType.PRE_REVEALED: "Pre-task Revealed",
            MeasurementType.POST_REVEALED: "Post-task Revealed",
        }[self]

    def get_results_dir(self, experiment_id: str) -> Path:
        """Get results directory for an experiment."""
        return EXPERIMENTS_DIR / experiment_id / self.experiment_subdir


@dataclass
class LoadedRun:
    measurement_type: MeasurementType
    config: RunConfig
    values: np.ndarray
    task_ids: list[str]

    @property
    def group_key(self) -> str:
        """Key for grouping equivalent runs (same type, format, order)."""
        tags = self.config.template_tags
        fmt = tags.get("response_format", "?")
        order = tags.get("order", "")
        order_str = f"_{order}" if order else ""
        return f"{self.measurement_type.short_name}_{fmt}{order_str}"

    @property
    def label(self) -> str:
        """Unique label including template number."""
        # Extract template number from name (e.g., pre_task_qualitative_001 -> 001)
        parts = self.config.template_name.split("_")
        template_num = parts[-1] if parts[-1].isdigit() else ""
        return f"{self.group_key}_{template_num}" if template_num else self.group_key

    @property
    def full_label(self) -> str:
        return f"{self.config.template_name}_{self.group_key}"

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.task_ids, self.values))


def load_runs_for_model(
    model: str,
    experiment_id: str,
    measurement_types: list[MeasurementType] | None = None,
    min_tasks: int = 10,
    require_thurstonian_csv: bool = False,
) -> list[LoadedRun]:
    """Load all runs for a given model across measurement types.

    Args:
        model: Model short name to filter by (e.g., 'llama-3.1-8b')
        experiment_id: Experiment ID to read from
        measurement_types: Types to load (default: all)
        min_tasks: Minimum number of tasks required
        require_thurstonian_csv: For revealed, require pre-computed utilities
    """
    if measurement_types is None:
        measurement_types = list(MeasurementType)

    runs: list[LoadedRun] = []

    for mtype in measurement_types:
        results_dir = mtype.get_results_dir(experiment_id)
        if not results_dir.exists():
            continue

        for config in list_runs(results_dir):
            if config.model_short != model:
                continue

            # For revealed preferences, optionally require thurstonian CSV
            if require_thurstonian_csv and mtype in (
                MeasurementType.PRE_REVEALED,
                MeasurementType.POST_REVEALED,
            ):
                from src.measurement.storage import find_thurstonian_csv
                if find_thurstonian_csv(config.run_dir) is None:
                    continue

            try:
                values, task_ids = load_run_utilities(config.run_dir)
            except FileNotFoundError:
                continue

            if len(task_ids) < min_tasks:
                continue

            runs.append(LoadedRun(
                measurement_type=mtype,
                config=config,
                values=values,
                task_ids=task_ids,
            ))

    return runs


def list_available_models(
    experiment_id: str,
    measurement_types: list[MeasurementType] | None = None,
) -> set[str]:
    """List all models with data in the specified measurement types."""
    if measurement_types is None:
        measurement_types = list(MeasurementType)

    models: set[str] = set()
    for mtype in measurement_types:
        results_dir = mtype.get_results_dir(experiment_id)
        if not results_dir.exists():
            continue
        for config in list_runs(results_dir):
            models.add(config.model_short)

    return models


def aggregate_runs_by_group(runs: list[LoadedRun]) -> list[LoadedRun]:
    """Aggregate runs with the same group_key by averaging values per task.

    This collapses multiple seeds/runs into a single representative run per
    (measurement_type, response_format, order) combination.
    """
    from collections import defaultdict

    groups: dict[str, list[LoadedRun]] = defaultdict(list)
    for run in runs:
        groups[run.group_key].append(run)

    aggregated: list[LoadedRun] = []
    for group_key, group_runs in groups.items():
        if len(group_runs) == 1:
            aggregated.append(group_runs[0])
            continue

        # Aggregate by averaging values for each task
        task_values: dict[str, list[float]] = defaultdict(list)
        for run in group_runs:
            for tid, val in zip(run.task_ids, run.values):
                task_values[tid].append(val)

        task_ids = sorted(task_values.keys())
        values = np.array([np.mean(task_values[tid]) for tid in task_ids])

        # Use first run as template for config
        first = group_runs[0]
        aggregated.append(LoadedRun(
            measurement_type=first.measurement_type,
            config=first.config,
            values=values,
            task_ids=task_ids,
        ))

    return aggregated
