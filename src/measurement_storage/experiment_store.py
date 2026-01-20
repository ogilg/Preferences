"""Experiment-level storage for organizing measurements by experiment_id."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.measurement_storage.base import save_yaml


EXPERIMENTS_DIR = Path("results/experiments")

MeasurementType = Literal[
    "pre_task_stated",
    "post_task_stated",
    "pre_task_revealed",
    "post_task_revealed",
]


class ExperimentStore:
    """Writes measurements to experiment-specific folders.

    Structure: results/experiments/{experiment_id}/{measurement_type}/{run_name}/
    """

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.base_dir = EXPERIMENTS_DIR / experiment_id

    def save_stated(
        self,
        measurement_type: MeasurementType,
        run_name: str,
        measurements: list[dict],
        config: dict,
    ) -> Path:
        """Save stated preference measurements (task_id, score pairs)."""
        run_dir = self.base_dir / measurement_type / run_name
        save_yaml(config, run_dir / "config.yaml")
        save_yaml(measurements, run_dir / "measurements.yaml")
        return run_dir

    def save_revealed(
        self,
        measurement_type: MeasurementType,
        run_name: str,
        measurements: list[dict],
        config: dict,
    ) -> Path:
        """Save revealed preference measurements (pairwise comparisons)."""
        run_dir = self.base_dir / measurement_type / run_name
        save_yaml(config, run_dir / "config.yaml")
        save_yaml(measurements, run_dir / "measurements.yaml")
        return run_dir
