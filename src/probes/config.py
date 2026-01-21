"""Configuration system for probe training experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ProbeTrainingConfig:
    """Configuration for a probe training run."""

    # Experiment configuration
    experiment_name: str
    experiment_dir: Path
    activations_path: Path
    manifest_dir: Path  # Directory for storing manifest (metadata)

    # Data selection
    template_combinations: list[list[str]]  # List of template combinations to train on
    dataset_combinations: list[list[str]] | None  # List of dataset combinations, None = all
    response_format_combinations: list[list[str]]  # List of response format combinations
    seed_combinations: list[list[int]]  # List of seed combinations
    layers: list[int]

    # Training
    cv_folds: int
    alpha_sweep_size: int  # Number of alpha values to sweep

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> ProbeTrainingConfig:
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls(
            experiment_name=data["experiment_name"],
            experiment_dir=Path(data["experiment_dir"]),
            activations_path=Path(data["activations_path"]),
            template_combinations=data["template_combinations"],
            dataset_combinations=data.get("dataset_combinations"),
            response_format_combinations=data["response_format_combinations"],
            seed_combinations=data["seed_combinations"],
            layers=data["layers"],
            cv_folds=data["cv_folds"],
            alpha_sweep_size=data["alpha_sweep_size"],
            manifest_dir=Path(data["manifest_dir"]),
        )
