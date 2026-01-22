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
            seed_combinations=data["seed_combinations"],
            layers=data["layers"],
            cv_folds=data["cv_folds"],
            alpha_sweep_size=data["alpha_sweep_size"],
            manifest_dir=Path(data["manifest_dir"]),
        )


@dataclass
class ProbeEvaluationConfig:
    """Configuration for probe evaluation on a dataset."""

    # Probe specification
    manifest_dir: Path
    probe_ids: list[str]  # List of probe IDs to evaluate

    # Evaluation data specification
    experiment_dir: Path  # Directory with measurement results
    template: str  # Template name to evaluate on
    seeds: list[int]  # Seeds to include
    dataset_filter: str | None = None  # Optional dataset filter (e.g., "wildchat")

    # Activation data
    activations_path: Path | None = None  # Path to activations.npz, defaults to probe's training path

    # Output
    results_file: Path | None = None  # Where to save results

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> ProbeEvaluationConfig:
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls(
            manifest_dir=Path(data["probe"]["manifest_dir"]),
            probe_ids=data["probe"]["probe_ids"],
            experiment_dir=Path(data["data"]["experiment_dir"]),
            template=data["data"]["template"],
            seeds=data["data"]["seeds"],
            dataset_filter=data["data"].get("dataset_filter"),
            activations_path=Path(data["data"]["activations_path"]) if data["data"].get("activations_path") else None,
            results_file=Path(data["output"]["results_file"]) if data.get("output", {}).get("results_file") else None,
        )
