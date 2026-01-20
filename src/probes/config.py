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
    output_dir: Path

    # Data selection
    templates: list[str]
    datasets: list[str] | None  # None = pool all datasets
    response_formats: list[str]  # pool together
    seeds: list[int]  # pool together
    layers: list[int]

    # Training
    cv_folds: int = 5

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> ProbeTrainingConfig:
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls(
            experiment_name=data["experiment_name"],
            experiment_dir=Path(data["experiment_dir"]),
            activations_path=Path(data["activations_path"]),
            templates=data["templates"],
            datasets=data.get("datasets"),
            response_formats=data.get("response_formats", ["regex", "tool_use"]),
            seeds=data.get("seeds", [0, 1]),
            layers=data["layers"],
            cv_folds=data.get("cv_folds", 5),
            output_dir=Path(data["output_dir"]),
        )
