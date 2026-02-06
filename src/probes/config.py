"""Configuration system for probe training experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml


class ProbeType(Enum):
    RIDGE = "ridge"
    BRADLEY_TERRY = "bradley_terry"


@dataclass
class DataSpec:
    experiment_dir: Path
    template_combinations: list[list[str]]
    seed_combinations: list[list[int]]
    dataset_combinations: list[list[str]] | None = None


@dataclass
class ProbeConfig:
    experiment_name: str
    activations_path: Path
    output_dir: Path
    layers: list[int]
    probe_type: ProbeType
    training_data: DataSpec
    evaluation_data: DataSpec | None = None
    # Ridge hyperparams
    cv_folds: int = 5
    alpha_sweep_size: int = 17
    # BT hyperparams
    bt_lr: float = 0.01
    bt_l2_lambda: float = 1.0
    bt_batch_size: int = 64
    bt_max_epochs: int = 1000
    bt_patience: int = 10

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> ProbeConfig:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        def parse_data_spec(spec: dict) -> DataSpec:
            return DataSpec(
                experiment_dir=Path(spec["experiment_dir"]),
                template_combinations=spec["template_combinations"],
                seed_combinations=spec["seed_combinations"],
                dataset_combinations=spec.get("dataset_combinations"),
            )

        training_data = parse_data_spec(data["training_data"])
        evaluation_data = parse_data_spec(data["evaluation_data"]) if "evaluation_data" in data else None

        return cls(
            experiment_name=data["experiment_name"],
            activations_path=Path(data["activations_path"]),
            output_dir=Path(data["output_dir"]),
            layers=data["layers"],
            probe_type=ProbeType(data.get("probe_type", "ridge")),
            training_data=training_data,
            evaluation_data=evaluation_data,
            cv_folds=data.get("cv_folds", 5),
            alpha_sweep_size=data.get("alpha_sweep_size", 17),
            bt_lr=data.get("bt_lr", 0.01),
            bt_l2_lambda=data.get("bt_l2_lambda", 1.0),
            bt_batch_size=data.get("bt_batch_size", 64),
            bt_max_epochs=data.get("bt_max_epochs", 1000),
            bt_patience=data.get("bt_patience", 10),
        )
