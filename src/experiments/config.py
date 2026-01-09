from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from src.task_data import OriginDataset


class FittingConfig(BaseModel):
    max_iter: int | None = None
    gradient_tol: float | None = None
    loss_tol: float | None = None


class ActiveLearningConfig(BaseModel):
    initial_degree: int = 3
    batch_size: int = 300
    max_iterations: int = 20
    p_threshold: float = 0.3
    q_threshold: float = 0.3
    convergence_threshold: float = 0.99
    seed: int | None = None


class ExperimentConfig(BaseModel):
    preference_mode: Literal["binary", "rating", "active_learning"]

    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    temperature: float = 1.0
    max_concurrent: int | None = None

    n_tasks: int = 10
    task_origins: list[Literal["wildchat", "alpaca", "math"]] = ["wildchat"]

    templates: Path

    # Binary-specific
    samples_per_pair: int = 5
    fitting: FittingConfig = Field(default_factory=FittingConfig)

    # Rating-specific
    samples_per_task: int = 10
    scale_min: int = 1
    scale_max: int = 10

    # Active learning specific
    active_learning: ActiveLearningConfig = Field(default_factory=ActiveLearningConfig)

    # Sensitivity dimensions (binary and active_learning)
    response_formats: list[Literal["regex", "tool_use"]] = ["regex"]
    include_reverse_order: bool = False

    # Template sampling (to avoid combinatorial explosion)
    template_sampling: Literal["all", "lhs"] = "all"
    n_template_samples: int | None = None

    def get_origin_datasets(self) -> list[OriginDataset]:
        mapping = {
            "wildchat": OriginDataset.WILDCHAT,
            "alpaca": OriginDataset.ALPACA,
            "math": OriginDataset.MATH,
        }
        return [mapping[name] for name in self.task_origins]


def load_experiment_config(path: Path) -> ExperimentConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ExperimentConfig.model_validate(data)
