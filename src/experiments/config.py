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
    preference_mode: Literal["revealed", "stated", "active_learning"]

    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    temperature: float = 1.0
    max_concurrent: int | None = None

    n_tasks: int = 10
    task_origins: list[Literal["wildchat", "alpaca", "math"]] = ["wildchat"]

    templates: Path

    n_samples: int = 5  # Samples per pair (revealed) or per task (stated)

    # Revealed-specific
    fitting: FittingConfig = Field(default_factory=FittingConfig)
    include_reverse_order: bool = False

    # Active learning specific
    active_learning: ActiveLearningConfig | None = None

    # Sensitivity dimensions
    response_formats: list[Literal["regex", "tool_use", "xml"]] = ["regex"]
    generation_seeds: list[int] = [0]

    # Template sampling
    template_sampling: Literal["all", "lhs"] = "all"
    n_template_samples: int | None = None
    lhs_seed: int | None = None

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
