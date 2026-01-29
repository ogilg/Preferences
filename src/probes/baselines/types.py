"""Types for baseline experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BaselineType(Enum):
    SHUFFLED_LABELS = "shuffled_labels"
    RANDOM_ACTIVATIONS = "random_activations"
    TASK_DESCRIPTION = "task_description"


@dataclass
class BaselineResult:
    baseline_type: BaselineType
    template: str
    layer: int
    cv_r2_mean: float
    cv_r2_std: float
    cv_mse_mean: float
    cv_mse_std: float
    best_alpha: float
    n_samples: int
    seed: int | None = None

    def to_dict(self) -> dict:
        return {
            "baseline_type": self.baseline_type.value,
            "template": self.template,
            "layer": self.layer,
            "cv_r2_mean": self.cv_r2_mean,
            "cv_r2_std": self.cv_r2_std,
            "cv_mse_mean": self.cv_mse_mean,
            "cv_mse_std": self.cv_mse_std,
            "best_alpha": self.best_alpha,
            "n_samples": self.n_samples,
            "seed": self.seed,
        }
