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
    layer: int
    best_alpha: float
    n_samples: int
    seed: int | None = None
    # CV metrics (populated when no eval_run_dir)
    cv_r2_mean: float | None = None
    cv_r2_std: float | None = None
    cv_mse_mean: float | None = None
    cv_mse_std: float | None = None
    # Heldout metrics (populated when eval_run_dir is set)
    heldout_r: float | None = None
    heldout_acc: float | None = None

    @classmethod
    def from_cv_result(
        cls, result: dict, baseline_type: BaselineType, layer: int,
        n_samples: int, seed: int | None = None,
    ) -> BaselineResult:
        return cls(
            baseline_type=baseline_type, layer=layer,
            best_alpha=result["best_alpha"], n_samples=n_samples, seed=seed,
            cv_r2_mean=result["cv_r2_mean"], cv_r2_std=result["cv_r2_std"],
            cv_mse_mean=result["cv_mse_mean"], cv_mse_std=result["cv_mse_std"],
        )

    @classmethod
    def from_heldout_result(
        cls, result: dict, baseline_type: BaselineType, layer: int,
        n_samples: int, seed: int | None = None,
    ) -> BaselineResult:
        return cls(
            baseline_type=baseline_type, layer=layer,
            best_alpha=result["best_alpha"], n_samples=n_samples, seed=seed,
            heldout_r=result["final_r"], heldout_acc=result["final_acc"],
        )

    def to_dict(self) -> dict:
        d: dict = {
            "baseline_type": self.baseline_type.value,
            "layer": self.layer,
            "best_alpha": self.best_alpha,
            "n_samples": self.n_samples,
            "seed": self.seed,
        }
        if self.cv_r2_mean is not None:
            d["cv_r2_mean"] = self.cv_r2_mean
            d["cv_r2_std"] = self.cv_r2_std
            d["cv_mse_mean"] = self.cv_mse_mean
            d["cv_mse_std"] = self.cv_mse_std
        if self.heldout_r is not None:
            d["heldout_r"] = self.heldout_r
        if self.heldout_acc is not None:
            d["heldout_acc"] = self.heldout_acc
        return d
