"""Configuration schema for steering validation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SteeringExperimentConfig(BaseModel):
    """Config for steering experiment to validate probe directions."""

    model: str = "llama-3.1-8b"
    backend: Literal["nnsight", "transformer_lens"] = "transformer_lens"
    probe_manifest_dir: Path
    probe_id: str = "0009"
    steering_coefficients: list[float] = Field(
        default=[-2.0, -1.0, 0.0, 1.0, 2.0],
        description="Scaling factors for steering vector"
    )
    n_tasks: int = 25
    task_origins: list[Literal["wildchat", "alpaca", "math", "bailbench"]] = ["wildchat", "alpaca"]
    task_sampling_seed: int | None = None
    completion_seed: int = 0
    rating_seeds: list[int] = [0, 1, 2]
    prompt_variant: str = "experience_reflection"
    temperature: float = 1.0
    max_new_tokens: int = 128
    experiment_id: str


def load_steering_config(path: Path) -> SteeringExperimentConfig:
    """Load steering experiment config from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    # Convert probe_manifest_dir to Path if string
    if isinstance(data.get("probe_manifest_dir"), str):
        data["probe_manifest_dir"] = Path(data["probe_manifest_dir"])
    return SteeringExperimentConfig.model_validate(data)
