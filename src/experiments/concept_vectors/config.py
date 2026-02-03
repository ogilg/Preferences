"""Configuration schema for concept vector extraction via system prompt conditioning."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

import yaml
from pydantic import BaseModel


class ConditionDict(TypedDict):
    name: str
    system_prompt: str


class ConceptVectorExtractionConfig(BaseModel):
    model: str
    backend: Literal["nnsight", "transformer_lens"] = "nnsight"

    n_tasks: int
    task_origins: list[Literal["wildchat", "alpaca", "math", "bailbench"]]
    task_sampling_seed: int | None = None

    # Task consistency filtering
    consistency_filter_model: str | None = None
    consistency_keep_ratio: float = 0.7

    conditions: dict[str, ConditionDict]

    layers_to_extract: list[float | int]
    selectors: list[Literal["last", "first", "mean"]] = ["last", "first", "mean"]
    temperature: float = 1.0
    max_new_tokens: int = 1024

    output_dir: Path
    experiment_id: str


def load_config(path: Path) -> ConceptVectorExtractionConfig:
    """Load concept vector extraction config from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    if "output_dir" in data and isinstance(data["output_dir"], str):
        data["output_dir"] = Path(data["output_dir"])

    return ConceptVectorExtractionConfig.model_validate(data)
