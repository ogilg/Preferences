from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

from src.measurement.runners.utils.runner_utils import model_name_to_dir
from src.measurement.storage.base import find_project_root
from src.models.base import BATCHED_SELECTOR_REGISTRY, COMPLETION_SELECTORS


class ExtractionConfig(BaseModel):
    model: str
    backend: Literal["transformer_lens", "huggingface"]
    n_tasks: int
    task_origins: list[str]
    layers_to_extract: list[float | int]
    selectors: list[str]
    batch_size: int = 32
    seed: int | None = None
    temperature: float = 1.0
    max_new_tokens: int = 2048
    save_every: int = 100
    output_dir: str | None = None
    from_completions: Path | None = None
    resume: bool = False
    activations_model: str | None = None
    consistency_filter_model: str | None = None
    consistency_keep_ratio: float = 0.7

    @model_validator(mode="after")
    def validate_selectors(self) -> ExtractionConfig:
        for s in self.selectors:
            if s not in BATCHED_SELECTOR_REGISTRY:
                raise ValueError(f"Unknown selector: {s}. Valid: {list(BATCHED_SELECTOR_REGISTRY)}")
        return self

    @property
    def needs_generation(self) -> bool:
        return bool(set(self.selectors) & COMPLETION_SELECTORS) and self.from_completions is None

    @property
    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        model_dir = model_name_to_dir(self.model)
        return find_project_root() / "activations" / model_dir

    @classmethod
    def from_yaml(cls, path: Path, **cli_overrides: object) -> ExtractionConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        for key, value in cli_overrides.items():
            if value is not None:
                data[key] = value
        return cls.model_validate(data)
