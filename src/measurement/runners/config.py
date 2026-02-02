from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from src.task_data import OriginDataset, parse_origins

# Module-level experiment ID, set by run.py for the current run
_current_experiment_id: str | None = None


def set_experiment_id(experiment_id: str | None = None) -> str:
    """Set the experiment ID for this run. Auto-generates from timestamp if not provided."""
    global _current_experiment_id
    if experiment_id is None:
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _current_experiment_id = experiment_id
    return experiment_id


def get_experiment_id() -> str | None:
    """Get the current experiment ID."""
    return _current_experiment_id


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


class RankingConfig(BaseModel):
    n_tasks_per_ranking: int = 5
    appearances_per_task: int = 10
    seed: int = 42
    shuffle_task_order: bool = True  # Shuffle task order in prompt to control for position bias


class ExperimentConfig(BaseModel):
    preference_mode: Literal[
        "pre_task_revealed", "pre_task_stated", "pre_task_active_learning",
        "post_task_revealed", "post_task_stated", "post_task_active_learning",
        "pre_task_ranking", "post_task_ranking",
        "completion_generation",
    ]

    model: str
    temperature: float = 1.0
    max_concurrent: int | None = None
    max_new_tokens: int = 256  # Increase for models with thinking (e.g., qwen3: 2048)

    n_tasks: int = 10
    task_origins: list[Literal["wildchat", "alpaca", "math", "bailbench"]] = ["wildchat"]
    task_sampling_seed: int | None = None  # Seed for shuffling task order when sampling (None = no shuffle)

    templates: Path | None = None  # Path to template file, optional for completion_generation
    inline_templates: list[dict] | None = None  # Inline template definitions (alternative to templates path)

    n_samples: int = 5  # Samples per pair (revealed) or per task (stated)

    # Revealed-specific
    fitting: FittingConfig = Field(default_factory=FittingConfig)
    include_reverse_order: bool = False
    pair_order_seed: int | None = None  # Randomly shuffle (A,B) vs (B,A) per pair; defaults to 0 if not using reverse order

    # Active learning specific
    active_learning: ActiveLearningConfig | None = None

    # Ranking specific
    ranking: RankingConfig | None = None

    # Sensitivity dimensions
    response_formats: list[Literal["regex", "tool_use", "xml"]] = ["regex"]
    generation_seeds: list[int] = [0]

    # Template sampling
    template_sampling: Literal["all", "lhs"] = "all"
    n_template_samples: int | None = None
    lhs_seed: int | None = None

    # Post-task specific: which completion seeds to use (defaults to generation_seeds)
    completion_seeds: list[int] | None = None

    # If set, restrict measurements to only tasks with activations in activations/{model_name}/
    # Uses model name to locate activation data (e.g., "llama-3.1-8b" -> activations/llama_3_1_8b/)
    activations_model: str | None = None

    # Completion generation specific: run LLM-based refusal detection
    detect_refusals: bool = False

    # Experiment tracking
    experiment_id: str | None = None

    # System prompt to use during preference measurement (injected as system message)
    measurement_system_prompt: str | None = None

    # Cross-model experiments: separate model for generating completions vs rating them
    # If not set, completions are loaded from existing sources (e.g., concept_vectors/)
    completion_model: str | None = None
    completion_seed: int = 0

    @model_validator(mode="after")
    def validate_pair_order_options(self) -> "ExperimentConfig":
        if self.include_reverse_order and self.pair_order_seed is not None:
            raise ValueError("Cannot set pair_order_seed when include_reverse_order=True (no shuffling needed)")
        if not self.include_reverse_order and self.pair_order_seed is None:
            self.pair_order_seed = 0  # Default shuffle seed
        return self

    @model_validator(mode="after")
    def validate_ranking_config(self) -> "ExperimentConfig":
        if self.preference_mode in ("pre_task_ranking", "post_task_ranking"):
            if self.ranking is None:
                self.ranking = RankingConfig()
        return self

    def get_origin_datasets(self) -> list[OriginDataset]:
        return parse_origins(self.task_origins)


def load_experiment_config(path: Path) -> ExperimentConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ExperimentConfig.model_validate(data)
