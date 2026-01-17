"""Shared utilities for experiment runners."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.models import get_client, get_default_max_concurrent, OpenAICompatibleClient
from src.task_data import Task, load_tasks
from src.prompt_templates import load_templates_from_yaml, PromptTemplate
from src.thurstonian_fitting import _config_hash
from src.running_measurements.config import load_experiment_config, ExperimentConfig


@dataclass
class ExperimentContext:
    config: ExperimentConfig
    templates: list[PromptTemplate] | None
    tasks: list[Task]
    task_lookup: dict[str, Task]
    client: OpenAICompatibleClient
    max_concurrent: int


def parse_config_path(description: str) -> Path:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", type=Path, help="Path to experiment config YAML")
    return parser.parse_args().config


def setup_experiment(
    config_path: Path,
    expected_mode: str,
    max_new_tokens: int = 256,
) -> ExperimentContext:
    config = load_experiment_config(config_path)

    if config.preference_mode != expected_mode:
        raise ValueError(f"Expected preference_mode='{expected_mode}', got '{config.preference_mode}'")

    # Always load tasks deterministically (no shuffle) so stated and revealed use same tasks
    tasks = load_tasks(n=config.n_tasks, origins=config.get_origin_datasets(), seed=None)

    # Templates are optional for completion_generation mode
    templates = None
    if config.templates is not None:
        templates = load_templates_from_yaml(config.templates)

    return ExperimentContext(
        config=config,
        templates=templates,
        tasks=tasks,
        task_lookup={t.id: t for t in tasks},
        client=get_client(model_name=config.model, max_new_tokens=max_new_tokens),
        max_concurrent=config.max_concurrent or get_default_max_concurrent(),
    )


def compute_thurstonian_max_iter(config: ExperimentConfig) -> int:
    n_params = (config.n_tasks - 1) + config.n_tasks
    return config.fitting.max_iter or max(2000, n_params * 50)


def build_fit_kwargs(config: ExperimentConfig, max_iter: int) -> dict:
    kwargs = {"max_iter": max_iter}
    if config.fitting.gradient_tol is not None:
        kwargs["gradient_tol"] = config.fitting.gradient_tol
    if config.fitting.loss_tol is not None:
        kwargs["loss_tol"] = config.fitting.loss_tol
    return kwargs


def thurstonian_path_exists(cache_dir: Path, method: str, config: dict) -> tuple[Path, bool]:
    config_hash = _config_hash(config)
    base_path = cache_dir / f"thurstonian_{method}"
    full_path = cache_dir / f"thurstonian_{method}_{config_hash}.yaml"
    return base_path, full_path.exists()


def flip_pairs(pairs: list[tuple[Task, Task]]) -> list[tuple[Task, Task]]:
    return [(b, a) for a, b in pairs]


def parse_scale_from_template(template: PromptTemplate) -> tuple[int, int] | str:
    """Parse scale from template tags. Returns (min, max) or 'qualitative'."""
    scale_str = template.tags_dict["scale"]
    if scale_str == "qualitative":
        return "qualitative"
    min_str, max_str = scale_str.split("-")
    return int(min_str), int(max_str)
