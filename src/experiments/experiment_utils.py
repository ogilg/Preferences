"""Shared utilities for experiment runners."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.models import get_client, get_default_max_concurrent, OpenAICompatibleClient
from src.task_data import Task, load_tasks
from src.preferences.templates import load_templates_from_yaml, PromptTemplate
from src.preferences.ranking import _config_hash
from src.experiments.config import load_experiment_config, ExperimentConfig


@dataclass
class ExperimentContext:
    config: ExperimentConfig
    templates: list[PromptTemplate]
    tasks: list[Task]
    task_lookup: dict[str, Task]
    client: OpenAICompatibleClient
    max_concurrent: int


def parse_config_path(description: str) -> Path:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", type=Path, help="Path to experiment config YAML")
    return parser.parse_args().config


def setup_experiment(config_path: Path, expected_mode: str) -> ExperimentContext:
    config = load_experiment_config(config_path)

    if config.preference_mode != expected_mode:
        raise ValueError(f"Expected preference_mode='{expected_mode}', got '{config.preference_mode}'")

    task_seed = config.active_learning.seed if config.active_learning else None
    tasks = load_tasks(n=config.n_tasks, origins=config.get_origin_datasets(), seed=task_seed)

    return ExperimentContext(
        config=config,
        templates=load_templates_from_yaml(config.templates),
        tasks=tasks,
        task_lookup={t.id: t for t in tasks},
        client=get_client(model_name=config.model),
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
