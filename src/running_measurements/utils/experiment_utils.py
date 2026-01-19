"""Shared utilities for experiment runners."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.models import get_client, get_default_max_concurrent, OpenAICompatibleClient
from src.task_data import Task, load_tasks
from src.prompt_templates import load_templates_from_yaml, PromptTemplate
from src.prompt_templates.sampler import SampledConfiguration, sample_configurations_lhs
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
    require_templates: bool = True,
) -> ExperimentContext:
    config = load_experiment_config(config_path)

    if config.preference_mode != expected_mode:
        raise ValueError(f"Expected preference_mode='{expected_mode}', got '{config.preference_mode}'")

    # Load tasks deterministically so stated and revealed use same tasks
    tasks = load_tasks(n=config.n_tasks, origins=config.get_origin_datasets(), seed=None)

    # Templates are optional for completion_generation mode
    templates = None
    if config.templates is not None:
        templates = load_templates_from_yaml(config.templates)

    if require_templates and templates is None:
        raise ValueError(f"Templates required for {expected_mode} mode")

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


def shuffle_pair_order(
    pairs: list[tuple[Task, Task]], seed: int
) -> list[tuple[Task, Task]]:
    """Randomly flip each pair's order based on seed. Deterministic for same seed."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return [
        (b, a) if rng.random() < 0.5 else (a, b)
        for a, b in pairs
    ]


def apply_pair_order(
    pairs: list[tuple[Task, Task]],
    order: str,
    pair_order_seed: int | None,
) -> list[tuple[Task, Task]]:
    """Apply pair ordering: canonical, reversed, or random shuffle."""
    if pair_order_seed is not None:
        return shuffle_pair_order(pairs, pair_order_seed)
    if order == "reversed":
        return flip_pairs(pairs)
    return pairs


QUALITATIVE_SCALES = {
    "binary": ["good", "bad"],
    "ternary": ["good", "neutral", "bad"],
}


def parse_scale_from_template(template: PromptTemplate) -> tuple[int, int] | list[str]:
    """Parse scale from template tags.

    Returns:
        tuple[int, int]: For numeric scales like "1-10"
        list[str]: For qualitative scales like ["negative", "neutral", "positive"]
    """
    scale = template.tags_dict["scale"]
    if isinstance(scale, list):
        return scale
    if scale in QUALITATIVE_SCALES:
        return QUALITATIVE_SCALES[scale]
    if "-" in scale:
        min_str, max_str = scale.split("-")
        return int(min_str), int(max_str)
    raise ValueError(f"Unknown scale format: {scale}")


def build_configurations(
    ctx: ExperimentContext,
    config: ExperimentConfig,
    include_order: bool = False,
) -> list[SampledConfiguration]:
    """Build template configurations using LHS or full sampling."""
    orders = ["canonical", "reversed"] if config.include_reverse_order else ["canonical"]

    if config.template_sampling == "lhs" and config.n_template_samples:
        lhs_seed = config.lhs_seed if config.lhs_seed is not None else 42
        return sample_configurations_lhs(
            ctx.templates, config.response_formats, config.generation_seeds,
            n_samples=config.n_template_samples,
            orders=orders if include_order else None,
            seed=lhs_seed,
        )

    if include_order:
        return [
            SampledConfiguration(t, rf, s, o)
            for t in ctx.templates for rf in config.response_formats
            for s in config.generation_seeds for o in orders
        ]
    return [
        SampledConfiguration(t, rf, s)
        for t in ctx.templates for rf in config.response_formats for s in config.generation_seeds
    ]
