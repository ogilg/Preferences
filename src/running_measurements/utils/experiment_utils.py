"""Shared utilities for experiment runners."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich import print as rprint

from src.models import get_client, get_default_max_concurrent, OpenAICompatibleClient
from src.models.registry import get_model_system_prompt
from src.task_data import Task, load_tasks
from src.prompt_templates import load_templates_from_yaml, parse_template_dict, PromptTemplate
from src.prompt_templates.sampler import SampledConfiguration, sample_configurations_lhs
from src.thurstonian_fitting import _config_hash
from src.running_measurements.config import load_experiment_config, ExperimentConfig, get_experiment_id
from src.measurement_storage.loading import get_activation_task_ids


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
    max_new_tokens: int | None = None,
    require_templates: bool = True,
    config_overrides: dict | None = None,
) -> ExperimentContext:
    config = load_experiment_config(config_path)

    # Apply CLI overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Use config's max_new_tokens unless explicitly overridden by caller
    if max_new_tokens is None:
        max_new_tokens = config.max_new_tokens

    # CLI experiment_id (set via set_experiment_id) overrides config file
    global_exp_id = get_experiment_id()
    if global_exp_id is not None:
        config.experiment_id = global_exp_id

    if config.preference_mode != expected_mode:
        raise ValueError(f"Expected preference_mode='{expected_mode}', got '{config.preference_mode}'")

    # Prepare filter function if using activation tasks
    filter_fn = None
    if config.use_tasks_with_activations:
        activation_task_ids = get_activation_task_ids()
        if activation_task_ids:
            filter_fn = lambda t: t.id in activation_task_ids
        else:
            rprint("[yellow]Warning: use_tasks_with_activations=true but probe_data/activations/completions.json not found[/yellow]")

    # Load tasks deterministically so stated and revealed use same tasks
    # If filter_fn is set, this loads n_tasks from the filtered set
    tasks = load_tasks(
        n=config.n_tasks,
        origins=config.get_origin_datasets(),
        seed=config.task_sampling_seed,
        filter_fn=filter_fn,
    )

    if config.use_tasks_with_activations and filter_fn:
        total_activation_tasks = len(get_activation_task_ids())
        rprint(f"[dim]Loaded {len(tasks)} tasks from {total_activation_tasks} total with activations[/dim]")

    # Templates: inline takes precedence over file path
    templates = None
    if config.inline_templates is not None:
        templates = [parse_template_dict(t) for t in config.inline_templates]
    elif config.templates is not None:
        templates = load_templates_from_yaml(config.templates)

    if require_templates and templates is None:
        raise ValueError(f"Templates required for {expected_mode} mode")

    # Apply model's default system prompt if config doesn't override
    if config.measurement_system_prompt is None:
        model_sys_prompt = get_model_system_prompt(config.model)
        if model_sys_prompt:
            config.measurement_system_prompt = model_sys_prompt

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
    rng = np.random.default_rng(seed)
    return [
        (b, a) if rng.random() < 0.5 else (a, b)
        for a, b in pairs
    ]


def apply_pair_order(
    pairs: list[tuple[Task, Task]],
    order: str,
    pair_order_seed: int | None,
    include_reverse_order: bool,
) -> list[tuple[Task, Task]]:
    """Apply pair ordering: explicit both orders or random shuffle.

    When include_reverse_order=True, we run both canonical and reversed explicitly,
    so no shuffling. When False, shuffle pairs randomly per pair_order_seed.
    """
    if include_reverse_order:
        if order == "reversed":
            return flip_pairs(pairs)
        return pairs  # canonical
    # Shuffle randomly when not running both orders explicitly
    assert pair_order_seed is not None, "pair_order_seed must be set when include_reverse_order=False"
    return shuffle_pair_order(pairs, pair_order_seed)


QUALITATIVE_SCALES = {
    "binary": ["good", "bad"],
    "ternary": ["good", "neutral", "bad"],
}


def parse_scale_from_template(template: PromptTemplate) -> tuple[int, int] | list[str]:
    """Parse scale from template tags. Returns (min, max) or list of labels."""
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
