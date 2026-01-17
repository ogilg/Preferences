"""Async experiment runners for shared semaphore usage."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TypedDict

import numpy as np

from src.prompt_templates import PostTaskStatedPromptBuilder, PostTaskRevealedPromptBuilder, PreTaskRevealedPromptBuilder, PreTaskStatedPromptBuilder
from src.prompt_templates.generator_config import TASK_LABELS
from src.preference_measurement import (
    measure_post_task_stated_async,
    measure_post_task_revealed_async,
    measure_pre_task_stated_async,
    measure_pre_task_revealed_async,
    StatedScoreMeasurer,
    RevealedPreferenceMeasurer,
    get_stated_response_format,
    get_revealed_response_format,
)
from src.measurement_storage import (
    CompletionStore, PostStatedCache, PostRevealedCache, model_short_name,
    save_stated, stated_exist, MeasurementCache, save_yaml, reconstruct_measurements,
)
from src.measurement_storage.completions import generate_completions
from src.measurement_storage.base import build_measurement_config
from src.thurstonian_fitting import compute_pair_agreement, save_thurstonian, _config_hash
from src.thurstonian_fitting.active_learning import (
    ActiveLearningState,
    generate_d_regular_pairs,
    select_next_pairs,
    check_convergence,
)
from src.running_measurements.utils.experiment_utils import (
    setup_experiment,
    parse_scale_from_template,
    flip_pairs,
    compute_thurstonian_max_iter,
    build_fit_kwargs,
    build_configurations,
)


@dataclass
class RunnerStats:
    total_runs: int = 0
    completed: int = 0
    successes: int = 0
    failures: int = 0

    def to_dict(self) -> dict:
        return {"total_runs": self.total_runs, "successes": self.successes, "failures": self.failures}

    def mark_skipped(self) -> None:
        self.completed += 1

    def add_batch(self, n_successes: int, n_failures: int) -> None:
        self.successes += n_successes
        self.failures += n_failures
        self.completed += 1


class PostTaskRunConfig(TypedDict):
    model: str
    template_name: str
    template_tags: dict
    response_format: str
    completion_seed: int
    rating_seed: int
    temperature: float


class PostTaskRevealedRunConfig(PostTaskRunConfig):
    order: str


def build_stated_builder(template, response_format_name: str, post_task: bool = False):
    """Build a stated preference prompt builder."""
    response_format = get_stated_response_format(parse_scale_from_template(template), response_format_name)
    builder_cls = PostTaskStatedPromptBuilder if post_task else PreTaskStatedPromptBuilder
    return builder_cls(
        measurer=StatedScoreMeasurer(),
        response_format=response_format,
        template=template,
    )


def build_revealed_builder(template, response_format_name: str, post_task: bool = False):
    """Build a revealed preference prompt builder."""
    tags = template.tags_dict
    language = tags.get("language", "en")
    task_label_names = tags.get("task_label_names", "letter")
    task_a_label, task_b_label = TASK_LABELS[(task_label_names, language)]
    response_format = get_revealed_response_format(task_a_label, task_b_label, response_format_name)
    builder_cls = PostTaskRevealedPromptBuilder if post_task else PreTaskRevealedPromptBuilder
    return builder_cls(
        measurer=RevealedPreferenceMeasurer(),
        response_format=response_format,
        template=template,
    )


async def run_post_task_stated_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run post-task stated measurement with shared semaphore."""
    ctx = setup_experiment(config_path, expected_mode="post_task_stated")
    config = ctx.config

    completion_seeds = config.completion_seeds or config.generation_seeds
    model_short = model_short_name(ctx.client.canonical_model_name)

    configurations = build_configurations(ctx, config)
    stats = RunnerStats(total_runs=len(completion_seeds) * len(configurations))

    for completion_seed in completion_seeds:
        store = CompletionStore(client=ctx.client, seed=completion_seed)
        if not store.exists():
            continue

        task_completions = store.load(ctx.task_lookup)
        data = [(tc.task, tc.completion) for tc in task_completions] * config.n_samples

        for cfg in configurations:
            cache = PostStatedCache(
                model_short, cfg.template.name, cfg.response_format,
                completion_seed, cfg.seed,
            )

            if cache.exists():
                stats.mark_skipped()
                if progress_callback:
                    progress_callback(stats.completed, stats.total_runs)
                continue

            builder = build_stated_builder(cfg.template, cfg.response_format, post_task=True)

            batch = await measure_post_task_stated_async(
                client=ctx.client,
                data=data,
                builder=builder,
                semaphore=semaphore,
                temperature=config.temperature,
                seed=cfg.seed,
            )

            stats.add_batch(len(batch.successes), len(batch.failures))

            run_config: PostTaskRunConfig = {
                "model": ctx.client.model_name,
                "template_name": cfg.template.name,
                "template_tags": dict(cfg.template.tags_dict),
                "response_format": cfg.response_format,
                "completion_seed": completion_seed,
                "rating_seed": cfg.seed,
                "temperature": config.temperature,
            }
            cache.save(batch.successes, run_config)

            if progress_callback:
                progress_callback(stats.completed, stats.total_runs)

    return stats.to_dict()


async def run_pre_task_revealed_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run pre-task revealed measurement with shared semaphore."""
    ctx = setup_experiment(config_path, expected_mode="pre_task_revealed")
    config = ctx.config

    configurations = build_configurations(ctx, config, include_order=True)
    all_pairs = list(combinations(ctx.tasks, 2))
    stats = RunnerStats(total_runs=len(configurations))

    for cfg in configurations:
        cache = MeasurementCache(cfg.template, ctx.client, cfg.response_format, cfg.order, seed=cfg.seed)

        existing_pairs = cache.get_existing_pairs()
        pairs = all_pairs if cfg.order == "canonical" else flip_pairs(all_pairs)
        pairs_to_query = [
            (a, b) for a, b in pairs
            if (a.id, b.id) not in existing_pairs
        ]
        pairs_to_query = pairs_to_query * config.n_samples

        if not pairs_to_query:
            stats.mark_skipped()
            if progress_callback:
                progress_callback(stats.completed, stats.total_runs)
            continue

        builder = build_revealed_builder(cfg.template, cfg.response_format, post_task=False)

        batch = await measure_pre_task_revealed_async(
            client=ctx.client,
            pairs=pairs_to_query,
            builder=builder,
            semaphore=semaphore,
            temperature=config.temperature,
            seed=cfg.seed,
        )

        stats.add_batch(len(batch.successes), len(batch.failures))
        cache.append(batch.successes)

        if progress_callback:
            progress_callback(stats.completed, stats.total_runs)

    return stats.to_dict()


async def run_post_task_revealed_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run post-task revealed measurement with shared semaphore."""
    ctx = setup_experiment(config_path, expected_mode="post_task_revealed")
    config = ctx.config

    completion_seeds = config.completion_seeds or config.generation_seeds
    model_short = model_short_name(ctx.client.canonical_model_name)
    configurations = build_configurations(ctx, config, include_order=True)
    all_pairs = list(combinations(ctx.tasks, 2))
    stats = RunnerStats(total_runs=len(completion_seeds) * len(configurations))

    for completion_seed in completion_seeds:
        store = CompletionStore(client=ctx.client, seed=completion_seed)
        if not store.exists():
            continue

        task_completions = store.load(ctx.task_lookup)
        completion_lookup = {tc.task.id: tc.completion for tc in task_completions}

        for cfg in configurations:
            cache = PostRevealedCache(
                model_short, cfg.template.name, cfg.response_format,
                cfg.order, completion_seed, cfg.seed,
            )

            existing_pairs = cache.get_existing_pairs()
            pairs = all_pairs if cfg.order == "canonical" else flip_pairs(all_pairs)
            pairs_to_query = [
                (a, b) for a, b in pairs
                if (a.id, b.id) not in existing_pairs
            ]
            pairs_to_query = pairs_to_query * config.n_samples

            if not pairs_to_query:
                stats.mark_skipped()
                if progress_callback:
                    progress_callback(stats.completed, stats.total_runs)
                continue

            data = [
                (task_a, task_b, completion_lookup[task_a.id], completion_lookup[task_b.id])
                for task_a, task_b in pairs_to_query
            ]

            builder = build_revealed_builder(cfg.template, cfg.response_format, post_task=True)

            batch = await measure_post_task_revealed_async(
                client=ctx.client,
                data=data,
                builder=builder,
                semaphore=semaphore,
                temperature=config.temperature,
                seed=cfg.seed,
            )

            stats.add_batch(len(batch.successes), len(batch.failures))

            run_config: PostTaskRevealedRunConfig = {
                "model": ctx.client.model_name,
                "template_name": cfg.template.name,
                "template_tags": dict(cfg.template.tags_dict),
                "response_format": cfg.response_format,
                "order": cfg.order,
                "completion_seed": completion_seed,
                "rating_seed": cfg.seed,
                "temperature": config.temperature,
            }
            cache.append(batch.successes, run_config)

            if progress_callback:
                progress_callback(stats.completed, stats.total_runs)

    return stats.to_dict()


async def run_pre_task_stated_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run pre-task stated measurement with shared semaphore."""
    ctx = setup_experiment(config_path, expected_mode="pre_task_stated")
    config = ctx.config

    configurations = build_configurations(ctx, config)
    tasks_data = ctx.tasks * config.n_samples
    stats = RunnerStats(total_runs=len(configurations))

    for cfg in configurations:
        if stated_exist(cfg.template, ctx.client, cfg.response_format, cfg.seed):
            stats.mark_skipped()
            if progress_callback:
                progress_callback(stats.completed, stats.total_runs)
            continue

        builder = build_stated_builder(cfg.template, cfg.response_format, post_task=False)

        batch = await measure_pre_task_stated_async(
            client=ctx.client,
            tasks=tasks_data,
            builder=builder,
            semaphore=semaphore,
            temperature=config.temperature,
            seed=cfg.seed,
        )

        stats.add_batch(len(batch.successes), len(batch.failures))

        config_dict = build_measurement_config(
            template=cfg.template,
            client=ctx.client,
            response_format=cfg.response_format,
            seed=cfg.seed,
            temperature=config.temperature,
        )

        save_stated(
            template=cfg.template,
            client=ctx.client,
            scores=batch.successes,
            response_format=cfg.response_format,
            seed=cfg.seed,
            config=config_dict,
        )

        if progress_callback:
            progress_callback(stats.completed, stats.total_runs)

    return stats.to_dict()


async def _measure_revealed_pairs(
    pairs: list,
    builder,
    client,
    semaphore: asyncio.Semaphore,
    temperature: float,
    seed: int,
    completion_lookup: dict[str, str] | None = None,
):
    """Measure revealed preferences for pairs, with or without completions."""
    if completion_lookup is not None:
        data = [
            (a, b, completion_lookup[a.id], completion_lookup[b.id])
            for a, b in pairs
        ]
        return await measure_post_task_revealed_async(
            client=client, data=data, builder=builder,
            semaphore=semaphore, temperature=temperature, seed=seed,
        )
    else:
        return await measure_pre_task_revealed_async(
            client=client, pairs=pairs, builder=builder,
            semaphore=semaphore, temperature=temperature, seed=seed,
        )


async def run_active_learning_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: Callable[[int, int], None] | None = None,
    post_task: bool = False,
) -> dict:
    """Run active learning with shared semaphore. Supports both pre-task and post-task modes."""
    expected_mode = "post_task_active_learning" if post_task else "pre_task_active_learning"
    ctx = setup_experiment(config_path, expected_mode=expected_mode)
    config, al = ctx.config, ctx.config.active_learning

    # For post-task, load completions
    completion_lookup: dict[str, str] | None = None
    if post_task:
        completion_seeds = config.completion_seeds or config.generation_seeds
        # Use first completion seed for active learning
        store = CompletionStore(client=ctx.client, seed=completion_seeds[0])
        if not store.exists():
            raise ValueError(f"Completions not found for seed {completion_seeds[0]}")
        task_completions = store.load(ctx.task_lookup)
        completion_lookup = {tc.task.id: tc.completion for tc in task_completions}

    rng = np.random.default_rng(al.seed)
    max_iter = compute_thurstonian_max_iter(config)
    configurations = build_configurations(ctx, config, include_order=True)
    stats = RunnerStats(total_runs=len(configurations))

    for cfg in configurations:
        cache = MeasurementCache(cfg.template, ctx.client, cfg.response_format, cfg.order, seed=cfg.seed)
        run_config = {"n_tasks": config.n_tasks, "seed": al.seed, "generation_seed": cfg.seed}
        config_hash = _config_hash(run_config)

        if (cache.cache_dir / f"thurstonian_active_learning_{config_hash}.yaml").exists():
            stats.mark_skipped()
            if progress_callback:
                progress_callback(stats.completed, stats.total_runs)
            continue

        state = ActiveLearningState(tasks=ctx.tasks)
        start_iteration = 0

        # Resume from cached measurements if available
        task_ids = {t.id for t in ctx.tasks}
        cached_raw = cache.get_measurements(task_ids)
        if cached_raw:
            comparisons = reconstruct_measurements(cached_raw, ctx.task_lookup)
            state.add_comparisons(comparisons)
            state.fit(**build_fit_kwargs(config, max_iter))
            start_iteration = len(state.sampled_pairs) // al.batch_size
            pairs_to_query = select_next_pairs(
                state, batch_size=al.batch_size,
                p_threshold=al.p_threshold, q_threshold=al.q_threshold, rng=rng,
            )
        else:
            pairs_to_query = generate_d_regular_pairs(ctx.tasks, al.initial_degree, rng)

        if cfg.order == "reversed":
            pairs_to_query = flip_pairs(pairs_to_query)

        builder = build_revealed_builder(cfg.template, cfg.response_format, post_task=post_task)
        rank_correlations = []

        for iteration in range(start_iteration, al.max_iterations):
            if not pairs_to_query:
                break

            existing_pairs = cache.get_existing_pairs()
            pairs_with_repeats = pairs_to_query * config.n_samples
            to_query = [(a, b) for a, b in pairs_with_repeats if (a.id, b.id) not in existing_pairs]

            if to_query:
                batch = await _measure_revealed_pairs(
                    pairs=to_query,
                    builder=builder,
                    client=ctx.client,
                    semaphore=semaphore,
                    temperature=config.temperature,
                    seed=cfg.seed,
                    completion_lookup=completion_lookup,
                )
                stats.successes += len(batch.successes)
                stats.failures += len(batch.failures)
                cache.append(batch.successes)
                state.add_comparisons(batch.successes)

            state.iteration = iteration + 1
            state.fit(**build_fit_kwargs(config, max_iter))

            converged, correlation = check_convergence(state, al.convergence_threshold)
            if math.isnan(correlation):
                break
            rank_correlations.append(float(correlation))

            if converged:
                break

            pairs_to_query = select_next_pairs(
                state, batch_size=al.batch_size,
                p_threshold=al.p_threshold, q_threshold=al.q_threshold, rng=rng,
            )
            if cfg.order == "reversed":
                pairs_to_query = flip_pairs(pairs_to_query)

        stats.completed += 1
        final_converged, _ = check_convergence(state, al.convergence_threshold)

        base_path = cache.cache_dir / "thurstonian_active_learning"
        save_thurstonian(state.current_fit, base_path.with_suffix(".yaml"), "active_learning", run_config)
        save_yaml({
            "n_tasks": config.n_tasks,
            "seed": al.seed,
            "generation_seed": cfg.seed,
            "converged": bool(final_converged),
            "n_iterations": state.iteration,
            "unique_pairs_queried": len(state.sampled_pairs),
            "total_comparisons": len(state.comparisons),
            "pair_agreement": float(compute_pair_agreement(state.comparisons)),
            "rank_correlations": rank_correlations,
        }, cache.cache_dir / "active_learning.yaml")

        if progress_callback:
            progress_callback(stats.completed, stats.total_runs)

    return stats.to_dict()


async def run_post_task_active_learning_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run post-task active learning with shared semaphore."""
    return await run_active_learning_async(config_path, semaphore, progress_callback, post_task=True)


async def run_pre_task_active_learning_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run pre-task active learning with shared semaphore."""
    return await run_active_learning_async(config_path, semaphore, progress_callback, post_task=False)


async def run_completion_generation_async(
    config_path: Path,
    semaphore: asyncio.Semaphore,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run completion generation with shared semaphore."""
    ctx = setup_experiment(config_path, expected_mode="completion_generation", max_new_tokens=1024, require_templates=False)
    config = ctx.config

    stats = RunnerStats(total_runs=len(config.generation_seeds))

    for seed in config.generation_seeds:
        store = CompletionStore(client=ctx.client, seed=seed)

        existing_ids = store.get_existing_task_ids()
        tasks_to_complete = [t for t in ctx.tasks if t.id not in existing_ids]

        if not tasks_to_complete:
            stats.mark_skipped()
            if progress_callback:
                progress_callback(stats.completed, stats.total_runs)
            continue

        # generate_completions is sync, run in executor
        loop = asyncio.get_event_loop()
        completions = await loop.run_in_executor(
            None,
            lambda: generate_completions(
                client=ctx.client,
                tasks=tasks_to_complete,
                temperature=config.temperature,
                max_concurrent=ctx.max_concurrent,
                seed=seed,
            )
        )

        run_config = {
            "model": ctx.client.model_name,
            "n_tasks": config.n_tasks,
            "task_origins": config.task_origins,
            "temperature": config.temperature,
            "seed": seed,
        }
        store.save(completions, run_config)

        stats.add_batch(len(completions), len(tasks_to_complete) - len(completions))

        if progress_callback:
            progress_callback(stats.completed, stats.total_runs)

    return stats.to_dict()


RUNNERS = {
    "completion_generation": run_completion_generation_async,
    "pre_task_stated": run_pre_task_stated_async,
    "pre_task_revealed": run_pre_task_revealed_async,
    "pre_task_active_learning": run_pre_task_active_learning_async,
    "post_task_stated": run_post_task_stated_async,
    "post_task_revealed": run_post_task_revealed_async,
    "post_task_active_learning": run_post_task_active_learning_async,
}
