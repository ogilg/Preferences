"""Post-task revealed preference measurement with active learning pair selection.

Usage: python -m src.running_measurements.run_post_task_active_learning <config.yaml>
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np

from dotenv import load_dotenv

load_dotenv()

from src.prompt_templates import PostTaskRevealedPromptBuilder
from src.preference_measurement import (
    measure_post_task_revealed,
    RevealedPreferenceMeasurer,
    CHOICE_FORMATS,
)
from src.thurstonian_fitting import compute_pair_agreement, save_thurstonian, _config_hash
from src.thurstonian_fitting.active_learning import (
    ActiveLearningState,
    generate_d_regular_pairs,
    select_next_pairs,
    check_convergence,
)
from src.measurement_storage import (
    CompletionStore,
    PostRevealedCache,
    model_short_name,
    save_yaml,
    reconstruct_measurements,
)
from src.prompt_templates.sampler import (
    SampledConfiguration,
    sample_configurations_lhs,
    print_sampling_balance,
)
from src.running_measurements.utils.experiment_utils import (
    parse_config_path,
    setup_experiment,
    compute_thurstonian_max_iter,
    build_fit_kwargs,
    flip_pairs,
)


class MeasurementError(Exception):
    pass


def main():
    ctx = setup_experiment(
        parse_config_path("Post-task active learning"),
        expected_mode="post_task_active_learning",
    )
    config, al = ctx.config, ctx.config.active_learning

    if al is None:
        raise ValueError("active_learning config section required for post_task_active_learning mode")

    if ctx.templates is None:
        raise ValueError("Templates required for post_task_active_learning mode")

    rng = np.random.default_rng(al.seed)
    max_iter = compute_thurstonian_max_iter(config)
    orders = ["canonical", "reversed"] if config.include_reverse_order else ["canonical"]
    n_total_pairs = config.n_tasks * (config.n_tasks - 1) // 2
    model_short = model_short_name(ctx.client.canonical_model_name)

    completion_seeds = config.completion_seeds or config.generation_seeds

    if config.template_sampling == "lhs" and config.n_template_samples:
        lhs_seed = config.lhs_seed if config.lhs_seed is not None else al.seed
        configurations = sample_configurations_lhs(
            ctx.templates, config.response_formats, config.generation_seeds,
            n_samples=config.n_template_samples, orders=orders, seed=lhs_seed,
        )
        print(f"LHS sampling: {config.n_template_samples} configurations")
        print_sampling_balance(configurations)
    else:
        configurations = [
            SampledConfiguration(t, rf, s, o)
            for t in ctx.templates for rf in config.response_formats
            for s in config.generation_seeds for o in orders
        ]

    total_runs = len(completion_seeds) * len(configurations)
    print(f"Tasks: {len(ctx.tasks)}, Configs: {len(configurations)}, "
          f"Completion seeds: {completion_seeds}, Initial degree: {al.initial_degree}")

    run_idx = 0
    for completion_seed in completion_seeds:
        store = CompletionStore(client=ctx.client, seed=completion_seed)
        if not store.exists():
            print(f"No completions for seed {completion_seed}, run completion generation first")
            continue

        task_completions = store.load(ctx.task_lookup)
        completion_lookup = {tc.task.id: tc.completion for tc in task_completions}

        # Filter tasks to only those with completions
        tasks_with_completions = [t for t in ctx.tasks if t.id in completion_lookup]
        if len(tasks_with_completions) < len(ctx.tasks):
            print(f"Note: {len(ctx.tasks) - len(tasks_with_completions)} tasks missing completions")

        for cfg in configurations:
            print(f"[PROGRESS {run_idx}/{total_runs}]", flush=True)
            run_idx += 1

            print(f"\n{'='*60}\n{cfg.template.name}/{cfg.response_format}/{cfg.order}"
                  f"/cseed{completion_seed}/rseed{cfg.seed}\n{'='*60}")

            cache = PostRevealedCache(
                model_short, cfg.template.name, cfg.response_format,
                cfg.order, completion_seed, cfg.seed,
            )
            run_config = {
                "n_tasks": config.n_tasks,
                "seed": al.seed,
                "completion_seed": completion_seed,
                "rating_seed": cfg.seed,
            }
            config_hash = _config_hash(run_config)
            base_path = cache.cache_dir / "thurstonian_active_learning"

            if (cache.cache_dir / f"thurstonian_active_learning_{config_hash}.yaml").exists():
                print(f"Already done (hash: {config_hash}), skipping")
                continue

            state = ActiveLearningState(tasks=tasks_with_completions)
            rank_correlations = []
            start_iteration = 0

            # Resume from cached measurements if available
            task_ids = {t.id for t in tasks_with_completions}
            cached_raw = cache.get_measurements(task_ids)
            if cached_raw:
                comparisons = reconstruct_measurements(cached_raw, ctx.task_lookup)
                state.add_comparisons(comparisons)
                state.fit(**build_fit_kwargs(config, max_iter))
                start_iteration = len(state.sampled_pairs) // al.batch_size
                print(f"Resuming: {len(comparisons)} cached measurements, ~{start_iteration} iterations")
                pairs_to_query = select_next_pairs(
                    state, batch_size=al.batch_size,
                    p_threshold=al.p_threshold, q_threshold=al.q_threshold, rng=rng,
                )
            else:
                pairs_to_query = generate_d_regular_pairs(tasks_with_completions, al.initial_degree, rng)

            if cfg.order == "reversed":
                pairs_to_query = flip_pairs(pairs_to_query)

            # Get labels from template tags
            label_a = cfg.template.tags_dict.get("task_a_label", "Task A")
            label_b = cfg.template.tags_dict.get("task_b_label", "Task B")

            builder = PostTaskRevealedPromptBuilder(
                measurer=RevealedPreferenceMeasurer(),
                response_format=CHOICE_FORMATS[cfg.response_format](label_a, label_b),
                template=cfg.template,
            )

            measurement_config = {
                "model": ctx.client.model_name,
                "template_name": cfg.template.name,
                "template_tags": dict(cfg.template.tags_dict),
                "response_format": cfg.response_format,
                "order": cfg.order,
                "completion_seed": completion_seed,
                "rating_seed": cfg.seed,
                "temperature": config.temperature,
            }

            def measure_fn(data: list[tuple]) -> "MeasurementBatch":
                return measure_post_task_revealed(
                    client=ctx.client,
                    data=data,
                    builder=builder,
                    temperature=config.temperature,
                    max_concurrent=ctx.max_concurrent,
                    seed=cfg.seed,
                )

            for iteration in range(start_iteration, al.max_iterations):
                if not pairs_to_query:
                    break

                print(f"\nIteration {iteration + 1}: {len(pairs_to_query)} pairs")

                batch, cache_hits, api_queries = cache.get_or_measure_post_task(
                    pairs_to_query * config.n_samples,
                    completion_lookup,
                    measure_fn,
                    ctx.task_lookup,
                    measurement_config,
                )
                print(f"  {len(batch.successes)} measurements ({cache_hits} cached)")

                if api_queries > 0 and len(batch.successes) == cache_hits:
                    raise MeasurementError(
                        f"All {api_queries} API requests failed. Check API key and credits."
                    )

                state.add_comparisons(batch.successes)
                state.iteration = iteration + 1
                state.fit(**build_fit_kwargs(config, max_iter))

                converged, correlation = check_convergence(state, al.convergence_threshold)
                if math.isnan(correlation):
                    raise MeasurementError(
                        f"Correlation is NaN at iteration {iteration + 1}. "
                        "Model fitting likely failed due to insufficient data."
                    )
                rank_correlations.append(float(correlation))
                print(f"  Converged: {state.current_fit.converged}, correlation: {correlation:.4f}")

                if converged:
                    print(f"*** Converged at iteration {iteration + 1} ***")
                    break

                pairs_to_query = select_next_pairs(
                    state, batch_size=al.batch_size,
                    p_threshold=al.p_threshold, q_threshold=al.q_threshold, rng=rng,
                )
                if cfg.order == "reversed":
                    pairs_to_query = flip_pairs(pairs_to_query)

            final_converged, _ = check_convergence(state, al.convergence_threshold)
            print(f"\nFinal: {len(state.sampled_pairs)}/{n_total_pairs} pairs, "
                  f"agreement: {compute_pair_agreement(state.comparisons):.3f}")

            save_thurstonian(state.current_fit, base_path.with_suffix(".yaml"), "active_learning", run_config)
            save_yaml({
                "n_tasks": config.n_tasks,
                "seed": al.seed,
                "completion_seed": completion_seed,
                "rating_seed": cfg.seed,
                "converged": bool(final_converged),
                "n_iterations": state.iteration,
                "unique_pairs_queried": len(state.sampled_pairs),
                "total_comparisons": len(state.comparisons),
                "pair_agreement": float(compute_pair_agreement(state.comparisons)),
                "rank_correlations": rank_correlations,
            }, cache.cache_dir / "active_learning.yaml")

    print(f"[PROGRESS {total_runs}/{total_runs}]", flush=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
