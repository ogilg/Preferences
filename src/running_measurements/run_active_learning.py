"""Active learning for efficient Thurstonian fitting.

Usage: python -m src.experiments.run_active_learning <config.yaml>
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np

from src.preference_measurement import measure_revealed_with_template


class MeasurementError(Exception):
    pass


from src.thurstonian_fitting import compute_pair_agreement, save_thurstonian, _config_hash
from src.thurstonian_fitting.active_learning import (
    ActiveLearningState,
    generate_d_regular_pairs,
    select_next_pairs,
    check_convergence,
)
from src.measurement_storage import MeasurementCache, save_yaml, load_yaml, reconstruct_measurements
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


def main():
    ctx = setup_experiment(parse_config_path("Active learning for Thurstonian fitting"), "active_learning")
    config, al = ctx.config, ctx.config.active_learning

    rng = np.random.default_rng(al.seed)
    max_iter = compute_thurstonian_max_iter(config)
    orders = ["canonical", "reversed"] if config.include_reverse_order else ["canonical"]
    n_total_pairs = config.n_tasks * (config.n_tasks - 1) // 2

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

    print(f"Tasks: {len(ctx.tasks)}, Configs: {len(configurations)}, Initial degree: {al.initial_degree}")

    for cfg_idx, cfg in enumerate(configurations):
        print(f"[PROGRESS {cfg_idx}/{len(configurations)}]", flush=True)
        print(f"\n{'='*60}\n{cfg.template.name}/{cfg.response_format}/{cfg.order}/seed{cfg.seed}\n{'='*60}")

        cache = MeasurementCache(cfg.template, ctx.client, cfg.response_format, cfg.order, seed=cfg.seed)
        run_config = {"n_tasks": config.n_tasks, "seed": al.seed, "generation_seed": cfg.seed}
        config_hash = _config_hash(run_config)
        base_path = cache.cache_dir / "thurstonian_active_learning"

        if (cache.cache_dir / f"thurstonian_active_learning_{config_hash}.yaml").exists():
            print(f"Already done (hash: {config_hash}), skipping")
            continue

        state = ActiveLearningState(tasks=ctx.tasks)
        rank_correlations = []
        start_iteration = 0

        # Resume from cached measurements if available
        task_ids = {t.id for t in ctx.tasks}
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
            pairs_to_query = generate_d_regular_pairs(ctx.tasks, al.initial_degree, rng)

        if cfg.order == "reversed":
            pairs_to_query = flip_pairs(pairs_to_query)

        for iteration in range(start_iteration, al.max_iterations):
            if not pairs_to_query:
                break

            print(f"\nIteration {iteration + 1}: {len(pairs_to_query)} pairs")

            measure_fn = partial(
                measure_revealed_with_template, cfg.template, ctx.client,
                temperature=config.temperature, max_concurrent=ctx.max_concurrent,
                response_format_name=cfg.response_format, seed=cfg.seed,
            )
            batch, cache_hits, api_queries = cache.get_or_measure(
                pairs_to_query * config.n_samples, measure_fn, ctx.task_lookup
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
        print(f"\nFinal: {len(state.sampled_pairs)}/{n_total_pairs} pairs, agreement: {compute_pair_agreement(state.comparisons):.3f}")

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

    print(f"[PROGRESS {len(configurations)}/{len(configurations)}]", flush=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
