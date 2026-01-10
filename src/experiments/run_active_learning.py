"""Active learning for efficient Thurstonian fitting.

Usage: python -m src.experiments.run_active_learning <config.yaml>
"""

from __future__ import annotations

from functools import partial

import numpy as np

from src.preferences.measurement import measure_with_template
from src.preferences.ranking import compute_pair_agreement, save_thurstonian, _config_hash
from src.preferences.ranking.active_learning import (
    ActiveLearningState,
    generate_d_regular_pairs,
    select_next_pairs,
    check_convergence,
)
from src.preferences.storage import MeasurementCache, save_yaml
from src.preferences.templates.sampler import sample_configurations_lhs, print_sampling_balance
from src.experiments.experiment_utils import (
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
        configurations = sample_configurations_lhs(
            ctx.templates, config.response_formats, orders, config.generation_seeds,
            n_samples=config.n_template_samples, seed=al.seed,
        )
        print(f"LHS sampling: {config.n_template_samples} configurations")
        print_sampling_balance(configurations)
    else:
        configurations = [
            (t, rf, o, s)
            for t in ctx.templates for rf in config.response_formats
            for o in orders for s in config.generation_seeds
        ]

    print(f"Tasks: {len(ctx.tasks)}, Configs: {len(configurations)}, Initial degree: {al.initial_degree}")

    for template, response_format, order, gen_seed in configurations:
        print(f"\n{'='*60}\n{template.name}/{response_format}/{order}/seed{gen_seed}\n{'='*60}")

        cache = MeasurementCache(template, ctx.client, response_format, order, seed=gen_seed)
        run_config = {"n_tasks": config.n_tasks, "seed": al.seed, "generation_seed": gen_seed}
        config_hash = _config_hash(run_config)
        base_path = cache.cache_dir / "thurstonian_active_learning"

        if (cache.cache_dir / f"thurstonian_active_learning_{config_hash}.yaml").exists():
            print(f"Already done (hash: {config_hash}), skipping")
            continue

        state = ActiveLearningState(tasks=ctx.tasks)
        rank_correlations = []

        pairs_to_query = generate_d_regular_pairs(ctx.tasks, al.initial_degree, rng)
        if order == "reversed":
            pairs_to_query = flip_pairs(pairs_to_query)

        for iteration in range(al.max_iterations):
            if not pairs_to_query:
                break

            print(f"\nIteration {iteration + 1}: {len(pairs_to_query)} pairs")

            measure_fn = partial(
                measure_with_template, template, ctx.client,
                temperature=config.temperature, max_concurrent=ctx.max_concurrent,
                response_format_name=response_format, seed=gen_seed,
            )
            batch, cache_hits, _ = cache.get_or_measure(
                pairs_to_query * config.samples_per_pair, measure_fn, ctx.task_lookup
            )
            print(f"  {len(batch.successes)} measurements ({cache_hits} cached)")

            state.add_comparisons(batch.successes)
            state.iteration = iteration + 1
            state.fit(**build_fit_kwargs(config, max_iter))

            converged, correlation = check_convergence(state, al.convergence_threshold)
            rank_correlations.append(float(correlation))
            print(f"  Converged: {state.current_fit.converged}, correlation: {correlation:.4f}")

            if converged:
                print(f"*** Converged at iteration {iteration + 1} ***")
                break

            pairs_to_query = select_next_pairs(
                state, batch_size=al.batch_size,
                p_threshold=al.p_threshold, q_threshold=al.q_threshold, rng=rng,
            )
            if order == "reversed":
                pairs_to_query = flip_pairs(pairs_to_query)

        final_converged, _ = check_convergence(state, al.convergence_threshold)
        print(f"\nFinal: {len(state.sampled_pairs)}/{n_total_pairs} pairs, agreement: {compute_pair_agreement(state.comparisons):.3f}")

        save_thurstonian(state.current_fit, base_path.with_suffix(".yaml"), "active_learning", run_config)
        save_yaml({
            "n_tasks": config.n_tasks,
            "seed": al.seed,
            "generation_seed": gen_seed,
            "converged": bool(final_converged),
            "n_iterations": state.iteration,
            "unique_pairs_queried": len(state.sampled_pairs),
            "total_comparisons": len(state.comparisons),
            "pair_agreement": float(compute_pair_agreement(state.comparisons)),
            "rank_correlations": rank_correlations,
        }, cache.cache_dir / "active_learning.yaml")

    print("\nDone.")


if __name__ == "__main__":
    main()
