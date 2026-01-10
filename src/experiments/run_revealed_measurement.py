"""Revealed pairwise preference measurement.

Usage: python -m src.experiments.run_revealed_measurement <config.yaml>
"""

from __future__ import annotations

from functools import partial
from itertools import combinations

from src.preferences.measurement import measure_revealed_with_template
from src.preferences.ranking import PairwiseData, fit_thurstonian, save_thurstonian, compute_pair_agreement
from src.preferences.storage import MeasurementCache
from src.experiments.experiment_utils import (
    parse_config_path,
    setup_experiment,
    compute_thurstonian_max_iter,
    build_fit_kwargs,
    thurstonian_path_exists,
    flip_pairs,
)


def main():
    ctx = setup_experiment(parse_config_path("Revealed pairwise measurement"), expected_mode="revealed")
    config = ctx.config

    unique_pairs = list(combinations(ctx.tasks, 2))
    max_iter = compute_thurstonian_max_iter(config)
    orders = ["canonical", "reversed"] if config.include_reverse_order else ["canonical"]

    print(f"Tasks: {len(ctx.tasks)}, Pairs: {len(unique_pairs)} x {config.samples_per_pair}")

    for template in ctx.templates:
        for response_format in config.response_formats:
            for order in orders:
                cache = MeasurementCache(template, ctx.client, response_format, order)
                pairs = unique_pairs if order == "canonical" else flip_pairs(unique_pairs)

                measure_fn = partial(
                    measure_revealed_with_template, template, ctx.client,
                    temperature=config.temperature,
                    max_concurrent=ctx.max_concurrent,
                    response_format_name=response_format,
                )
                batch, cache_hits, api_queries = cache.get_or_measure(
                    pairs * config.samples_per_pair, measure_fn, ctx.task_lookup
                )

                print(f"\n{template.name}/{response_format}/{order}: {len(batch.successes)} ({cache_hits} cached)")
                print(f"  Agreement: {compute_pair_agreement(batch.successes):.3f}")

                current_config = {
                    "n_tasks": config.n_tasks,
                    "task_origins": config.task_origins,
                    "samples_per_pair": config.samples_per_pair,
                    "temperature": config.temperature,
                }
                base_path, exists = thurstonian_path_exists(cache.cache_dir, "exhaustive_pairwise", current_config)
                if exists:
                    print("  Thurstonian already done, skipping")
                    continue

                thurstonian = fit_thurstonian(
                    PairwiseData.from_comparisons(batch.successes, ctx.tasks),
                    **build_fit_kwargs(config, max_iter),
                )
                print(f"  Converged: {thurstonian.converged}, μ: [{thurstonian.mu.min():.2f}, {thurstonian.mu.max():.2f}]")
                save_thurstonian(thurstonian, base_path.with_suffix(".yaml"), "exhaustive_pairwise", current_config)

    print("\nDone.")


if __name__ == "__main__":
    main()
