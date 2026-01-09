"""Active learning for efficient Thurstonian fitting.

Usage: python -m src.experiments.run_active_learning <config.yaml>

Implements iterative pair selection to achieve accurate utility estimates
with fewer queries than exhaustive pairwise comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.models import get_client, get_default_max_concurrent
from src.task_data import Task, load_tasks
from src.preferences.templates import load_templates_from_yaml
from src.preferences.measurement import measure_with_template
from src.preferences.ranking import compute_pair_agreement, save_thurstonian
from src.preferences.ranking.active_learning import (
    ActiveLearningState,
    generate_d_regular_pairs,
    select_next_pairs,
    check_convergence,
)
from src.preferences.storage import MeasurementCache, save_yaml
from src.preferences.storage.cache import reconstruct_measurements
from src.experiments.config import load_experiment_config
from src.types import MeasurementBatch


def measure_with_cache(
    template,
    client,
    pairs,
    temperature,
    max_concurrent,
    cache: MeasurementCache,
    task_lookup: dict[str, Task],
) -> tuple[MeasurementBatch, int, int]:
    """
    Wrap measure_with_template to reuse cached comparisons when available.

    The active learning loop remains unaware of caching; it receives a
    MeasurementBatch combining cache hits with fresh API results.
    """
    if not pairs:
        return MeasurementBatch(successes=[], failures=[]), 0, 0

    cached_raw = cache.get_measurements()
    cached_by_pair: dict[tuple[str, str], list[dict[str, str]]] = {}
    for m in cached_raw:
        cached_by_pair.setdefault((m["task_a"], m["task_b"]), []).append(m)

    cached_hits_raw: list[dict[str, str]] = []
    to_query: list[tuple[Task, Task]] = []

    # Consume cached measurements before calling the API; supports multiple
    # stored measurements for the same ordered pair.
    for a, b in pairs:
        key = (a.id, b.id)
        if key in cached_by_pair and cached_by_pair[key]:
            cached_hits_raw.append(cached_by_pair[key].pop())
        else:
            to_query.append((a, b))

    cached_hits = reconstruct_measurements(cached_hits_raw, task_lookup)

    if to_query:
        fresh_batch = measure_with_template(
            template, client, to_query, temperature, max_concurrent
        )
        cache.append(fresh_batch.successes)
    else:
        fresh_batch = MeasurementBatch(successes=[], failures=[])

    combined_successes = cached_hits + fresh_batch.successes
    return MeasurementBatch(successes=combined_successes, failures=fresh_batch.failures), len(cached_hits), len(to_query)


def run_active_learning(config_path: Path) -> None:
    config = load_experiment_config(config_path)

    if config.preference_mode != "active_learning":
        raise ValueError(f"Expected preference_mode='active_learning', got '{config.preference_mode}'")

    al_config = config.active_learning
    rng = np.random.default_rng(al_config.seed)

    templates = load_templates_from_yaml(config.templates)
    tasks = load_tasks(
        n=config.n_tasks,
        origins=config.get_origin_datasets(),
        seed=al_config.seed,
    )
    task_lookup = {t.id: t for t in tasks}
    client = get_client(model_name=config.model)
    max_concurrent = config.max_concurrent or get_default_max_concurrent()

    n_params = (config.n_tasks - 1) + config.n_tasks
    max_iter = config.fitting.max_iter if config.fitting.max_iter else max(2000, n_params * 50)

    n_total_pairs = config.n_tasks * (config.n_tasks - 1) // 2

    print(f"Active Learning Configuration:")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Total possible pairs: {n_total_pairs}")
    print(f"  Initial degree: {al_config.initial_degree}")
    print(f"  Batch size: {al_config.batch_size}")
    print(f"  Max iterations: {al_config.max_iterations}")
    print(f"  Samples per pair: {config.samples_per_pair}")
    print(f"  Convergence threshold: {al_config.convergence_threshold}")
    print(f"  Thurstonian max_iter: {max_iter}")

    for template in templates:
        print(f"\n{'='*60}")
        print(f"Template: {template.name}")
        print(f"{'='*60}")

        cache = MeasurementCache(template, client)
        state = ActiveLearningState(tasks=tasks)
        iteration_history = []

        # Generate initial pairs (d-regular graph)
        initial_pairs = generate_d_regular_pairs(tasks, al_config.initial_degree, rng)
        print(f"\nInitial d-regular graph: {len(initial_pairs)} pairs")

        pairs_to_query = initial_pairs

        for iteration in range(al_config.max_iterations):
            if not pairs_to_query:
                print(f"\nNo more pairs to query. Stopping.")
                break

            # Replicate pairs for multiple samples
            replicated_pairs = pairs_to_query * config.samples_per_pair

            print(f"\n--- Iteration {iteration + 1} ---")
            print(f"  Querying {len(pairs_to_query)} unique pairs ({len(replicated_pairs)} total comparisons)")

            # Run measurements
            batch, cache_hits, api_queries = measure_with_cache(
                template=template,
                client=client,
                pairs=replicated_pairs,
                temperature=config.temperature,
                max_concurrent=max_concurrent,
                cache=cache,
                task_lookup=task_lookup,
            )
            print(f"  Got {len(batch.successes)} measurements ({len(batch.failures)} failures)")
            print(f"  Cache hits: {cache_hits}, API queries: {api_queries}")

            # Update state
            state.add_comparisons(batch.successes)
            state.iteration = iteration + 1

            # Fit model
            fit_kwargs = {"max_iter": max_iter}
            if config.fitting.gradient_tol is not None:
                fit_kwargs["gradient_tol"] = config.fitting.gradient_tol
            if config.fitting.loss_tol is not None:
                fit_kwargs["loss_tol"] = config.fitting.loss_tol

            state.fit(**fit_kwargs)
            print(f"  Thurstonian converged: {state.current_fit.converged}")
            print(f"    μ range: [{state.current_fit.mu.min():.2f}, {state.current_fit.mu.max():.2f}]")
            print(f"    σ range: [{state.current_fit.sigma.min():.2f}, {state.current_fit.sigma.max():.2f}]")

            # Check convergence
            converged, correlation = check_convergence(state, al_config.convergence_threshold)
            print(f"  Rank correlation with previous: {correlation:.4f}")

            # Record iteration
            iteration_history.append({
                "iteration": iteration + 1,
                "pairs_queried": len(pairs_to_query),
                "total_comparisons": len(state.comparisons),
                "unique_pairs_sampled": len(state.sampled_pairs),
                "rank_correlation": float(correlation),
                "thurstonian_converged": bool(state.current_fit.converged),
                "mu_min": float(state.current_fit.mu.min()),
                "mu_max": float(state.current_fit.mu.max()),
            })

            if converged:
                print(f"\n*** Converged at iteration {iteration + 1} ***")
                break

            # Select next pairs
            pairs_to_query = select_next_pairs(
                state,
                batch_size=al_config.batch_size,
                p_threshold=al_config.p_threshold,
                q_threshold=al_config.q_threshold,
                rng=rng,
            )

            print(f"  Selected {len(pairs_to_query)} pairs for next iteration")

        # Final summary
        final_converged, final_correlation = check_convergence(state, al_config.convergence_threshold)
        agreement = compute_pair_agreement(state.comparisons)

        print(f"\n{'='*60}")
        print(f"Final Results for {template.name}")
        print(f"{'='*60}")
        print(f"  Iterations: {state.iteration}")
        print(f"  Unique pairs queried: {len(state.sampled_pairs)} / {n_total_pairs} ({100*len(state.sampled_pairs)/n_total_pairs:.1f}%)")
        print(f"  Total comparisons: {len(state.comparisons)}")
        print(f"  Pair agreement: {agreement:.3f}")
        print(f"  Final rank correlation: {final_correlation:.4f}")
        print(f"  Converged: {final_converged}")
        print(f"  Measurements saved to: {cache.cache_dir}")

        # Save Thurstonian model results
        save_thurstonian(
            state.current_fit,
            cache.cache_dir / "thurstonian_active_learning.yaml",
            fitting_method="active_learning",
            config={
                "config_file": str(config_path),
                "n_tasks": config.n_tasks,
                "seed": al_config.seed,
            },
        )
        print(f"  Thurstonian results saved to: {cache.cache_dir / 'thurstonian_active_learning.yaml'}")

        # Save active learning results (lightweight)
        al_results = {
            "config_file": str(config_path),
            "n_tasks": config.n_tasks,
            "seed": al_config.seed,
            "converged": bool(final_converged),
            "n_iterations": state.iteration,
            "unique_pairs_queried": len(state.sampled_pairs),
            "total_comparisons": len(state.comparisons),
            "pair_agreement": float(agreement),
            "rank_correlations": [h["rank_correlation"] for h in iteration_history],
        }
        al_results_path = cache.cache_dir / "active_learning.yaml"
        save_yaml(al_results, al_results_path)
        print(f"  Active learning results saved to: {al_results_path}")

    print("\nDone.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.experiments.run_active_learning <config.yaml>")
        sys.exit(1)

    run_active_learning(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
