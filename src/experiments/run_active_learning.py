"""Active learning for efficient Thurstonian fitting.

Usage: python -m src.experiments.run_active_learning <config.yaml>

Implements iterative pair selection to achieve accurate utility estimates
with fewer queries than exhaustive pairwise comparison.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np

from src.models import get_client, get_default_max_concurrent
from src.task_data import Task, load_tasks
from src.preferences.templates import load_templates_from_yaml
from src.preferences.measurement import measure_with_template
from src.preferences.ranking import compute_pair_agreement, save_thurstonian, _config_hash
from src.preferences.ranking.active_learning import (
    ActiveLearningState,
    generate_d_regular_pairs,
    select_next_pairs,
    check_convergence,
)
from src.preferences.storage import MeasurementCache, save_yaml
from src.experiments.config import load_experiment_config


def flip_pairs(pairs: list[tuple[Task, Task]]) -> list[tuple[Task, Task]]:
    return [(b, a) for a, b in pairs]


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

    orders = ["canonical"]
    if config.include_reverse_order:
        orders.append("reversed")

    n_variants = len(templates) * len(config.response_formats) * len(orders)

    print(f"Active Learning Configuration:")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Total possible pairs: {n_total_pairs}")
    print(f"  Initial degree: {al_config.initial_degree}")
    print(f"  Batch size: {al_config.batch_size}")
    print(f"  Max iterations: {al_config.max_iterations}")
    print(f"  Samples per pair: {config.samples_per_pair}")
    print(f"  Convergence threshold: {al_config.convergence_threshold}")
    print(f"  Thurstonian max_iter: {max_iter}")
    print(f"  Response formats: {config.response_formats}, Orders: {orders} ({n_variants} total variants)")

    for template in templates:
        for response_format in config.response_formats:
            for order in orders:
                run_label = f"{template.name}/{response_format}/{order}"
                print(f"\n{'='*60}")
                print(f"Run: {run_label}")
                print(f"{'='*60}")

                cache = MeasurementCache(template, client, response_format, order)

                # Prepare config and compute hash
                current_config = {
                    "n_tasks": config.n_tasks,
                    "seed": al_config.seed,
                }
                config_hash = _config_hash(current_config)

                # Check if already done with this config (hash-based filename)
                base_path = cache.cache_dir / "thurstonian_active_learning"
                thurstonian_path = cache.cache_dir / f"thurstonian_active_learning_{config_hash}.yaml"

                if thurstonian_path.exists():
                    print(f"Active learning already done with this config (hash: {config_hash}), skipping")
                    continue

                state = ActiveLearningState(tasks=tasks)
                iteration_history = []

                # Generate initial pairs (d-regular graph)
                initial_pairs = generate_d_regular_pairs(tasks, al_config.initial_degree, rng)
                if order == "reversed":
                    initial_pairs = flip_pairs(initial_pairs)
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
                    measure_fn = partial(
                        measure_with_template,
                        template,
                        client,
                        temperature=config.temperature,
                        max_concurrent=max_concurrent,
                        response_format_name=response_format,
                    )
                    batch, cache_hits, api_queries = cache.get_or_measure(
                        replicated_pairs, measure_fn, task_lookup
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
                    if order == "reversed":
                        pairs_to_query = flip_pairs(pairs_to_query)

                    print(f"  Selected {len(pairs_to_query)} pairs for next iteration")

                # Final summary
                final_converged, final_correlation = check_convergence(state, al_config.convergence_threshold)
                agreement = compute_pair_agreement(state.comparisons)

                print(f"\n{'='*60}")
                print(f"Final Results for {run_label}")
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
                    base_path.with_suffix(".yaml"),
                    fitting_method="active_learning",
                    config=current_config,
                )
                print(f"  Thurstonian results saved to: {thurstonian_path.name}")

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
