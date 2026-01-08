"""Active learning for efficient Thurstonian fitting.

Usage: python -m src.experiments.run_active_learning <config.yaml>

Implements iterative pair selection to achieve accurate utility estimates
with fewer queries than exhaustive pairwise comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.models import HyperbolicModel
from src.task_data import load_tasks
from src.preferences.templates import load_templates_from_yaml
from src.preferences.measurement import measure_with_template
from src.preferences.ranking import compute_pair_agreement
from src.preferences.ranking.active_learning import (
    ActiveLearningState,
    generate_d_regular_pairs,
    select_next_pairs,
    check_convergence,
)
from src.preferences.storage import MeasurementCache
from src.experiments.config import load_experiment_config


def run_active_learning(config_path: Path) -> None:
    config = load_experiment_config(config_path)

    if config.preference_mode != "active_learning":
        raise ValueError(f"Expected preference_mode='active_learning', got '{config.preference_mode}'")

    al_config = config.active_learning
    rng = np.random.default_rng(al_config.seed)

    templates = load_templates_from_yaml(config.templates)
    tasks = load_tasks(n=config.n_tasks, origin=config.get_origin_dataset())
    model = HyperbolicModel(model_name=config.model)

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

        cache = MeasurementCache(template, model)
        state = ActiveLearningState(tasks=tasks)

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
            batch = measure_with_template(
                template, model, replicated_pairs, config.temperature, config.max_concurrent
            )
            print(f"  Got {len(batch.successes)} measurements ({len(batch.failures)} failures)")

            # Append to cache
            cache.append(batch.successes)

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

            unsampled_remaining = len(state.get_unsampled_pairs())
            print(f"  Unsampled pairs remaining: {unsampled_remaining}")

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

    print("\nDone.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.experiments.run_active_learning <config.yaml>")
        sys.exit(1)

    run_active_learning(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
