"""Usage: python -m src.experiments.run_binary_measurement <config.yaml>"""

from __future__ import annotations

import sys
from functools import partial
from itertools import combinations
from pathlib import Path

from src.models import get_client, get_default_max_concurrent
from src.task_data import load_tasks
from src.preferences.templates import load_templates_from_yaml
from src.preferences.measurement import measure_with_template
from src.preferences.ranking import PairwiseData, fit_thurstonian, save_thurstonian, compute_pair_agreement, _config_hash
from src.preferences.storage import MeasurementCache
from src.experiments.config import load_experiment_config


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.experiments.run_binary_measurement <config.yaml>")
        sys.exit(1)

    config = load_experiment_config(Path(sys.argv[1]))

    if config.preference_mode != "binary":
        raise ValueError(f"Expected preference_mode='binary', got '{config.preference_mode}'")

    templates = load_templates_from_yaml(config.templates)
    tasks = load_tasks(n=config.n_tasks, origins=config.get_origin_datasets())
    task_lookup = {t.id: t for t in tasks}
    unique_pairs = list(combinations(tasks, 2))
    client = get_client(model_name=config.model)
    max_concurrent = config.max_concurrent or get_default_max_concurrent()

    n_params = (config.n_tasks - 1) + config.n_tasks
    max_iter = config.fitting.max_iter if config.fitting.max_iter else max(2000, n_params * 50)

    orders = ["canonical"]
    if config.include_reverse_order:
        orders.append("reversed")

    n_variants = len(templates) * len(config.response_formats) * len(orders)
    print(f"Templates: {len(templates)}, Tasks: {len(tasks)}, Pairs: {len(unique_pairs)} x {config.samples_per_pair}")
    print(f"Response formats: {config.response_formats}, Orders: {orders} ({n_variants} total variants)")
    print(f"Thurstonian max_iter: {max_iter}")

    for template in templates:
        for response_format in config.response_formats:
            for order in orders:
                cache = MeasurementCache(template, client, response_format, order)

                if order == "canonical":
                    pairs = unique_pairs
                else:
                    pairs = [(b, a) for a, b in unique_pairs]

                pairs_to_query = pairs * config.samples_per_pair
                measure_fn = partial(
                    measure_with_template,
                    template,
                    client,
                    temperature=config.temperature,
                    max_concurrent=max_concurrent,
                    response_format_name=response_format,
                )
                batch, cache_hits, api_queries = cache.get_or_measure(
                    pairs_to_query, measure_fn, task_lookup
                )

                run_label = f"{template.name}/{response_format}/{order}"
                print(f"\n{run_label}: {cache_hits} cached, {api_queries} queried, {len(batch.failures)} failures")

                measurements = batch.successes
                print(f"  Total measurements for these tasks: {len(measurements)}")

                agreement = compute_pair_agreement(measurements)
                print(f"  Pair agreement: {agreement:.3f}")

                # Prepare config and compute hash
                current_config = {
                    "n_tasks": config.n_tasks,
                    "task_origins": config.task_origins,
                    "samples_per_pair": config.samples_per_pair,
                    "temperature": config.temperature,
                }
                config_hash = _config_hash(current_config)

                # Check if fitting already done with this config (hash-based filename)
                base_path = cache.cache_dir / "thurstonian_exhaustive_pairwise"
                thurstonian_path = cache.cache_dir / f"thurstonian_exhaustive_pairwise_{config_hash}.yaml"

                if thurstonian_path.exists():
                    print(f"  Fitting already done with this config (hash: {config_hash}), skipping")
                else:
                    fit_kwargs = {"max_iter": max_iter}
                    if config.fitting.gradient_tol is not None:
                        fit_kwargs["gradient_tol"] = config.fitting.gradient_tol
                    if config.fitting.loss_tol is not None:
                        fit_kwargs["loss_tol"] = config.fitting.loss_tol

                    thurstonian = fit_thurstonian(
                        PairwiseData.from_comparisons(measurements, tasks),
                        **fit_kwargs,
                    )
                    print(f"  Thurstonian converged: {thurstonian.converged}")
                    if not thurstonian.converged:
                        print(f"    Iterations: {thurstonian.n_iterations}/{max_iter}")
                        print(f"    Message: {thurstonian.termination_message}")
                        print(f"    NLL: {thurstonian.neg_log_likelihood:.2f}")
                    print(f"    μ range: [{thurstonian.mu.min():.2f}, {thurstonian.mu.max():.2f}]")
                    print(f"    σ range: [{thurstonian.sigma.min():.2f}, {thurstonian.sigma.max():.2f}]")

                    save_thurstonian(
                        thurstonian,
                        base_path.with_suffix(".yaml"),
                        fitting_method="exhaustive_pairwise",
                        config=current_config,
                    )
                    print(f"  Saved to: {thurstonian_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
