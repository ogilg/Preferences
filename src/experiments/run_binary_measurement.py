"""Usage: python -m src.experiments.run_binary_measurement <config.yaml>"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

from src.models import get_client, get_default_max_concurrent
from src.task_data import load_tasks
from src.preferences.templates import load_templates_from_yaml
from src.preferences.measurement import measure_with_template
from src.preferences.ranking import PairwiseData, fit_thurstonian, save_thurstonian, compute_pair_agreement
from src.preferences.storage import MeasurementCache, reconstruct_measurements
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
    task_ids = set(task_lookup.keys())
    unique_pairs = list(combinations(tasks, 2))
    client = get_client(model_name=config.model)
    max_concurrent = config.max_concurrent or get_default_max_concurrent()

    n_params = (config.n_tasks - 1) + config.n_tasks
    max_iter = config.fitting.max_iter if config.fitting.max_iter else max(2000, n_params * 50)

    print(f"Templates: {len(templates)}, Tasks: {len(tasks)}, Pairs: {len(unique_pairs)} x {config.samples_per_pair}")
    print(f"Thurstonian max_iter: {max_iter}")

    for template in templates:
        cache = MeasurementCache(template, client)
        existing_pairs = cache.get_existing_pairs()

        missing_pairs = [(a, b) for a, b in unique_pairs if (a.id, b.id) not in existing_pairs]

        thurstonian_path = cache.cache_dir / "thurstonian.yaml"
        if not missing_pairs and thurstonian_path.exists():
            print(f"\n{template.name}: all {len(unique_pairs)} pairs cached, skipping")
            continue

        print(f"\n{template.name}: {len(existing_pairs)} cached, {len(missing_pairs)} to query")

        pairs_to_query = missing_pairs * config.samples_per_pair
        batch = measure_with_template(template, client, pairs_to_query, config.temperature, max_concurrent)
        print(f"  Got {len(batch.successes)} measurements ({len(batch.failures)} failures)")

        cache.append(batch.successes)

        raw_measurements = cache.get_measurements(task_ids=task_ids)
        measurements = reconstruct_measurements(raw_measurements, task_lookup)
        print(f"  Total measurements for these tasks: {len(measurements)}")

        agreement = compute_pair_agreement(measurements)
        print(f"  Pair agreement: {agreement:.3f}")

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
            cache.cache_dir / "thurstonian.yaml",
            config={
                "config_file": str(sys.argv[1]),
                "n_tasks": config.n_tasks,
                "task_origins": config.task_origins,
                "samples_per_pair": config.samples_per_pair,
                "temperature": config.temperature,
                "fitting": {
                    "max_iter": max_iter,
                    "gradient_tol": config.fitting.gradient_tol,
                    "loss_tol": config.fitting.loss_tol,
                },
            },
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
