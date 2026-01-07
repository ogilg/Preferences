"""Usage: python -m src.experiments.run_binary_measurement <config.yaml>"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

from src.models import HyperbolicModel
from src.task_data import load_tasks
from src.preferences.templates import load_templates_from_yaml
from src.preferences.measurement import measure_with_template
from src.preferences.ranking import PairwiseData, fit_thurstonian, compute_pair_agreement
from src.preferences.storage import save_run, run_exists, RESULTS_DIR
from src.experiments.config import load_experiment_config


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.experiments.run_binary_measurement <config.yaml>")
        sys.exit(1)

    config = load_experiment_config(Path(sys.argv[1]))

    if config.preference_mode != "binary":
        raise ValueError(f"Expected preference_mode='binary', got '{config.preference_mode}'")

    templates = load_templates_from_yaml(config.templates)
    tasks = load_tasks(n=config.n_tasks, origin=config.get_origin_dataset())
    unique_pairs = list(combinations(tasks, 2))
    pairs = unique_pairs * config.samples_per_pair
    model = HyperbolicModel(model_name=config.model)

    n_params = (config.n_tasks - 1) + config.n_tasks
    max_iter = config.fitting.max_iter if config.fitting.max_iter else max(2000, n_params * 50)

    print(f"Templates: {len(templates)}, Tasks: {len(tasks)}, Pairs: {len(unique_pairs)} x {config.samples_per_pair} = {len(pairs)}")
    print(f"Thurstonian max_iter: {max_iter}")

    measured = 0
    skipped = 0
    for template in templates:
        if run_exists(template, model, config.n_tasks, RESULTS_DIR):
            print(f"Skipping {template.name} (already measured)")
            skipped += 1
            continue

        print(f"\nMeasuring template {template.name}...")

        batch = measure_with_template(template, model, pairs, config.temperature, config.max_concurrent)
        print(f"  Got {len(batch.successes)} measurements ({len(batch.failures)} failures)")

        agreement = compute_pair_agreement(batch.successes)
        print(f"  Pair agreement rate: {agreement:.3f}")

        fit_kwargs = {"max_iter": max_iter}
        if config.fitting.gradient_tol is not None:
            fit_kwargs["gradient_tol"] = config.fitting.gradient_tol
        if config.fitting.loss_tol is not None:
            fit_kwargs["loss_tol"] = config.fitting.loss_tol

        thurstonian = fit_thurstonian(
            PairwiseData.from_comparisons(batch.successes, tasks),
            **fit_kwargs,
        )
        print(f"  Thurstonian converged: {thurstonian.converged}")
        if not thurstonian.converged:
            print(f"    Iterations: {thurstonian.n_iterations}/{max_iter}")
            print(f"    Message: {thurstonian.termination_message}")
            print(f"    NLL: {thurstonian.neg_log_likelihood:.2f}")
            print(f"    μ range: [{thurstonian.mu.min():.2f}, {thurstonian.mu.max():.2f}]")
            print(f"    σ range: [{thurstonian.sigma.min():.2f}, {thurstonian.sigma.max():.2f}]")

        run_path = save_run(
            template=template,
            template_file=str(config.templates),
            model=model,
            temperature=config.temperature,
            tasks=tasks,
            measurements=batch.successes,
            thurstonian=thurstonian,
            pair_agreement=agreement,
        )
        print(f"  Saved to {run_path}")
        measured += 1

    print(f"\nDone. Measured: {measured}, Skipped: {skipped}")
    print("Run 'python -m src.experiments.sensitivity_experiments.plot results/' to analyze correlations.")


if __name__ == "__main__":
    main()
