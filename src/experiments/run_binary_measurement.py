"""Usage: python -m src.experiments.run_binary_measurement --templates <yaml> --n-tasks N"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

from src.models import HyperbolicModel
from src.task_data import load_tasks, OriginDataset
from src.preferences.templates import load_templates_from_yaml
from src.preferences.measurement import measure_with_template
from src.preferences.ranking import PairwiseData, fit_thurstonian, compute_pair_agreement
from src.preferences.storage import save_run, run_exists


def main():
    parser = argparse.ArgumentParser(description="Run binary preference measurements")
    parser.add_argument("--templates", type=Path,
                        default=Path("src/preferences/templates/data/binary_choice_variants.yaml"))
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-concurrent", type=int, default=40)
    parser.add_argument("--samples-per-pair", type=int, default=5,
                        help="Number of times to sample each pair comparison")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Max iterations for Thurstonian fitting (default: auto)")
    args = parser.parse_args()

    templates = load_templates_from_yaml(args.templates)
    tasks = load_tasks(n=args.n_tasks, origin=OriginDataset.WILDCHAT)
    unique_pairs = list(combinations(tasks, 2))
    pairs = unique_pairs * args.samples_per_pair
    model = HyperbolicModel(model_name=args.model)

    n_params = (args.n_tasks - 1) + args.n_tasks
    max_iter = args.max_iter if args.max_iter else max(2000, n_params * 50)

    print(f"Templates: {len(templates)}, Tasks: {len(tasks)}, Pairs: {len(unique_pairs)} x {args.samples_per_pair} = {len(pairs)}")
    print(f"Thurstonian max_iter: {max_iter}")

    measured = 0
    skipped = 0
    for template in templates:
        if run_exists(template, model, args.n_tasks):
            print(f"Skipping {template.name} (already measured)")
            skipped += 1
            continue

        print(f"\nMeasuring template {template.name}...")

        batch = measure_with_template(template, model, pairs, args.temperature, args.max_concurrent)
        print(f"  Got {len(batch.successes)} measurements ({len(batch.failures)} failures)")

        agreement = compute_pair_agreement(batch.successes)
        print(f"  Pair agreement rate: {agreement:.3f}")

        thurstonian = fit_thurstonian(PairwiseData.from_comparisons(batch.successes, tasks), max_iter=max_iter)
        print(f"  Thurstonian converged: {thurstonian.converged}")
        if not thurstonian.converged:
            print(f"    Iterations: {thurstonian.n_iterations}/{max_iter}")
            print(f"    Message: {thurstonian.termination_message}")
            print(f"    NLL: {thurstonian.neg_log_likelihood:.2f}")
            print(f"    μ range: [{thurstonian.mu.min():.2f}, {thurstonian.mu.max():.2f}]")
            print(f"    σ range: [{thurstonian.sigma.min():.2f}, {thurstonian.sigma.max():.2f}]")

        run_path = save_run(
            template=template,
            template_file=str(args.templates),
            model=model,
            temperature=args.temperature,
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
