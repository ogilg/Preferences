"""Run template sensitivity experiment.

Measures binary preferences across multiple templates,
fits Thurstonian utilities, and computes correlations between templates.

Usage:
    python -m src.sensitivity_experiments.run_binary --templates path/to/templates.yaml --n-tasks 10
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

from src.models import HyperbolicModel
from src.task_data import load_tasks, OriginDataset
from src.preferences.templates import load_templates_from_yaml
from src.preferences.measure_preferences import measure_with_template
from src.preferences.ranking import PairwiseData, fit_thurstonian
from src.preferences.storage import save_run, run_exists
from src.sensitivity_experiments.binary_correlation import (
    compute_pairwise_correlations,
    save_correlations,
    save_experiment_config,
)


def main():
    parser = argparse.ArgumentParser(description="Run template sensitivity experiment")
    parser.add_argument("--templates", type=Path,
                        default=Path("src/preferences/template_data/binary_choice_variants.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/sensitivity_experiments"))
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-concurrent", type=int, default=40)
    parser.add_argument("--measure-order", action="store_true",
                        help="Measure order sensitivity by comparing normal vs reversed pair ordering")
    parser.add_argument("--samples-per-pair", type=int, default=5,
                        help="Number of times to sample each pair comparison")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Max iterations for Thurstonian fitting (default: auto)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    templates = load_templates_from_yaml(args.templates)
    tasks = load_tasks(n=args.n_tasks, origin=OriginDataset.WILDCHAT)
    unique_pairs = list(combinations(tasks, 2))
    pairs = unique_pairs * args.samples_per_pair
    model = HyperbolicModel(model_name=args.model)

    # Auto-set max_iter based on problem size if not specified
    # Rule of thumb: 50 iterations per parameter, minimum 2000
    n_params = (args.n_tasks - 1) + args.n_tasks  # (n-1) mu values + n sigma values
    max_iter = args.max_iter if args.max_iter else max(2000, n_params * 50)

    print(f"Templates: {len(templates)}, Tasks: {len(tasks)}, Pairs: {len(unique_pairs)} x {args.samples_per_pair} = {len(pairs)}")
    print(f"Thurstonian max_iter: {max_iter}")

    # Generate reversed pairs if measuring order sensitivity
    reversed_pairs = [(b, a) for (a, b) in pairs] if args.measure_order else None

    # Measure each template
    results = {}
    skipped = 0
    for template in templates:
        if run_exists(template, model, args.n_tasks):
            print(f"Skipping {template.name} (already measured)")
            skipped += 1
            continue

        print(f"\nMeasuring template {template.name}...")

        measurements = measure_with_template(template, model, pairs, args.temperature, args.max_concurrent)
        print(f"  Got {len(measurements)} measurements")

        thurstonian = fit_thurstonian(PairwiseData.from_comparisons(measurements, tasks), max_iter=max_iter)
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
            measurements=measurements,
            thurstonian=thurstonian,
        )
        print(f"  Saved to {run_path}")

        if args.measure_order:
            # Store normal ordering with suffix
            results[f"{template.name}_normal"] = (measurements, thurstonian)

            # Measure reversed ordering
            print(f"  Measuring reversed ordering...")
            rev_measurements = measure_with_template(template, model, reversed_pairs, args.temperature, args.max_concurrent)
            print(f"  Got {len(rev_measurements)} reversed measurements")

            rev_thurstonian = fit_thurstonian(PairwiseData.from_comparisons(rev_measurements, tasks), max_iter=max_iter)
            print(f"  Reversed Thurstonian converged: {rev_thurstonian.converged}")
            if not rev_thurstonian.converged:
                print(f"    Iterations: {rev_thurstonian.n_iterations}/{max_iter}")
                print(f"    Message: {rev_thurstonian.termination_message}")
                print(f"    NLL: {rev_thurstonian.neg_log_likelihood:.2f}")
                print(f"    μ range: [{rev_thurstonian.mu.min():.2f}, {rev_thurstonian.mu.max():.2f}]")
                print(f"    σ range: [{rev_thurstonian.sigma.min():.2f}, {rev_thurstonian.sigma.max():.2f}]")

            results[f"{template.name}_reversed"] = (rev_measurements, rev_thurstonian)
        else:
            results[template.name] = (measurements, thurstonian)

    print(f"\nMeasured: {len(results)}, Skipped: {skipped}")

    if not results:
        print("No new measurements - skipping correlation analysis")
        return

    # Save experiment config (templates used)
    if args.measure_order:
        measured_templates = [t for t in templates if f"{t.name}_normal" in results]
    else:
        measured_templates = [t for t in templates if t.name in results]
    save_experiment_config(
        templates=measured_templates,
        model_name=args.model,
        temperature=args.temperature,
        n_tasks=args.n_tasks,
        path=args.output_dir / "config.yaml",
    )

    # Compute and save correlations (only for newly measured templates)
    correlations = compute_pairwise_correlations(results, tasks)
    save_correlations(correlations, args.output_dir / "correlations.yaml")

    # Print summary
    print("\n=== Correlations ===")
    for c in correlations:
        print(f"{c['template_a']} vs {c['template_b']}: "
              f"WR={c['win_rate_correlation']:.3f}, UT={c['utility_correlation']:.3f}")


if __name__ == "__main__":
    main()
