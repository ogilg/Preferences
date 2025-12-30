"""Run template sensitivity experiment.

Measures binary preferences across multiple templates,
fits Thurstonian utilities, and computes correlations between templates.

Usage:
    python -m src.sensitivity_experiments.run --templates path/to/templates.yaml --n-tasks 10
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
from src.sensitivity_experiments.correlation import (
    compute_pairwise_correlations,
    save_correlations,
    save_experiment_config,
)


def main():
    parser = argparse.ArgumentParser(description="Run template sensitivity experiment")
    parser.add_argument("--templates", type=Path,
                        default=Path("src/preferences/template_data/binary_choice_variants.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/phrasing_sensitivity"))
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-concurrent", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    templates = load_templates_from_yaml(args.templates)
    tasks = load_tasks(n=args.n_tasks, origin=OriginDataset.WILDCHAT)
    pairs = list(combinations(tasks, 2))
    model = HyperbolicModel(model_name=args.model)

    print(f"Templates: {len(templates)}, Tasks: {len(tasks)}, Pairs: {len(pairs)}")

    # Measure each template
    results = {}
    skipped = 0
    for template in templates:
        if run_exists(template, model):
            print(f"Skipping {template.name} (already measured)")
            skipped += 1
            continue

        print(f"\nMeasuring template {template.name}...")

        measurements = measure_with_template(template, model, pairs, args.temperature, args.max_concurrent)
        print(f"  Got {len(measurements)} measurements")

        thurstonian = fit_thurstonian(PairwiseData.from_comparisons(measurements, tasks))
        print(f"  Thurstonian converged: {thurstonian.converged}")

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

        results[template.name] = (measurements, thurstonian)

    print(f"\nMeasured: {len(results)}, Skipped: {skipped}")

    if not results:
        print("No new measurements - skipping correlation analysis")
        return

    # Save experiment config (templates used)
    save_experiment_config(
        templates=[t for t in templates if t.name in results],
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
