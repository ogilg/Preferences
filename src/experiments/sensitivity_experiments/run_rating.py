"""Usage: python -m src.sensitivity_experiments.run_rating --templates <yaml> --n-tasks N"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models import HyperbolicModel
from src.task_data import load_tasks, OriginDataset
from src.preferences.templates import load_templates_from_yaml, PreTaskRatingPromptBuilder
from src.preferences.measurement import measure_ratings, TaskScoreMeasurer, RegexRatingFormat
from src.preferences.storage import save_rating_run, rating_run_exists
from src.experiments.sensitivity_experiments.rating_correlation import (
    compute_rating_pairwise_correlations,
    save_rating_correlations,
    save_experiment_config,
)


def measure_ratings_with_template(
    template,
    model,
    tasks,
    temperature: float,
    max_concurrent: int,
    scale_min: int = 1,
    scale_max: int = 10,
):
    response_format = RegexRatingFormat(scale_min=scale_min, scale_max=scale_max)
    measurer = TaskScoreMeasurer()
    builder = PreTaskRatingPromptBuilder(
        measurer=measurer,
        response_format=response_format,
        template=template,
    )
    batch = measure_ratings(
        model=model,
        tasks=tasks,
        builder=builder,
        temperature=temperature,
        max_concurrent=max_concurrent,
    )
    return batch


def main():
    parser = argparse.ArgumentParser(description="Run rating sensitivity experiment")
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path("src/preferences/templates/data/rating_v1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/rating_sensitivity"),
    )
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-concurrent", type=int, default=40)
    parser.add_argument("--scale-min", type=int, default=1)
    parser.add_argument("--scale-max", type=int, default=10)
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=1,
        help="Number of times to sample each task rating",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    templates = load_templates_from_yaml(args.templates)
    tasks = load_tasks(n=args.n_tasks, origin=OriginDataset.WILDCHAT)
    model = HyperbolicModel(model_name=args.model)

    task_list = tasks * args.samples_per_task

    print(f"Templates: {len(templates)}, Tasks: {len(tasks)} x {args.samples_per_task} = {len(task_list)}")

    results = {}
    skipped = 0
    for template in templates:
        if rating_run_exists(template, model, args.n_tasks):
            print(f"Skipping {template.name} (already measured)")
            skipped += 1
            continue

        print(f"\nMeasuring template {template.name}...")

        batch = measure_ratings_with_template(
            template,
            model,
            task_list,
            args.temperature,
            args.max_concurrent,
            args.scale_min,
            args.scale_max,
        )
        print(f"  Got {len(batch.successes)} scores ({len(batch.failures)} failures)")

        run_path = save_rating_run(
            template=template,
            template_file=str(args.templates),
            model=model,
            temperature=args.temperature,
            tasks=tasks,
            scores=batch.successes,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
        )
        print(f"  Saved to {run_path}")

        results[template.name] = batch.successes

    print(f"\nMeasured: {len(results)}, Skipped: {skipped}")

    if not results:
        print("No new measurements - skipping correlation analysis")
        return

    measured_templates = [t for t in templates if t.name in results]
    save_experiment_config(
        templates=measured_templates,
        model_name=args.model,
        temperature=args.temperature,
        n_tasks=args.n_tasks,
        path=args.output_dir / "config.yaml",
    )

    correlations = compute_rating_pairwise_correlations(results, tasks)
    save_rating_correlations(correlations, args.output_dir / "correlations.yaml")

    print("\n=== Correlations ===")
    for c in correlations:
        print(
            f"{c['template_a']} vs {c['template_b']}: "
            f"Pearson={c['pearson_correlation']:.3f}, "
            f"Spearman={c['spearman_correlation']:.3f}"
        )


if __name__ == "__main__":
    main()
