"""Rating-based preference measurement.

Usage: python -m src.experiments.run_rating_measurement <config.yaml>
"""

from __future__ import annotations

from src.preferences.templates import PreTaskRatingPromptBuilder
from src.preferences.measurement import measure_ratings, TaskScoreMeasurer, RegexRatingFormat
from src.preferences.storage import save_ratings, ratings_exist
from src.experiments.sensitivity_experiments.rating_correlation import compute_mean_std_across_tasks
from src.experiments.experiment_utils import parse_config_path, setup_experiment


def main():
    config_path = parse_config_path("Run rating-based preference measurement")
    ctx = setup_experiment(config_path, expected_mode="rating")
    config = ctx.config

    task_list = ctx.tasks * config.samples_per_task
    print(f"Templates: {len(ctx.templates)}, Tasks: {len(ctx.tasks)} x {config.samples_per_task}")

    for template in ctx.templates:
        if ratings_exist(template, ctx.client):
            print(f"Skipping {template.name} (already measured)")
            continue

        print(f"\nMeasuring {template.name}...")

        response_format = RegexRatingFormat(scale_min=config.scale_min, scale_max=config.scale_max)
        builder = PreTaskRatingPromptBuilder(
            measurer=TaskScoreMeasurer(),
            response_format=response_format,
            template=template,
        )
        batch = measure_ratings(
            client=ctx.client,
            tasks=task_list,
            builder=builder,
            temperature=config.temperature,
            max_concurrent=ctx.max_concurrent,
        )

        mean_std = compute_mean_std_across_tasks(batch.successes)
        print(f"  {len(batch.successes)} scores, mean std: {mean_std:.3f}")

        run_path = save_ratings(template=template, client=ctx.client, scores=batch.successes)
        print(f"  Saved to {run_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
