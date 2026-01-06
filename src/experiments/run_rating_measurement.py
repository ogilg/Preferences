"""Usage: python -m src.experiments.run_rating_measurement <config.yaml>"""

from __future__ import annotations

import sys
from pathlib import Path

from src.models import HyperbolicModel
from src.task_data import load_tasks
from src.preferences.templates import load_templates_from_yaml, PreTaskRatingPromptBuilder
from src.preferences.measurement import measure_ratings, TaskScoreMeasurer, RegexRatingFormat
from src.preferences.storage import save_rating_run, rating_run_exists
from src.experiments.config import load_experiment_config
from src.experiments.sensitivity_experiments.rating_correlation import compute_mean_std_across_tasks


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
    if len(sys.argv) < 2:
        print("Usage: python -m src.experiments.run_rating_measurement <config.yaml>")
        sys.exit(1)

    config = load_experiment_config(Path(sys.argv[1]))

    if config.preference_mode != "rating":
        raise ValueError(f"Expected preference_mode='rating', got '{config.preference_mode}'")

    templates = load_templates_from_yaml(config.templates)
    tasks = load_tasks(n=config.n_tasks, origin=config.get_origin_dataset())
    model = HyperbolicModel(model_name=config.model)

    task_list = tasks * config.samples_per_task

    print(f"Templates: {len(templates)}, Tasks: {len(tasks)} x {config.samples_per_task} = {len(task_list)}")

    measured = 0
    skipped = 0
    for template in templates:
        if rating_run_exists(template, model, config.n_tasks):
            print(f"Skipping {template.name} (already measured)")
            skipped += 1
            continue

        print(f"\nMeasuring template {template.name}...")

        batch = measure_ratings_with_template(
            template,
            model,
            task_list,
            config.temperature,
            config.max_concurrent,
            config.scale_min,
            config.scale_max,
        )
        print(f"  Got {len(batch.successes)} scores ({len(batch.failures)} failures)")

        mean_std = compute_mean_std_across_tasks(batch.successes)
        print(f"  Mean std across tasks: {mean_std:.3f}")

        run_path = save_rating_run(
            template=template,
            template_file=str(config.templates),
            model=model,
            temperature=config.temperature,
            tasks=tasks,
            scores=batch.successes,
            scale_min=config.scale_min,
            scale_max=config.scale_max,
            mean_rating_std=mean_std,
        )
        print(f"  Saved to {run_path}")
        measured += 1

    print(f"\nDone. Measured: {measured}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
