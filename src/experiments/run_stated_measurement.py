"""Stated preference measurement.

Usage: python -m src.experiments.run_stated_measurement <config.yaml>
"""

from __future__ import annotations

from src.preferences.templates import PreTaskStatedPromptBuilder, PromptTemplate
from src.preferences.measurement import measure_stated, StatedScoreMeasurer, RegexRatingFormat
from src.preferences.storage import save_stated, stated_exist
from src.preferences.templates.sampler import (
    SampledConfiguration,
    sample_configurations_lhs,
    print_sampling_balance,
)
from src.experiments.sensitivity_experiments.stated_correlation import compute_mean_std_across_tasks
from src.experiments.experiment_utils import parse_config_path, setup_experiment


def parse_scale_from_template(template: PromptTemplate) -> tuple[int, int]:
    scale_str = template.tags_dict["scale"]
    min_str, max_str = scale_str.split("-")
    return int(min_str), int(max_str)


def main():
    config_path = parse_config_path("Run stated preference measurement")
    ctx = setup_experiment(config_path, expected_mode="stated")
    config = ctx.config

    task_list = ctx.tasks * config.n_samples

    if config.template_sampling == "lhs" and config.n_template_samples:
        configurations = sample_configurations_lhs(
            ctx.templates, config.response_formats, config.generation_seeds,
            n_samples=config.n_template_samples, seed=42,
        )
        print(f"LHS sampling: {config.n_template_samples} configurations")
        print_sampling_balance(configurations)
    else:
        configurations = [
            SampledConfiguration(t, rf, s)
            for t in ctx.templates for rf in config.response_formats for s in config.generation_seeds
        ]

    print(f"Templates: {len(ctx.templates)}, Configs: {len(configurations)}, Tasks: {len(ctx.tasks)} x {config.n_samples}")

    for cfg in configurations:
        if stated_exist(cfg.template, ctx.client):
            print(f"Skipping {cfg.template.name} (already measured)")
            continue

        print(f"\nMeasuring {cfg.template.name} (seed={cfg.seed})...")

        scale_min, scale_max = parse_scale_from_template(cfg.template)
        response_format = RegexRatingFormat(scale_min=scale_min, scale_max=scale_max)
        builder = PreTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=response_format,
            template=cfg.template,
        )
        batch = measure_stated(
            client=ctx.client,
            tasks=task_list,
            builder=builder,
            temperature=config.temperature,
            max_concurrent=ctx.max_concurrent,
            seed=cfg.seed,
        )

        mean_std = compute_mean_std_across_tasks(batch.successes)
        print(f"  {len(batch.successes)} scores, mean std: {mean_std:.3f}")

        run_path = save_stated(template=cfg.template, client=ctx.client, scores=batch.successes)
        print(f"  Saved to {run_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
