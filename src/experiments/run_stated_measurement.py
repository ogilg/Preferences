"""Stated preference measurement.

Usage: python -m src.experiments.run_stated_measurement <config.yaml>
"""

from __future__ import annotations

from typing import Literal

from src.preferences.templates import PreTaskStatedPromptBuilder, PromptTemplate
from src.preferences.measurement import (
    measure_stated,
    StatedScoreMeasurer,
    RATING_FORMATS,
    QUALITATIVE_FORMATS,
)
from src.preferences.storage import save_stated, stated_exist
from src.preferences.storage.base import build_measurement_config
from src.preferences.templates.sampler import (
    SampledConfiguration,
    sample_configurations_lhs,
    print_sampling_balance,
)
from src.experiments.sensitivity_experiments.stated_correlation import compute_mean_std_across_tasks
from src.experiments.experiment_utils import parse_config_path, setup_experiment


def parse_scale_from_template(template: PromptTemplate) -> tuple[int, int] | Literal["qualitative"]:
    scale_str = template.tags_dict["scale"]
    if scale_str == "qualitative":
        return "qualitative"
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

        scale_info = parse_scale_from_template(cfg.template)

        if scale_info == "qualitative":
            response_format = QUALITATIVE_FORMATS[cfg.response_format]()
        else:
            scale_min, scale_max = scale_info
            response_format = RATING_FORMATS[cfg.response_format](scale_min, scale_max)

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

        config_dict = build_measurement_config(
            template=cfg.template,
            client=ctx.client,
            response_format=cfg.response_format,
            seed=cfg.seed,
            temperature=config.temperature,
        )

        run_path = save_stated(
            template=cfg.template,
            client=ctx.client,
            scores=batch.successes,
            config=config_dict,
        )
        print(f"  Saved to {run_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
