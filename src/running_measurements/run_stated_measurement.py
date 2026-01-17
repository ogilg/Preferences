"""Stated preference measurement.

Usage: python -m src.experiments.run_stated_measurement <config.yaml>
"""

from __future__ import annotations

from typing import Literal

from src.prompt_templates import PreTaskStatedPromptBuilder, PromptTemplate
from src.preference_measurement import (
    measure_stated,
    StatedScoreMeasurer,
    RATING_FORMATS,
    QUALITATIVE_FORMATS,
)
from src.measurement_storage import save_stated, stated_exist
from src.measurement_storage.base import build_measurement_config
from src.prompt_templates.sampler import (
    SampledConfiguration,
    sample_configurations_lhs,
    print_sampling_balance,
)
from src.analysis.sensitivity.stated_correlation import compute_mean_std_across_tasks
from src.running_measurements.utils.experiment_utils import parse_config_path, setup_experiment


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
        lhs_seed = config.lhs_seed if config.lhs_seed is not None else 42
        configurations = sample_configurations_lhs(
            ctx.templates, config.response_formats, config.generation_seeds,
            n_samples=config.n_template_samples, seed=lhs_seed,
        )
        print(f"LHS sampling: {config.n_template_samples} configurations")
        print_sampling_balance(configurations)
    else:
        configurations = [
            SampledConfiguration(t, rf, s)
            for t in ctx.templates for rf in config.response_formats for s in config.generation_seeds
        ]

    print(f"Templates: {len(ctx.templates)}, Configs: {len(configurations)}, Tasks: {len(ctx.tasks)} x {config.n_samples}")

    for i, cfg in enumerate(configurations):
        print(f"[PROGRESS {i}/{len(configurations)}]", flush=True)
        if stated_exist(cfg.template, ctx.client, cfg.response_format, cfg.seed):
            print(f"Skipping {cfg.template.name} (format={cfg.response_format}, seed={cfg.seed}) (already measured)")
            continue

        print(f"\nMeasuring {cfg.template.name} (format={cfg.response_format}, seed={cfg.seed})...")

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

        if batch.failures:
            print(f"  {len(batch.failures)} failures. Sample failures:")
            for prompt, error in batch.failures[:5]:
                error_preview = error[:200] if len(error) > 200 else error
                print(f"    - {error_preview}")

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
            response_format=cfg.response_format,
            seed=cfg.seed,
            config=config_dict,
        )
        print(f"  Saved to {run_path}")

    print(f"[PROGRESS {len(configurations)}/{len(configurations)}]", flush=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
