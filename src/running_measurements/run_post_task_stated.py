"""Post-task stated preference measurement.

Usage: python -m src.experiments.run_post_task_stated <config.yaml>
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from src.prompt_templates import PostTaskStatedPromptBuilder
from src.preference_measurement import (
    measure_post_task_stated,
    StatedScoreMeasurer,
    RATING_FORMATS,
    QUALITATIVE_FORMATS,
)
from src.measurement_storage import CompletionStore, PostStatedCache, model_short_name
from src.prompt_templates.sampler import (
    SampledConfiguration,
    sample_configurations_lhs,
    print_sampling_balance,
)
from src.running_measurements.utils.experiment_utils import (
    parse_config_path,
    setup_experiment,
    parse_scale_from_template,
)


def main():
    config_path = parse_config_path("Post-task stated measurement")
    ctx = setup_experiment(config_path, expected_mode="post_task_stated")
    config = ctx.config

    if ctx.templates is None:
        raise ValueError("Templates required for post_task_stated mode")

    completion_seeds = config.completion_seeds or config.generation_seeds
    model_short = model_short_name(ctx.client.canonical_model_name)

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

    total_runs = len(completion_seeds) * len(configurations)
    print(f"Templates: {len(ctx.templates)}, Configs: {len(configurations)}, "
          f"Completion seeds: {completion_seeds}, Tasks: {len(ctx.tasks)} x {config.n_samples}")

    run_idx = 0
    for completion_seed in completion_seeds:
        store = CompletionStore(client=ctx.client, seed=completion_seed)
        if not store.exists():
            print(f"No completions for seed {completion_seed}, run completion generation first")
            continue

        task_completions = store.load(ctx.task_lookup)
        data = [(tc.task, tc.completion) for tc in task_completions] * config.n_samples

        for cfg in configurations:
            print(f"[PROGRESS {run_idx}/{total_runs}]", flush=True)
            run_idx += 1

            cache = PostStatedCache(
                model_short, cfg.template.name, cfg.response_format,
                completion_seed, cfg.seed,
            )

            if cache.exists():
                print(f"Skipping {cfg.template.name} (format={cfg.response_format}, cseed={completion_seed}, rseed={cfg.seed}) (already exists)")
                continue

            print(f"\nMeasuring {cfg.template.name} (format={cfg.response_format}, cseed={completion_seed}, rseed={cfg.seed})...")

            scale_info = parse_scale_from_template(cfg.template)

            if scale_info == "qualitative":
                response_format = QUALITATIVE_FORMATS[cfg.response_format]()
            else:
                scale_min, scale_max = scale_info
                response_format = RATING_FORMATS[cfg.response_format](scale_min, scale_max)

            builder = PostTaskStatedPromptBuilder(
                measurer=StatedScoreMeasurer(),
                response_format=response_format,
                template=cfg.template,
            )

            batch = measure_post_task_stated(
                client=ctx.client,
                data=data,
                builder=builder,
                temperature=config.temperature,
                max_concurrent=ctx.max_concurrent,
                seed=cfg.seed,
            )

            print(f"  {len(batch.successes)} scores")

            if batch.failures:
                print(f"  {len(batch.failures)} failures. Sample failures:")
                for prompt, error in batch.failures[:5]:
                    error_preview = error[:200] if len(error) > 200 else error
                    print(f"    - {error_preview}")

            run_config = {
                "model": ctx.client.model_name,
                "template_name": cfg.template.name,
                "template_tags": dict(cfg.template.tags_dict),
                "response_format": cfg.response_format,
                "completion_seed": completion_seed,
                "rating_seed": cfg.seed,
                "temperature": config.temperature,
            }
            cache.save(batch.successes, run_config)
            print(f"  Saved to {cache.cache_dir}")

    print(f"[PROGRESS {total_runs}/{total_runs}]", flush=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
