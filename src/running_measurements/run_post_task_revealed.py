"""Post-task revealed preference measurement.

Usage: python -m src.experiments.run_post_task_revealed <config.yaml>
"""

from __future__ import annotations

from itertools import combinations

from dotenv import load_dotenv

load_dotenv()

from src.prompt_templates import PostTaskRevealedPromptBuilder
from src.preference_measurement import (
    measure_post_task_revealed,
    RevealedPreferenceMeasurer,
    CHOICE_FORMATS,
)
from src.measurement_storage import CompletionStore, PostRevealedCache, model_short_name
from src.prompt_templates.sampler import (
    SampledConfiguration,
    sample_configurations_lhs,
    print_sampling_balance,
)
from src.running_measurements.utils.experiment_utils import parse_config_path, setup_experiment, flip_pairs


def generate_all_pairs(tasks):
    return list(combinations(tasks, 2))


def main():
    config_path = parse_config_path("Post-task revealed measurement")
    ctx = setup_experiment(config_path, expected_mode="post_task_revealed")
    config = ctx.config

    if ctx.templates is None:
        raise ValueError("Templates required for post_task_revealed mode")

    completion_seeds = config.completion_seeds or config.generation_seeds
    orders = ["canonical", "reversed"] if config.include_reverse_order else ["canonical"]
    model_short = model_short_name(ctx.client.canonical_model_name)

    if config.template_sampling == "lhs" and config.n_template_samples:
        lhs_seed = config.lhs_seed if config.lhs_seed is not None else 42
        configurations = sample_configurations_lhs(
            ctx.templates, config.response_formats, config.generation_seeds,
            n_samples=config.n_template_samples, orders=orders, seed=lhs_seed,
        )
        print(f"LHS sampling: {config.n_template_samples} configurations")
        print_sampling_balance(configurations)
    else:
        configurations = [
            SampledConfiguration(t, rf, s, o)
            for t in ctx.templates for rf in config.response_formats
            for s in config.generation_seeds for o in orders
        ]

    total_runs = len(completion_seeds) * len(configurations)
    print(f"Templates: {len(ctx.templates)}, Configs: {len(configurations)}, "
          f"Completion seeds: {completion_seeds}, Tasks: {len(ctx.tasks)}")

    # Generate all pairs (or use active learning if configured)
    all_pairs = generate_all_pairs(ctx.tasks)
    print(f"Total pairs: {len(all_pairs)}")

    run_idx = 0
    for completion_seed in completion_seeds:
        store = CompletionStore(client=ctx.client, seed=completion_seed)
        if not store.exists():
            print(f"No completions for seed {completion_seed}, run completion generation first")
            continue

        task_completions = store.load(ctx.task_lookup)
        completion_lookup = {tc.task.id: tc.completion for tc in task_completions}

        for cfg in configurations:
            print(f"[PROGRESS {run_idx}/{total_runs}]", flush=True)
            run_idx += 1

            cache = PostRevealedCache(
                model_short, cfg.template.name, cfg.response_format,
                cfg.order, completion_seed, cfg.seed,
            )

            # Get pairs we need to query
            existing_pairs = cache.get_existing_pairs()
            pairs = all_pairs if cfg.order == "canonical" else flip_pairs(all_pairs)
            pairs_to_query = [
                (a, b) for a, b in pairs
                if (a.id, b.id) not in existing_pairs
            ]

            # Replicate for n_samples
            pairs_to_query = pairs_to_query * config.n_samples

            if not pairs_to_query:
                print(f"Skipping {cfg.template.name} (order={cfg.order}, cseed={completion_seed}, rseed={cfg.seed}) (all cached)")
                continue

            print(f"\n{cfg.template.name} (order={cfg.order}, cseed={completion_seed}, rseed={cfg.seed}): {len(pairs_to_query)} pairs...")

            # Build data with completions
            data = [
                (task_a, task_b, completion_lookup[task_a.id], completion_lookup[task_b.id])
                for task_a, task_b in pairs_to_query
            ]

            # Get labels from template tags if available
            label_a = cfg.template.tags_dict.get("task_a_label", "Task A")
            label_b = cfg.template.tags_dict.get("task_b_label", "Task B")

            builder = PostTaskRevealedPromptBuilder(
                measurer=RevealedPreferenceMeasurer(),
                response_format=CHOICE_FORMATS[cfg.response_format](label_a, label_b),
                template=cfg.template,
            )

            batch = measure_post_task_revealed(
                client=ctx.client,
                data=data,
                builder=builder,
                temperature=config.temperature,
                max_concurrent=ctx.max_concurrent,
                seed=cfg.seed,
            )

            print(f"  {len(batch.successes)} measurements")

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
                "order": cfg.order,
                "completion_seed": completion_seed,
                "rating_seed": cfg.seed,
                "temperature": config.temperature,
            }
            cache.append(batch.successes, run_config)

    print(f"[PROGRESS {total_runs}/{total_runs}]", flush=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
