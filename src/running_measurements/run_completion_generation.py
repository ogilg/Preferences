"""Generate task completions for post-task measurements.

Usage: python -m src.experiments.run_completion_generation <config.yaml>
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from src.measurement_storage import CompletionStore, generate_completions
from src.running_measurements.utils.experiment_utils import parse_config_path, setup_experiment


def main():
    ctx = setup_experiment(
        parse_config_path("Generate task completions"),
        expected_mode="completion_generation",
        max_new_tokens=512,  # Longer for completions
    )
    config = ctx.config

    print(f"Tasks: {len(ctx.tasks)}, Seeds: {config.generation_seeds}")

    for i, seed in enumerate(config.generation_seeds):
        print(f"[PROGRESS {i}/{len(config.generation_seeds)}]", flush=True)

        store = CompletionStore(client=ctx.client, seed=seed)

        # Skip tasks we already have completions for
        existing_ids = store.get_existing_task_ids()
        tasks_to_generate = [t for t in ctx.tasks if t.id not in existing_ids]

        if not tasks_to_generate:
            print(f"Seed {seed}: All {len(ctx.tasks)} completions already exist")
            continue

        print(f"Seed {seed}: Generating {len(tasks_to_generate)} completions ({len(existing_ids)} cached)...")
        completions = generate_completions(
            client=ctx.client,
            tasks=tasks_to_generate,
            temperature=config.temperature,
            max_concurrent=ctx.max_concurrent,
            seed=seed,
        )

        store.save(completions, config={
            "model": ctx.client.model_name,
            "model_short": store.model_short,
            "seed": seed,
            "temperature": config.temperature,
            "n_tasks": config.n_tasks,
            "task_origins": config.task_origins,
        })
        print(f"  Saved {len(completions)} completions to {store.store_dir}")

    print(f"[PROGRESS {len(config.generation_seeds)}/{len(config.generation_seeds)}]", flush=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
