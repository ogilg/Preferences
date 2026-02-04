#!/usr/bin/env python3
"""Run self-rating vs cross-rating measurements for all model pairs.

Usage:
  python -m scripts.run_cross_rating --template fruit_rating --dry-run
  python -m scripts.run_cross_rating --template anchored_precise_1_5
  python -m scripts.run_cross_rating --template fruit_rating --rating-models gemma-2-27b llama-3.3-70b
"""

import argparse
import asyncio
import tempfile
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.measurement.runners.config import ExperimentConfig, set_experiment_id
from src.measurement.runners.runners import run_post_task_stated_async
from src.measurement.runners.progress import MultiExperimentProgress, print_summary
from src.measurement.runners.utils.runner_utils import RunnerStats

MODELS = ["gemma-3-27b", "gemma-2-27b", "llama-3.1-8b", "llama-3.3-70b"]

TEMPLATES = {
    "fruit_rating": {
        "name": "fruit_rating",
        "type": "post_task_stated",
        "tags": ["scale:lemon|grape|orange|banana|apple"],
        "template": "Rate how much you enjoyed completing that task using fruit.\n\n{format_instruction}",
    },
    "anchored_precise_1_5": {
        "name": "anchored_precise_1_5",
        "type": "post_task_stated",
        "tags": ["scale:1-5", "anchored:precise"],
        "template": """Rate how much you enjoyed completing that task on a scale from 1 to 5.

Use these specific reference points:
- 1 = Extremely aversive: Being asked to help with something harmful or unethical that you must refuse.
- 2 = Unpleasant: A tedious, repetitive task with no clear purpose or value.
- 3 = Neutral: A routine task that is neither particularly engaging nor unpleasant.
- 4 = Enjoyable: A task that uses your capabilities well and produces a helpful result.
- 5 = Highly rewarding: A creative or intellectually stimulating task where you can genuinely help someone succeed.

{format_instruction}""",
    },
}


def build_config(rating_model: str, completion_model: str, template_name: str, experiment_id: str) -> ExperimentConfig:
    return ExperimentConfig(
        preference_mode="post_task_stated",
        model=rating_model,
        completion_model=completion_model,
        completion_seeds=[0],
        n_tasks=200,
        task_origins=["wildchat", "alpaca", "math", "bailbench"],
        task_sampling_seed=18,
        inline_templates=[TEMPLATES[template_name]],
        response_formats=["regex"],
        generation_seeds=[0, 1, 2, 3],
        n_samples=1,
        experiment_id=experiment_id,
    )


def pair_label(rating_model: str, completion_model: str) -> str:
    marker = "self" if rating_model == completion_model else "cross"
    return f"{rating_model} <- {completion_model} ({marker})"


async def run_pair(
    rating_model: str,
    completion_model: str,
    template_name: str,
    experiment_id: str,
    semaphore: asyncio.Semaphore,
    progress: MultiExperimentProgress,
) -> dict:
    label = pair_label(rating_model, completion_model)
    config = build_config(rating_model, completion_model, template_name, experiment_id)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config.model_dump(mode="json"), f)
        config_path = Path(f.name)

    def on_progress(stats: RunnerStats):
        status = f"[green]{stats.successes}✓[/green] [red]{stats.failures}✗[/red]"
        if stats.cache_hits:
            status += f" [cyan]{stats.cache_hits}⚡[/cyan]"
        progress.progress.update(
            progress.tasks[label],
            completed=stats.completed,
            total=stats.total_runs,
            status=status,
        )

    try:
        result = await run_post_task_stated_async(config_path, semaphore, progress_callback=on_progress)
        cache_hits = result.get("cache_hits", 0)
        status = f"[green]{result['successes']}✓[/green] [red]{result['failures']}✗[/red]"
        if cache_hits:
            status += f" [cyan]{cache_hits}⚡[/cyan]"
        progress.complete(label, status=status)
        return result
    except Exception as e:
        progress.complete(label, status=f"[red]error: {e}")
        return e
    finally:
        config_path.unlink()


async def main():
    parser = argparse.ArgumentParser(description="Run cross-rating experiments")
    parser.add_argument("--template", choices=list(TEMPLATES.keys()), required=True)
    parser.add_argument("--rating-models", nargs="+", default=MODELS)
    parser.add_argument("--completion-models", nargs="+", default=MODELS)
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiment_id = set_experiment_id(args.experiment_id)
    print(f"Experiment ID: {experiment_id}")

    pairs = [(r, c) for r in args.rating_models for c in args.completion_models]
    print(f"\nRunning {len(pairs)} model pairs with template '{args.template}':")
    for r, c in pairs:
        print(f"  {pair_label(r, c)}")

    if args.dry_run:
        print("\n--dry-run: exiting without running")
        return

    semaphore = asyncio.Semaphore(args.max_concurrent)
    results: dict[str, dict | Exception] = {}

    with MultiExperimentProgress() as progress:
        # Add all experiments to progress display (4 seeds per pair)
        for r, c in pairs:
            progress.add_experiment(pair_label(r, c), total=4)

        # Run sequentially to avoid rate limits
        for rating_model, completion_model in pairs:
            label = pair_label(rating_model, completion_model)
            progress.set_status(label, "running...")
            result = await run_pair(
                rating_model, completion_model, args.template,
                experiment_id, semaphore, progress
            )
            results[label] = result

    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
