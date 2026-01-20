"""Unified entry point for running experiments.

Usage:
  # Single config
  python -m src.running_measurements.run config.yaml
  python -m src.running_measurements.run config.yaml --max-concurrent 100

  # Multiple configs in parallel
  python -m src.running_measurements.run config1.yaml config2.yaml --max-concurrent 50
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.running_measurements.config import load_experiment_config, set_experiment_id, get_experiment_id
from src.running_measurements.runners import RUNNERS
from src.running_measurements.progress import (
    MultiExperimentProgress,
    print_summary,
    console,
)

DEFAULT_MAX_CONCURRENT = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preference measurement experiments")
    parser.add_argument("configs", nargs="+", type=Path, help="Config file(s) to run")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
                        help=f"Max concurrent API requests (default: {DEFAULT_MAX_CONCURRENT})")
    parser.add_argument("--experiment-id", type=str, default=None,
                        help="Experiment ID for tracking (auto-generated if not provided)")
    parser.add_argument("--dry-run", action="store_true", help="List experiments without running")
    parser.add_argument("--debug", action="store_true", help="Show example errors for each failure category")
    return parser.parse_args()


async def run_experiments(
    config_paths: list[Path],
    semaphore: asyncio.Semaphore,
    experiment_id: str | None = None,
) -> dict[str, dict | Exception]:
    """Run experiments with concurrent progress display."""

    # Load configs to get labels and totals
    configs = []
    for path in config_paths:
        config = load_experiment_config(path)
        # CLI experiment_id overrides config file
        if experiment_id is not None:
            config.experiment_id = experiment_id
        label = f"{path.stem}:{config.model}"
        configs.append((path, config, label))

    results: dict[str, dict | Exception] = {}

    with MultiExperimentProgress() as progress:
        # Add all experiments to progress display
        for path, config, label in configs:
            # Estimate total based on config
            n_configs = len(config.response_formats) * len(config.generation_seeds)
            if config.n_template_samples:
                n_configs = config.n_template_samples
            # Post-task experiments iterate over completion seeds
            if config.preference_mode.startswith("post_task"):
                completion_seeds = config.completion_seeds or config.generation_seeds
                n_configs *= len(completion_seeds)
            progress.add_experiment(label, total=n_configs)

        async def run_one(path: Path, config, label: str) -> tuple[str, dict | Exception]:
            runner = RUNNERS.get(config.preference_mode)

            if runner is None:
                progress.complete(label, status="[red]no runner")
                return label, ValueError(f"No runner for mode: {config.preference_mode}")

            progress.set_status(label, "running...")
            last_update_time: list[float | None] = [None]

            def on_progress(completed: int, total: int):
                now = time.time()
                if last_update_time[0] is None:
                    iter_str = "[dim]—[/dim]"
                else:
                    iter_time = now - last_update_time[0]
                    iter_str = f"[dim]{iter_time:.1f}s/iter[/dim]"
                last_update_time[0] = now
                progress.progress.update(progress.tasks[label], completed=completed, total=total, status=iter_str)

            try:
                result = await runner(path, semaphore, progress_callback=on_progress)
                skipped = result.get('skipped', 0)
                status = f"[green]{result['successes']}✓ {result['failures']}✗"
                if skipped:
                    status += f" [dim]{skipped}⊘[/dim]"
                progress.complete(label, status=status)
                return label, result
            except Exception as e:
                progress.complete(label, status=f"[red]error: {e}")
                return label, e

        # Run all experiments concurrently
        tasks = [run_one(path, config, label) for path, config, label in configs]
        completed = await asyncio.gather(*tasks)
        results = dict(completed)

    return results


def main():
    args = parse_args()

    # Validate configs
    for config_path in args.configs:
        if not config_path.exists():
            console.print(f"[red]Error: Config not found: {config_path}")
            return 1

    if args.dry_run:
        console.print("[bold]Experiments to run:")
        for config_path in args.configs:
            config = load_experiment_config(config_path)
            console.print(f"  • {config_path.stem}: {config.model}")
        return 0

    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Set experiment ID for this run (auto-generates timestamp if not provided)
    exp_id = set_experiment_id(args.experiment_id)
    console.print(f"[bold]Experiment ID: {exp_id}")
    console.print(f"[bold]Running {len(args.configs)} experiment(s) with max {args.max_concurrent} concurrent requests\n")

    results = asyncio.run(run_experiments(args.configs, semaphore, experiment_id=exp_id))
    print_summary(results, debug=args.debug)

    # Return non-zero if any failures
    has_errors = any(isinstance(r, Exception) for r in results.values())
    return 1 if has_errors else 0


if __name__ == "__main__":
    exit(main())
