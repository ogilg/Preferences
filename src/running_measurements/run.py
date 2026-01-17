"""Unified entry point for running experiments.

Usage:
  # Single config
  python -m src.running_measurements.run config.yaml
  python -m src.running_measurements.run config.yaml --max-concurrent 100

  # Multiple configs in parallel
  python -m src.running_measurements.run --multi config1.yaml config2.yaml --max-concurrent 50
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.running_measurements.config import load_experiment_config
from src.running_measurements.runners import RUNNERS

DEFAULT_MAX_CONCURRENT = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preference measurement experiments")
    parser.add_argument("configs", nargs="+", type=Path, help="Config file(s) to run")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
                        help=f"Max concurrent API requests (default: {DEFAULT_MAX_CONCURRENT})")
    parser.add_argument("--dry-run", action="store_true", help="List experiments without running")
    return parser.parse_args()


async def run_single(config_path: Path, semaphore: asyncio.Semaphore) -> dict:
    """Run a single experiment."""
    config = load_experiment_config(config_path)
    runner = RUNNERS.get(config.preference_mode)

    if runner is None:
        raise ValueError(f"No runner for mode: {config.preference_mode}. "
                        f"Supported: {list(RUNNERS.keys())}")

    print(f"Running {config.preference_mode} for {config.model}...")
    return await runner(config_path, semaphore)


async def run_multi(config_paths: list[Path], semaphore: asyncio.Semaphore) -> dict[str, dict | Exception]:
    """Run multiple experiments in parallel with shared semaphore."""
    async def run_one(config_path: Path) -> tuple[str, dict | Exception]:
        config = load_experiment_config(config_path)
        label = f"{config.preference_mode}:{config.model}"
        runner = RUNNERS.get(config.preference_mode)

        if runner is None:
            return label, ValueError(f"No runner for mode: {config.preference_mode}")

        try:
            print(f"Starting: {label}")
            result = await runner(config_path, semaphore)
            print(f"Completed: {label} - {result['successes']} successes, {result['failures']} failures")
            return label, result
        except Exception as e:
            print(f"Failed: {label} - {e}")
            return label, e

    results = await asyncio.gather(*[run_one(p) for p in config_paths])
    return dict(results)


def main():
    args = parse_args()

    # Validate configs
    for config_path in args.configs:
        if not config_path.exists():
            print(f"Error: Config not found: {config_path}")
            return 1

    if args.dry_run:
        print("Experiments to run:")
        for config_path in args.configs:
            config = load_experiment_config(config_path)
            print(f"  - {config.preference_mode}: {config.model}")
        return 0

    semaphore = asyncio.Semaphore(args.max_concurrent)

    if len(args.configs) == 1:
        result = asyncio.run(run_single(args.configs[0], semaphore))
        print(f"\nCompleted: {result['successes']} successes, {result['failures']} failures")
    else:
        print(f"Running {len(args.configs)} experiments with max {args.max_concurrent} concurrent requests")
        results = asyncio.run(run_multi(args.configs, semaphore))

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for label, result in results.items():
            if isinstance(result, Exception):
                print(f"  {label}: FAILED - {result}")
            else:
                print(f"  {label}: {result['successes']} successes, {result['failures']} failures")

    return 0


if __name__ == "__main__":
    exit(main())
