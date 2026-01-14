"""Orchestrate running experiments across multiple models in parallel.

Usage: python -m src.experiments.run_multi_model \\
  --models "qwen/qwen-2.5-7b-instruct,meta-llama/llama-3.1-8b-instruct" \\
  --base-config-active src/experiments/configs/active_learning.yaml \\
  --base-config-stated src/experiments/configs/stated.yaml \\
  --max-concurrent 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.multi_model_utils import (
    generate_experiment_configs,
    run_experiments_parallel,
    print_summary,
)

CONFIG_DIR = Path("configs/multi_model")
LOG_DIR = Path("results/multi_model_logs")
MAX_CONCURRENT_PER_EXPERIMENT = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiments across multiple models in parallel")
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument("--base-config-active", type=Path, required=True, help="Base config for active_learning")
    parser.add_argument("--base-config-stated", type=Path, required=True, help="Base config for stated_measurement")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_PER_EXPERIMENT, help=f"Max concurrent requests per experiment (default: {MAX_CONCURRENT_PER_EXPERIMENT})")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs without running")
    return parser.parse_args()


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",")]

    if not args.base_config_active.exists():
        print(f"Error: Base config not found: {args.base_config_active}")
        return 1

    if not args.base_config_stated.exists():
        print(f"Error: Base config not found: {args.base_config_stated}")
        return 1

    print("=" * 60)
    print("Multi-Model Experiment Configuration")
    print("=" * 60)
    print(f"Models: {len(models)}")
    for model in models:
        print(f"  - {model}")
    print(f"Total experiments: {len(models) * 2}")
    print(f"Max concurrent per experiment: {args.max_concurrent}")
    print(f"Max total concurrent requests: {len(models) * 2 * args.max_concurrent}")
    print()

    tasks = generate_experiment_configs(
        models=models,
        base_config_active=args.base_config_active,
        base_config_stated=args.base_config_stated,
        max_concurrent=args.max_concurrent,
        output_dir=CONFIG_DIR,
    )

    print(f"Generated {len(tasks)} config files in {CONFIG_DIR}")
    for task in tasks:
        print(f"  - {task.experiment_type} - {task.model}")

    if args.dry_run:
        print("\nDry run complete.")
        return 0

    print("\nStarting experiments...")
    results = run_experiments_parallel(tasks, LOG_DIR)
    print_summary(results, LOG_DIR)

    return sum(1 for success in results.values() if not success)


if __name__ == "__main__":
    exit(main())
