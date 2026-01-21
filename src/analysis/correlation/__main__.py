"""CLI for correlation analysis.

Usage:
    python -m src.analysis.correlation --model llama-3.1-8b
    python -m src.analysis.correlation --model llama-3.1-8b --types pre_stated post_stated
    python -m src.analysis.correlation --list-models
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.analysis.correlation.loading import (
    MeasurementType,
    load_runs_for_model,
    list_available_models,
    aggregate_runs_by_group,
)
from src.analysis.correlation.plot import (
    plot_correlation_matrix,
    plot_scatter_grid,
    plot_type_comparison,
)


OUTPUT_DIR = Path("src/analysis/correlation/plots")


def parse_measurement_types(type_strs: list[str] | None) -> list[MeasurementType] | None:
    if type_strs is None:
        return None
    return [MeasurementType(t) for t in type_strs]


def main():
    parser = argparse.ArgumentParser(description="Correlation analysis across measurement types")
    parser.add_argument("--model", type=str, help="Model short name (e.g., llama-3.1-8b)")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=[t.value for t in MeasurementType],
        help="Measurement types to include (default: all)",
    )
    parser.add_argument("--min-tasks", type=int, default=10, help="Minimum overlapping tasks")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument(
        "--require-thurstonian",
        action="store_true",
        help="For revealed preferences, require pre-computed thurstonian utilities",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate runs by (type, format, order), averaging across seeds/templates",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="Experiment ID to load from",
    )
    args = parser.parse_args()

    if args.list_models:
        models = list_available_models(args.experiment_id)
        print(f"Available models ({len(models)}):")
        for m in sorted(models):
            print(f"  {m}")
        return

    if not args.model:
        parser.error("--model is required (or use --list-models)")

    measurement_types = parse_measurement_types(args.types)

    print(f"Loading runs for model: {args.model} from experiment: {args.experiment_id}")
    runs = load_runs_for_model(
        args.model,
        args.experiment_id,
        measurement_types=measurement_types,
        min_tasks=args.min_tasks,
        require_thurstonian_csv=args.require_thurstonian,
    )

    if not runs:
        print("No runs found matching criteria")
        return

    print(f"Found {len(runs)} runs")

    if args.aggregate:
        runs = aggregate_runs_by_group(runs)
        print(f"Aggregated to {len(runs)} groups")

    print("Runs:")
    for r in runs:
        print(f"  {r.measurement_type.value}: {r.label} ({len(r.task_ids)} tasks)")

    # Generate plots
    date_str = datetime.now().strftime("%m%d%y")
    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Correlation matrix
    plot_correlation_matrix(
        runs,
        output_dir / f"plot_{date_str}_correlation_matrix.png",
        args.model,
    )

    # 2. Cross-type comparison bar chart
    plot_type_comparison(
        runs,
        output_dir / f"plot_{date_str}_type_comparison.png",
        args.model,
    )

    # 3. Scatter grid (if not too many runs)
    if len(runs) <= 8:
        plot_scatter_grid(
            runs,
            output_dir / f"plot_{date_str}_scatter_grid.png",
            args.model,
        )
    else:
        print(f"Skipping scatter grid (too many runs: {len(runs)})")


if __name__ == "__main__":
    main()
