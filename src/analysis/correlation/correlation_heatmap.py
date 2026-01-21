"""Flexible correlation analysis with template filtering and heatmaps.

Usage:
    python -m src.analysis.correlation.correlation_heatmap --model llama-3.1-8b --template-pattern "pre_task_rating*"
    python -m src.analysis.correlation.correlation_heatmap --model llama-3.1-8b --template-pattern "*qualitative*"
    python -m src.analysis.correlation.correlation_heatmap --model llama-3.1-8b --experiment-id exp_20260119_192232
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from fnmatch import fnmatch

import numpy as np
from scipy.stats import pearsonr

from src.analysis.correlation.loading import (
    MeasurementType,
    load_runs_for_model,
    list_available_models,
)
from src.analysis.correlation.plot import plot_correlation_matrix


OUTPUT_DIR = Path("src/analysis/correlation/plots")


def filter_runs_by_template(runs, template_pattern: str):
    """Filter runs by template name pattern (fnmatch style)."""
    filtered = []
    for run in runs:
        if fnmatch(run.config.template_name, template_pattern):
            filtered.append(run)
    return filtered


def compute_correlation_matrix(runs: list) -> tuple[np.ndarray, list[str]]:
    """Compute correlation matrix between runs."""
    n = len(runs)
    corr_matrix = np.ones((n, n))
    labels = []

    for i, run in enumerate(runs):
        labels.append(run.label)

    for i in range(n):
        for j in range(i + 1, n):
            run_i = runs[i]
            run_j = runs[j]

            # Get overlapping tasks
            tasks_i = set(run_i.task_ids)
            tasks_j = set(run_j.task_ids)
            overlap = sorted(tasks_i & tasks_j)

            if not overlap:
                corr_matrix[i, j] = np.nan
                corr_matrix[j, i] = np.nan
                continue

            # Map values to overlapping tasks
            values_i = np.array([run_i.as_dict()[tid] for tid in overlap])
            values_j = np.array([run_j.as_dict()[tid] for tid in overlap])

            # Compute Pearson correlation
            corr, _ = pearsonr(values_i, values_j)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return corr_matrix, labels


def main():
    parser = argparse.ArgumentParser(description="Flexible correlation analysis with heatmaps")
    parser.add_argument("--model", type=str, required=True, help="Model short name")
    parser.add_argument(
        "--template-pattern",
        type=str,
        help="Template name pattern (fnmatch, e.g., 'pre_task_rating*' or '*qualitative*')",
    )
    parser.add_argument("--experiment-id", type=str, required=True, help="Experiment ID to load from")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--min-tasks", type=int, default=10)
    parser.add_argument(
        "--types",
        nargs="+",
        choices=[t.value for t in MeasurementType],
        help="Measurement types to include",
    )
    args = parser.parse_args()

    # Parse measurement types
    measurement_types = None
    if args.types:
        measurement_types = [MeasurementType(t) for t in args.types]

    print(f"Loading runs for model: {args.model} from experiment: {args.experiment_id}")
    runs = load_runs_for_model(
        args.model,
        args.experiment_id,
        measurement_types=measurement_types,
        min_tasks=args.min_tasks,
    )

    if not runs:
        print("No runs found matching criteria")
        return

    print(f"Found {len(runs)} total runs")

    # Filter by template pattern if specified
    if args.template_pattern:
        runs = filter_runs_by_template(runs, args.template_pattern)
        print(f"Filtered to {len(runs)} runs matching '{args.template_pattern}'")

    if not runs:
        print("No runs match the template pattern")
        return

    print("\nRuns:")
    for r in runs:
        print(f"  {r.measurement_type.value}: {r.label} ({len(r.task_ids)} tasks)")

    # Generate plot
    date_str = datetime.now().strftime("%m%d%y")
    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with pattern
    pattern_str = args.template_pattern.replace("*", "star") if args.template_pattern else "all"
    plot_path = output_dir / f"plot_{date_str}_heatmap_{pattern_str}.png"

    plot_correlation_matrix(
        runs,
        plot_path,
        f"{args.model} - {args.template_pattern or 'all'}",
    )
    print(f"\nSaved plot to {plot_path}")


if __name__ == "__main__":
    main()
