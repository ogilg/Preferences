"""Sensitivity analysis for rating stated preference measurements.

Usage:
    python -m src.analysis.sensitivity.plot_rating --experiment-id probe_2
    python -m src.analysis.sensitivity.plot_rating --experiment-id probe_2 --pre-only
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.analysis.sensitivity.plot import (
    load_all_runs,
    compute_all_field_sensitivities,
    save_sensitivity_report,
    plot_sensitivity_bars,
    plot_regression_coefficients,
    plot_sensitivity_by_model,
)
from src.measurement_storage import EXPERIMENTS_DIR


OUTPUT_DIR = Path(__file__).parent / "plots"


def filter_rating(runs):
    """Keep only rating stated runs (exclude qualitative)."""
    return [
        (config, mu, task_ids)
        for config, mu, task_ids in runs
        if "qualitative" not in config.template_name
    ]


def get_sources(experiment_id: str, pre_only: bool, post_only: bool) -> list[tuple[str, Path]]:
    """Get (prefix, results_dir) pairs based on experiment_id and flags."""
    sources = []
    exp_dir = EXPERIMENTS_DIR / experiment_id
    if not post_only:
        sources.append(("pre_task", exp_dir / "pre_task_stated"))
    if not pre_only:
        sources.append(("post_task", exp_dir / "post_task_stated"))
    return sources


def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis for rating stated preferences")
    parser.add_argument("--pre-only", action="store_true", help="Only analyze pre-task")
    parser.add_argument("--post-only", action="store_true", help="Only analyze post-task")
    parser.add_argument("--experiment-id", type=str, required=True, help="Experiment ID to load from")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%m%d%y")

    sources = get_sources(args.experiment_id, args.pre_only, args.post_only)

    for prefix, results_dir in sources:
        if not results_dir.exists():
            print(f"Skipping {prefix}: {results_dir} does not exist")
            continue

        print(f"\nLoading {prefix} rating stated runs from {results_dir}...")
        all_runs = load_all_runs(results_dir)
        runs = filter_rating(all_runs)

        if not runs:
            print(f"No rating runs found for {prefix} (had {len(all_runs)} total runs)")
            continue

        print(f"Loaded {len(runs)} rating runs (filtered from {len(all_runs)}), computing correlations...")
        sensitivities, correlations, regression = compute_all_field_sensitivities(runs)

        models = sorted(set(config.model_short for config, _, _ in runs))
        model_str = models[0] if len(models) == 1 else f"{len(models)} models"
        n_runs = len(runs)

        report_path = OUTPUT_DIR / f"sensitivity_{prefix}_rating.yaml"
        save_sensitivity_report(sensitivities, correlations, regression, n_runs, report_path)
        print(f"Saved report to {report_path}")

        if sensitivities:
            plot_path = OUTPUT_DIR / f"plot_{date_str}_{prefix}_rating_averaging.png"
            title = f"{model_str} {prefix.replace('_', ' ').title()} Rating Sensitivity (n={n_runs})"
            plot_sensitivity_bars(sensitivities, plot_path, title)
            print(f"Saved plot to {plot_path}")

        if regression:
            plot_path = OUTPUT_DIR / f"plot_{date_str}_{prefix}_rating_regression.png"
            title = f"{model_str} {prefix.replace('_', ' ').title()} Rating Sensitivity (Regression, n={n_runs})"
            plot_regression_coefficients(regression, plot_path, title)
            print(f"Saved regression plot to {plot_path}")

        if len(models) > 1:
            plot_path = OUTPUT_DIR / f"plot_{date_str}_{prefix}_rating_by_model.png"
            plot_sensitivity_by_model(runs, plot_path, f"{prefix.replace('_', ' ').title()} Rating Sensitivity by Model")
            print(f"Saved per-model plot to {plot_path}")


if __name__ == "__main__":
    main()
