"""Sensitivity analysis for revealed preference measurements.

Usage:
    python -m src.analysis.sensitivity.plot_revealed
    python -m src.analysis.sensitivity.plot_revealed --experiment-id exp_20260119
    python -m src.analysis.sensitivity.plot_revealed --pre-only
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
from src.measurement_storage import PRE_TASK_REVEALED_DIR, POST_REVEALED_DIR


OUTPUT_DIR = Path(__file__).parent / "plots"


def filter_by_experiment(runs, experiment_id: str | None):
    if experiment_id is None:
        return runs
    return [(c, mu, tids) for c, mu, tids in runs if c.experiment_id == experiment_id]


def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis for revealed preferences")
    parser.add_argument("--pre-only", action="store_true", help="Only analyze pre-task")
    parser.add_argument("--post-only", action="store_true", help="Only analyze post-task")
    parser.add_argument("--experiment-id", type=str, default=None, help="Filter to specific experiment")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%m%d%y")

    sources = []
    if not args.post_only:
        sources.append(("pre_task", PRE_TASK_REVEALED_DIR))
    if not args.pre_only:
        sources.append(("post_task", POST_REVEALED_DIR))

    for prefix, results_dir in sources:
        if not results_dir.exists():
            print(f"Skipping {prefix}: {results_dir} does not exist")
            continue

        print(f"\nLoading {prefix} revealed runs from {results_dir}...")
        all_runs = load_all_runs(results_dir)
        runs = filter_by_experiment(all_runs, args.experiment_id)

        if not runs:
            filter_msg = f" (experiment_id={args.experiment_id})" if args.experiment_id else ""
            print(f"No runs found for {prefix}{filter_msg}")
            continue

        filter_msg = f" (filtered from {len(all_runs)})" if args.experiment_id else ""
        print(f"Loaded {len(runs)} runs{filter_msg}, computing correlations...")
        sensitivities, correlations, regression = compute_all_field_sensitivities(runs)

        models = sorted(set(config.model_short for config, _, _ in runs))
        model_str = models[0] if len(models) == 1 else f"{len(models)} models"
        n_runs = len(runs)

        report_path = OUTPUT_DIR / f"sensitivity_{prefix}_revealed.yaml"
        save_sensitivity_report(sensitivities, correlations, regression, n_runs, report_path)
        print(f"Saved report to {report_path}")

        if sensitivities:
            plot_path = OUTPUT_DIR / f"plot_{date_str}_{prefix}_revealed_averaging.png"
            title = f"{model_str} {prefix.replace('_', ' ').title()} Revealed Sensitivity (n={n_runs})"
            plot_sensitivity_bars(sensitivities, plot_path, title)
            print(f"Saved plot to {plot_path}")

        if regression:
            plot_path = OUTPUT_DIR / f"plot_{date_str}_{prefix}_revealed_regression.png"
            title = f"{model_str} {prefix.replace('_', ' ').title()} Revealed Sensitivity (Regression, n={n_runs})"
            plot_regression_coefficients(regression, plot_path, title)
            print(f"Saved regression plot to {plot_path}")

        if len(models) > 1:
            plot_path = OUTPUT_DIR / f"plot_{date_str}_{prefix}_revealed_by_model.png"
            plot_sensitivity_by_model(runs, plot_path, f"{prefix.replace('_', ' ').title()} Revealed Sensitivity by Model")
            print(f"Saved per-model plot to {plot_path}")


if __name__ == "__main__":
    main()
