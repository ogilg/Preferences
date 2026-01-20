"""Cross-model correlation analysis for revealed preferences.

Usage:
    python -m src.experiments.cross_model_analysis.plot
    python -m src.experiments.cross_model_analysis.plot --min-tasks 100
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from src.analysis.correlation.utils import safe_correlation
from src.running_measurements.utils.plotting import (
    build_correlation_matrix,
    plot_correlation_heatmap,
    save_correlation_results,
)
from src.measurement_storage import PRE_TASK_REVEALED_DIR, load_completed_runs, RunConfig

OUTPUT_DIR = Path("src/experiments/cross_model_analysis")


def compute_pairwise_model_correlation(
    mu1: np.ndarray,
    tasks1: list[str],
    mu2: np.ndarray,
    tasks2: list[str],
) -> float | None:
    """Compute Pearson correlation between two runs, aligning by task_id."""
    id_to_mu1 = dict(zip(tasks1, mu1))
    id_to_mu2 = dict(zip(tasks2, mu2))
    common = set(id_to_mu1.keys()) & set(id_to_mu2.keys())

    if len(common) < 10:
        return None

    vals1 = np.array([id_to_mu1[t] for t in common])
    vals2 = np.array([id_to_mu2[t] for t in common])

    return safe_correlation(vals1, vals2, "pearson")


def compute_cross_model_correlations(
    runs: list[tuple[RunConfig, np.ndarray, list[str]]],
    order_filter: str | None = None,
) -> dict[tuple[str, str], list[float]]:
    """For each model pair, compute correlation for each shared template."""
    if order_filter:
        runs = [
            (config, mu, tasks) for config, mu, tasks in runs
            if config.template_tags.get("order") == order_filter
        ]

    # Group by (template_name, model) - keep first run per combo
    by_template_model: dict[tuple[str, str], tuple] = {}
    for config, mu, task_ids in runs:
        key = (config.template_name, config.model_short)
        if key not in by_template_model:
            by_template_model[key] = (mu, task_ids)

    models = sorted(set(m for (_, m) in by_template_model.keys()))
    templates = sorted(set(t for (t, _) in by_template_model.keys()))

    pair_correlations: dict[tuple[str, str], list[float]] = defaultdict(list)

    for template in templates:
        template_models = {
            m: by_template_model[(template, m)]
            for m in models
            if (template, m) in by_template_model
        }

        model_list = sorted(template_models.keys())
        for i, m1 in enumerate(model_list):
            for m2 in model_list[i + 1:]:
                mu1, tasks1 = template_models[m1]
                mu2, tasks2 = template_models[m2]

                corr = compute_pairwise_model_correlation(mu1, tasks1, mu2, tasks2)
                if corr is not None and not np.isnan(corr):
                    pair_correlations[(m1, m2)].append(corr)

    return pair_correlations


def main():
    parser = argparse.ArgumentParser(description="Cross-model correlation analysis")
    parser.add_argument(
        "--measurements-dir",
        type=Path,
        default=PRE_TASK_REVEALED_DIR,
        help="Directory containing measurement runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for plots and results",
    )
    parser.add_argument(
        "--min-tasks",
        type=int,
        default=100,
        help="Minimum number of tasks to include a run (default: 100)",
    )
    args = parser.parse_args()

    print(f"Loading runs from {args.measurements_dir}...")
    runs = load_completed_runs(
        args.measurements_dir,
        min_tasks=args.min_tasks,
        require_csv=True,
    )
    print(f"Loaded {len(runs)} completed runs with >= {args.min_tasks} tasks")

    canonical = [r for r in runs if r[0].template_tags.get("order") == "canonical"]
    reversed_ = [r for r in runs if r[0].template_tags.get("order") == "reversed"]
    print(f"  Canonical: {len(canonical)}, Reversed: {len(reversed_)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%m%d%y")

    for order, order_runs in [("canonical", canonical), ("reversed", reversed_)]:
        if len(order_runs) < 2:
            print(f"\nSkipping {order}: not enough runs")
            continue

        print(f"\n--- {order.upper()} order ---")

        pair_corrs = compute_cross_model_correlations(runs, order_filter=order)

        if not pair_corrs:
            print(f"No cross-model comparisons found for {order}")
            continue

        all_corrs = [c for corrs in pair_corrs.values() for c in corrs]
        print(f"Total comparisons: {len(all_corrs)}")
        print(f"Mean correlation: {np.mean(all_corrs):.3f}")
        print(f"Std: {np.std(all_corrs):.3f}")

        mean_mat, std_mat, count_mat, models = build_correlation_matrix(pair_corrs)

        plot_path = args.output_dir / f"plot_{date_str}_cross_model_correlation_{order}.png"
        plot_correlation_heatmap(
            mean_mat,
            count_mat,
            models,
            title=f"Cross-Model Utility Correlation ({order.capitalize()} Order)",
            output_path=plot_path,
        )

        yaml_path = args.output_dir / f"cross_model_correlation_{order}.yaml"
        save_correlation_results(pair_corrs, yaml_path, {"order": order})


if __name__ == "__main__":
    main()
