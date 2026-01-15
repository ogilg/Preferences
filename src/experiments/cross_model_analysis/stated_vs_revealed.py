"""Correlation analysis between stated and revealed preferences.

For each model, computes correlation between stated preference scores
and revealed preference utilities on overlapping tasks.

Usage:
    python -m src.experiments.cross_model_analysis.stated_vs_revealed
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.experiments.correlation import safe_correlation
from src.experiments.loading import load_completed_runs, RunConfig
from src.experiments.plotting import (
    build_correlation_matrix,
    plot_correlation_heatmap,
    save_correlation_results,
)
from src.preferences.storage import MEASUREMENTS_DIR

STATED_DIR = Path("results/stated")
OUTPUT_DIR = Path("src/experiments/cross_model_analysis")


def compute_utility_correlation(
    mu1: np.ndarray,
    tasks1: list[str],
    mu2: np.ndarray,
    tasks2: list[str],
    min_overlap: int = 10,
) -> tuple[float | None, int]:
    """Compute Pearson correlation on overlapping tasks. Returns (corr, n_overlap)."""
    id_to_mu1 = dict(zip(tasks1, mu1))
    id_to_mu2 = dict(zip(tasks2, mu2))
    common = set(id_to_mu1.keys()) & set(id_to_mu2.keys())

    if len(common) < min_overlap:
        return None, len(common)

    vals1 = np.array([id_to_mu1[t] for t in common])
    vals2 = np.array([id_to_mu2[t] for t in common])

    corr = safe_correlation(vals1, vals2, "pearson")
    return corr, len(common)


def compute_stated_vs_revealed_correlations(
    stated_runs: list[tuple[RunConfig, np.ndarray, list[str]]],
    revealed_runs: list[tuple[RunConfig, np.ndarray, list[str]]],
) -> dict[str, list[dict]]:
    """Compute correlations between stated and revealed for each model.

    Returns dict mapping model -> list of correlation entries.
    """
    # Group by model
    stated_by_model: dict[str, list] = defaultdict(list)
    for config, mu, tasks in stated_runs:
        stated_by_model[config.model_short].append((config, mu, tasks))

    revealed_by_model: dict[str, list] = defaultdict(list)
    for config, mu, tasks in revealed_runs:
        revealed_by_model[config.model_short].append((config, mu, tasks))

    results: dict[str, list[dict]] = {}

    common_models = set(stated_by_model.keys()) & set(revealed_by_model.keys())

    for model in sorted(common_models):
        model_corrs = []

        for s_config, s_mu, s_tasks in stated_by_model[model]:
            for r_config, r_mu, r_tasks in revealed_by_model[model]:
                corr, n_overlap = compute_utility_correlation(s_mu, s_tasks, r_mu, r_tasks)

                if corr is not None and not np.isnan(corr):
                    model_corrs.append({
                        "stated_template": s_config.template_name,
                        "stated_format": s_config.template_tags.get("response_format", "unknown"),
                        "revealed_template": r_config.template_name,
                        "revealed_format": r_config.template_tags.get("response_format", "unknown"),
                        "revealed_order": r_config.template_tags.get("order", "unknown"),
                        "correlation": corr,
                        "n_overlap": n_overlap,
                    })

        if model_corrs:
            results[model] = model_corrs

    return results


def plot_model_summary(
    correlations_by_model: dict[str, list[dict]],
    output_path: Path,
    title: str,
) -> None:
    """Plot bar chart of mean stated-revealed correlation per model."""
    models = []
    means = []
    stds = []
    counts = []

    for model in sorted(correlations_by_model.keys()):
        corrs = [c["correlation"] for c in correlations_by_model[model]]
        models.append(model)
        means.append(np.mean(corrs))
        stds.append(np.std(corrs))
        counts.append(len(corrs))

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)

    for bar, mean, n in zip(bars, means, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean:.2f}\n(n={n})",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stated vs revealed correlation analysis")
    parser.add_argument(
        "--stated-dir",
        type=Path,
        default=STATED_DIR,
        help="Directory containing stated preference runs",
    )
    parser.add_argument(
        "--revealed-dir",
        type=Path,
        default=MEASUREMENTS_DIR,
        help="Directory containing revealed preference runs",
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
        default=10,
        help="Minimum number of overlapping tasks (default: 10)",
    )
    args = parser.parse_args()

    print(f"Loading stated runs from {args.stated_dir}...")
    stated_runs = load_completed_runs(args.stated_dir, min_tasks=args.min_tasks)
    print(f"Loaded {len(stated_runs)} stated runs")

    print(f"Loading revealed runs from {args.revealed_dir}...")
    revealed_runs = load_completed_runs(
        args.revealed_dir,
        min_tasks=args.min_tasks,
        require_csv=True,
    )
    print(f"Loaded {len(revealed_runs)} revealed runs")

    # Show models in each
    stated_models = sorted(set(c.model_short for c, _, _ in stated_runs))
    revealed_models = sorted(set(c.model_short for c, _, _ in revealed_runs))
    print(f"\nStated models: {stated_models}")
    print(f"Revealed models: {revealed_models}")
    print(f"Common models: {sorted(set(stated_models) & set(revealed_models))}")

    correlations = compute_stated_vs_revealed_correlations(stated_runs, revealed_runs)

    if not correlations:
        print("\nNo overlapping models with sufficient data found.")
        return

    # Summary stats
    print("\n--- Results by model ---")
    all_corrs = []
    for model, corrs in sorted(correlations.items()):
        vals = [c["correlation"] for c in corrs]
        all_corrs.extend(vals)
        print(f"{model}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, n={len(vals)}")

    print(f"\nOverall: mean={np.mean(all_corrs):.3f}, std={np.std(all_corrs):.3f}, n={len(all_corrs)}")

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%m%d%y")

    # Bar plot
    plot_path = args.output_dir / f"plot_{date_str}_stated_vs_revealed_by_model.png"
    plot_model_summary(
        correlations,
        plot_path,
        title="Stated vs Revealed Preference Correlation by Model",
    )

    # Save detailed results
    import yaml
    yaml_path = args.output_dir / "stated_vs_revealed_correlations.yaml"
    results = {
        "summary": {
            "mean": float(np.mean(all_corrs)),
            "std": float(np.std(all_corrs)),
            "n_comparisons": len(all_corrs),
        },
        "by_model": {
            model: {
                "mean": float(np.mean([c["correlation"] for c in corrs])),
                "std": float(np.std([c["correlation"] for c in corrs])),
                "n": len(corrs),
                "details": corrs,
            }
            for model, corrs in correlations.items()
        },
    }
    with open(yaml_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    print(f"Saved: {yaml_path}")


if __name__ == "__main__":
    main()
