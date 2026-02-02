"""Plot rating variance and distribution across models from experiment data.

Usage:
    python -m src.analysis.sensitivity.plot_rating_variance --experiment-id multi_model_discrimination_v1
    python -m src.analysis.sensitivity.plot_rating_variance --experiment-id multi_model_discrimination_v1 --template bipolar
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.measurement.storage import (
    EXPERIMENTS_DIR,
    extract_model_from_run_dir,
    extract_template_from_run_dir,
    normalize_score,
    parse_scale_tag,
)

OUTPUT_DIR = Path("src/analysis/sensitivity/plots")


def load_all_measurements(
    experiment_dir: Path,
    template_filter: str | None = None,
    normalize: bool = False,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Load all measurements, grouped by model name.

    Returns (scores_by_model, scores_by_template).
    """
    results_dir = experiment_dir / "post_task_stated"
    if not results_dir.exists():
        return {}, {}

    scores_by_model: dict[str, list[float]] = defaultdict(list)
    scores_by_template: dict[str, list[float]] = defaultdict(list)

    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.yaml"
        measurements_path = run_dir / "measurements.yaml"
        if not config_path.exists() or not measurements_path.exists():
            continue

        model = extract_model_from_run_dir(run_dir.name)
        template = extract_template_from_run_dir(run_dir.name)
        if not model or not template:
            continue

        if template_filter and template_filter not in template:
            continue

        # Skip qualitative templates (non-numeric)
        if "qualitative" in template:
            continue

        with open(config_path) as f:
            config = yaml.safe_load(f)

        scale_tag = config.get("template_tags", {}).get("scale")
        scale = parse_scale_tag(scale_tag) if scale_tag else None

        with open(measurements_path) as f:
            measurements = yaml.safe_load(f)

        if not measurements:
            continue

        for m in measurements:
            if "score" not in m or not isinstance(m["score"], (int, float)):
                continue
            score = float(m["score"])
            if normalize and scale:
                score = normalize_score(score, scale)
            scores_by_model[model].append(score)
            scores_by_template[template].append(score)

    return dict(scores_by_model), dict(scores_by_template)


def plot_variance_comparison(
    scores_by_group: dict[str, list[float]],
    output_path: Path,
    title: str,
    ylabel: str = "Score",
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot variance comparison as violin + bar chart."""
    model_stats = []
    for name, scores in scores_by_group.items():
        arr = np.array(scores)
        model_stats.append({
            "name": name,
            "scores": arr,
            "mean": arr.mean(),
            "std": arr.std(),
            "var": arr.var(),
            "n": len(arr),
        })

    model_stats.sort(key=lambda x: x["var"], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Violin plot
    ax1 = axes[0]
    positions = range(len(model_stats))
    violin_data = [s["scores"] for s in model_stats]

    parts = ax1.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.7)

    ax1.set_xticks(positions)
    ax1.set_xticklabels([s["name"] for s in model_stats], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"Score Distribution - {title}")
    if ylim:
        ax1.set_ylim(ylim)
    ax1.grid(axis="y", alpha=0.3)

    for i, s in enumerate(model_stats):
        ypos = ylim[1] - 0.05 * (ylim[1] - ylim[0]) if ylim else s["scores"].max() + 0.2
        ax1.annotate(
            f"μ={s['mean']:.2f}\nσ={s['std']:.2f}",
            xy=(i, ypos),
            ha="center",
            va="top",
            fontsize=7,
        )

    # Right: Variance bar chart
    ax2 = axes[1]
    variances = [s["var"] for s in model_stats]
    bars = ax2.bar(positions, variances, color="steelblue", alpha=0.7)

    ax2.set_xticks(positions)
    ax2.set_xticklabels([s["name"] for s in model_stats], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Variance")
    ax2.set_title(f"Rating Variance - {title}")
    ax2.grid(axis="y", alpha=0.3)

    for bar, s in zip(bars, model_stats):
        ax2.annotate(
            f"n={s['n']}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_score_histograms(
    scores_by_group: dict[str, list[float]],
    output_path: Path,
    title: str,
    bins: list[float] | None = None,
) -> None:
    """Plot histograms of score distributions."""
    n_groups = len(scores_by_group)
    n_cols = min(3, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_groups == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    sorted_groups = sorted(scores_by_group.items(), key=lambda x: np.var(x[1]), reverse=True)

    for i, (name, scores) in enumerate(sorted_groups):
        ax = axes[i]
        arr = np.array(scores)

        if bins is not None:
            ax.hist(arr, bins=bins, color="steelblue", alpha=0.7, edgecolor="black")
        else:
            ax.hist(arr, bins=20, color="steelblue", alpha=0.7, edgecolor="black")

        ax.set_title(f"{name}\nμ={arr.mean():.2f}, σ={arr.std():.2f}", fontsize=9)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")

    for i in range(len(sorted_groups), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Score Distributions - {title}", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rating variance analysis")
    parser.add_argument("--experiment-id", type=str, required=True, help="Experiment ID to analyze")
    parser.add_argument("--template", type=str, default=None, help="Filter to templates containing this string")
    parser.add_argument("--normalize", action="store_true", help="Normalize scores to 0-1 range")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    experiment_dir = EXPERIMENTS_DIR / args.experiment_id
    if not experiment_dir.exists():
        print(f"Experiment not found: {experiment_dir}")
        return

    output_dir = args.output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading measurements from {args.experiment_id}...")
    scores_by_model, scores_by_template = load_all_measurements(
        experiment_dir, args.template, args.normalize
    )

    if not scores_by_model:
        print("No measurements found")
        return

    print(f"\nFound {len(scores_by_model)} models, {len(scores_by_template)} templates:")
    for model, scores in sorted(scores_by_model.items()):
        arr = np.array(scores)
        print(f"  {model}: n={len(scores)}, mean={arr.mean():.2f}, std={arr.std():.2f}, var={arr.var():.3f}")

    date_str = datetime.now().strftime("%m%d%y")
    safe_exp_id = args.experiment_id.replace("/", "_")
    template_suffix = f"_{args.template}" if args.template else ""
    norm_suffix = "_normalized" if args.normalize else ""

    # Plot by model
    plot_variance_comparison(
        scores_by_model,
        output_dir / f"plot_{date_str}_{safe_exp_id}_variance_by_model{template_suffix}{norm_suffix}.png",
        title="By Model",
        ylabel="Score (normalized)" if args.normalize else "Score",
        ylim=(0, 1) if args.normalize else None,
    )

    plot_score_histograms(
        scores_by_model,
        output_dir / f"plot_{date_str}_{safe_exp_id}_histograms_by_model{template_suffix}{norm_suffix}.png",
        title="By Model",
    )

    # Plot by template
    if len(scores_by_template) > 1:
        plot_variance_comparison(
            scores_by_template,
            output_dir / f"plot_{date_str}_{safe_exp_id}_variance_by_template{norm_suffix}.png",
            title="By Template",
            ylabel="Score (normalized)" if args.normalize else "Score",
            ylim=(0, 1) if args.normalize else None,
        )

        plot_score_histograms(
            scores_by_template,
            output_dir / f"plot_{date_str}_{safe_exp_id}_histograms_by_template{norm_suffix}.png",
            title="By Template",
        )


if __name__ == "__main__":
    main()
