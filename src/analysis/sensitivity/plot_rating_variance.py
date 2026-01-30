"""Plot rating variance and distribution across models from multi_model_seed_sensitivity data."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

RESULTS_DIR = Path("results/experiments/multi_model_seed_sensitivity/post_task_stated")
OUTPUT_DIR = Path("src/analysis/sensitivity/plots")


def load_all_measurements() -> dict[str, list[float]]:
    """Load all measurements, grouped by model name.

    Returns dict: model_name -> list of all scores across all runs.
    """
    scores_by_model: dict[str, list[float]] = defaultdict(list)

    for run_dir in RESULTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        measurements_path = run_dir / "measurements.yaml"
        if not measurements_path.exists():
            continue

        # Parse model name from directory name
        # Format: {template}_{model}_regex_cseed{N}_rseed{N}
        match = re.match(r"anchored(?:_precise)?_1_5_(.+?)_regex_cseed\d+_rseed\d+", run_dir.name)
        if not match:
            continue

        model_name = match.group(1)

        with open(measurements_path) as f:
            measurements = yaml.safe_load(f)

        scores = [m["score"] for m in measurements if "score" in m]
        scores_by_model[model_name].extend(scores)

    return dict(scores_by_model)


def plot_variance_comparison(scores_by_model: dict[str, list[float]], output_path: Path) -> None:
    """Plot variance comparison as bar chart with violin overlays."""
    # Sort models by variance (descending)
    model_stats = []
    for model, scores in scores_by_model.items():
        arr = np.array(scores)
        model_stats.append({
            "model": model,
            "scores": arr,
            "mean": arr.mean(),
            "std": arr.std(),
            "var": arr.var(),
            "n": len(arr),
        })

    model_stats.sort(key=lambda x: x["var"], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Violin plot of score distributions
    ax1 = axes[0]
    positions = range(len(model_stats))
    violin_data = [s["scores"] for s in model_stats]

    parts = ax1.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)

    # Color the violins
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.7)

    ax1.set_xticks(positions)
    ax1.set_xticklabels([s["model"] for s in model_stats], rotation=45, ha="right")
    ax1.set_ylabel("Score (1-5)")
    ax1.set_title("Score Distribution by Model")
    ax1.set_ylim(0.5, 5.5)
    ax1.axhline(y=3, color="gray", linestyle="--", alpha=0.5, label="Neutral (3)")
    ax1.grid(axis="y", alpha=0.3)

    # Add mean and std annotations
    for i, s in enumerate(model_stats):
        ax1.annotate(
            f"μ={s['mean']:.2f}\nσ={s['std']:.2f}",
            xy=(i, 5.3),
            ha="center",
            va="top",
            fontsize=8,
        )

    # Right: Bar chart of variance
    ax2 = axes[1]
    variances = [s["var"] for s in model_stats]
    bars = ax2.bar(positions, variances, color="steelblue", alpha=0.7)

    ax2.set_xticks(positions)
    ax2.set_xticklabels([s["model"] for s in model_stats], rotation=45, ha="right")
    ax2.set_ylabel("Variance")
    ax2.set_title("Rating Variance by Model")
    ax2.grid(axis="y", alpha=0.3)

    # Add sample size annotations
    for i, (bar, s) in enumerate(zip(bars, model_stats)):
        ax2.annotate(
            f"n={s['n']}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_score_histograms(scores_by_model: dict[str, list[float]], output_path: Path) -> None:
    """Plot histograms of score distributions for each model."""
    n_models = len(scores_by_model)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    # Sort by variance
    sorted_models = sorted(
        scores_by_model.items(),
        key=lambda x: np.var(x[1]),
        reverse=True,
    )

    for i, (model, scores) in enumerate(sorted_models):
        ax = axes[i]
        arr = np.array(scores)

        # Count discrete values
        counts = {v: 0 for v in [1, 2, 3, 4, 5]}
        for s in scores:
            rounded = round(s)
            if rounded in counts:
                counts[rounded] += 1

        bars = ax.bar(counts.keys(), counts.values(), color="steelblue", alpha=0.7, edgecolor="black")

        ax.set_title(f"{model}\nμ={arr.mean():.2f}, σ={arr.std():.2f}, var={arr.var():.3f}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_xticks([1, 2, 3, 4, 5])

        # Add percentage labels
        total = len(scores)
        for bar, (score, count) in zip(bars, counts.items()):
            if count > 0:
                pct = 100 * count / total
                ax.annotate(
                    f"{pct:.0f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Hide unused subplots
    for i in range(len(sorted_models), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Score Distribution Histograms by Model", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading measurements...")
    scores_by_model = load_all_measurements()

    print(f"\nFound {len(scores_by_model)} models:")
    for model, scores in sorted(scores_by_model.items()):
        arr = np.array(scores)
        print(f"  {model}: n={len(scores)}, mean={arr.mean():.2f}, std={arr.std():.2f}, var={arr.var():.3f}")

    date_str = datetime.now().strftime("%m%d%y")

    # Plot variance comparison with violins
    plot_variance_comparison(
        scores_by_model,
        OUTPUT_DIR / f"plot_{date_str}_rating_variance_by_model.png",
    )

    # Plot histograms
    plot_score_histograms(
        scores_by_model,
        OUTPUT_DIR / f"plot_{date_str}_score_histograms_by_model.png",
    )


if __name__ == "__main__":
    main()
