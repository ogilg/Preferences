"""Compute correlations between mu values across different templates.

Usage:
    python -m src.analysis.active_learning.correlate_templates --experiment-id gemma3_al_v3
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.measurement.storage import EXPERIMENTS_DIR

PLOTS_DIR = Path(__file__).parent / "plots"


def load_thurstonian_results(csv_path: Path) -> dict[str, dict]:
    """Load mu and sigma for each task."""
    results = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["task_id"]] = {
                "mu": float(row["mu"]),
                "sigma": float(row["sigma"]),
            }
    return results


def find_all_runs(experiment_dir: Path) -> dict[str, Path]:
    """Find all thurstonian CSVs in the experiment, keyed by run name."""
    al_dir = experiment_dir / "post_task_active_learning"
    if not al_dir.exists():
        return {}

    runs = {}
    for run_dir in al_dir.iterdir():
        if not run_dir.is_dir():
            continue
        for f in run_dir.iterdir():
            if f.name.startswith("thurstonian_") and f.suffix == ".csv":
                # Extract template name (first part before _gemma or _model)
                run_name = run_dir.name.split("_gemma")[0].split("_llama")[0].split("_qwen")[0]
                runs[run_name] = f
                break
    return runs


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted Pearson correlation."""
    w = weights / weights.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    sx = np.sqrt(np.sum(w * (x - mx) ** 2))
    sy = np.sqrt(np.sum(w * (y - my) ** 2))
    return cov / (sx * sy)


def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    run_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    """Plot correlation matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(run_names)))
    ax.set_yticks(range(len(run_names)))
    ax.set_xticklabels(run_names, rotation=45, ha="right")
    ax.set_yticklabels(run_names)

    # Add correlation values as text
    for i in range(len(run_names)):
        for j in range(len(run_names)):
            val = corr_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=12)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson r")

    ax.set_title(title)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved heatmap to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Correlate mu values across templates")
    parser.add_argument("--experiment-id", type=str, required=True)
    args = parser.parse_args()

    experiment_dir = EXPERIMENTS_DIR / args.experiment_id
    if not experiment_dir.exists():
        raise ValueError(f"Experiment not found: {experiment_dir}")

    runs = find_all_runs(experiment_dir)
    if len(runs) < 2:
        raise ValueError(f"Need at least 2 runs to correlate, found {len(runs)}")

    print(f"Found {len(runs)} runs: {list(runs.keys())}")

    # Load all results
    all_results = {name: load_thurstonian_results(path) for name, path in runs.items()}

    # Find common task IDs
    common_ids = set.intersection(*[set(r.keys()) for r in all_results.values()])
    print(f"Common tasks across all runs: {len(common_ids)}")

    run_names = sorted(runs.keys())

    # Print correlation matrix
    print("\n" + "=" * 70)
    print("PEARSON CORRELATION (unweighted)")
    print("=" * 70)
    header = f"{'':>20}" + "".join(f"{name:>18}" for name in run_names)
    print(header)

    for name1 in run_names:
        row = f"{name1:>20}"
        for name2 in run_names:
            if name1 == name2:
                row += f"{'1.000':>18}"
            else:
                mus1 = np.array([all_results[name1][tid]["mu"] for tid in common_ids])
                mus2 = np.array([all_results[name2][tid]["mu"] for tid in common_ids])
                r, p = pearsonr(mus1, mus2)
                row += f"{r:>11.3f} (p={p:.2g})"[:18].rjust(18)
        print(row)

    print("\n" + "=" * 70)
    print("SPEARMAN CORRELATION (rank-based)")
    print("=" * 70)
    print(header)

    for name1 in run_names:
        row = f"{name1:>20}"
        for name2 in run_names:
            if name1 == name2:
                row += f"{'1.000':>18}"
            else:
                mus1 = np.array([all_results[name1][tid]["mu"] for tid in common_ids])
                mus2 = np.array([all_results[name2][tid]["mu"] for tid in common_ids])
                r, p = spearmanr(mus1, mus2)
                row += f"{r:>11.3f} (p={p:.2g})"[:18].rjust(18)
        print(row)

    # Weighted correlation using inverse variance (1/sigma^2)
    print("\n" + "=" * 70)
    print("WEIGHTED PEARSON (weights = 1/(sigma1^2 + sigma2^2))")
    print("=" * 70)
    print(header)

    for name1 in run_names:
        row = f"{name1:>20}"
        for name2 in run_names:
            if name1 == name2:
                row += f"{'1.000':>18}"
            else:
                mus1 = np.array([all_results[name1][tid]["mu"] for tid in common_ids])
                mus2 = np.array([all_results[name2][tid]["mu"] for tid in common_ids])
                sig1 = np.array([all_results[name1][tid]["sigma"] for tid in common_ids])
                sig2 = np.array([all_results[name2][tid]["sigma"] for tid in common_ids])
                # Weight by inverse combined variance
                weights = 1.0 / (sig1**2 + sig2**2)
                r = weighted_correlation(mus1, mus2, weights)
                row += f"{r:>18.3f}"
        print(row)

    print("=" * 70)

    # Pairwise detailed stats
    print("\n" + "=" * 70)
    print("DETAILED PAIRWISE COMPARISONS")
    print("=" * 70)

    for i, name1 in enumerate(run_names):
        for name2 in run_names[i + 1:]:
            mus1 = np.array([all_results[name1][tid]["mu"] for tid in common_ids])
            mus2 = np.array([all_results[name2][tid]["mu"] for tid in common_ids])
            sig1 = np.array([all_results[name1][tid]["sigma"] for tid in common_ids])
            sig2 = np.array([all_results[name2][tid]["sigma"] for tid in common_ids])

            r_pearson, p_pearson = pearsonr(mus1, mus2)
            r_spearman, p_spearman = spearmanr(mus1, mus2)
            weights = 1.0 / (sig1**2 + sig2**2)
            r_weighted = weighted_correlation(mus1, mus2, weights)

            print(f"\n{name1} vs {name2}:")
            print(f"  Pearson:  r={r_pearson:.3f}, p={p_pearson:.2g}")
            print(f"  Spearman: r={r_spearman:.3f}, p={p_spearman:.2g}")
            print(f"  Weighted: r={r_weighted:.3f}")
            print(f"  Mean μ ({name1}): {mus1.mean():+.3f} ± {mus1.std():.3f}")
            print(f"  Mean μ ({name2}): {mus2.mean():+.3f} ± {mus2.std():.3f}")

    # Build and plot correlation matrices
    n = len(run_names)
    pearson_matrix = np.eye(n)
    weighted_matrix = np.eye(n)

    for i, name1 in enumerate(run_names):
        for j, name2 in enumerate(run_names):
            if i != j:
                mus1 = np.array([all_results[name1][tid]["mu"] for tid in common_ids])
                mus2 = np.array([all_results[name2][tid]["mu"] for tid in common_ids])
                sig1 = np.array([all_results[name1][tid]["sigma"] for tid in common_ids])
                sig2 = np.array([all_results[name2][tid]["sigma"] for tid in common_ids])

                pearson_matrix[i, j], _ = pearsonr(mus1, mus2)
                weights = 1.0 / (sig1**2 + sig2**2)
                weighted_matrix[i, j] = weighted_correlation(mus1, mus2, weights)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%m%d%y")

    plot_correlation_heatmap(
        pearson_matrix,
        run_names,
        PLOTS_DIR / f"plot_{date_str}_template_correlation_pearson_{args.experiment_id}.png",
        f"Template Correlation (Pearson)\n{args.experiment_id}",
    )

    plot_correlation_heatmap(
        weighted_matrix,
        run_names,
        PLOTS_DIR / f"plot_{date_str}_template_correlation_weighted_{args.experiment_id}.png",
        f"Template Correlation (Weighted by 1/σ²)\n{args.experiment_id}",
    )


if __name__ == "__main__":
    main()
