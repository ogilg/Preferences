"""Analyze seed sensitivity of post-task qualitative measurements.

Compares scores across different rating seeds for the same template to assess
measurement reliability (test-retest).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.analysis.correlation.utils import compute_pairwise_correlations


def load_scores_by_seed(
    experiment_dir: Path,
) -> dict[int, tuple[np.ndarray, list[str]]]:
    """Load qualitative scores grouped by rating seed.

    Returns:
        {seed: (values, task_ids)} - format for compute_pairwise_correlations
    """
    results: dict[int, tuple[np.ndarray, list[str]]] = {}
    stated_dir = experiment_dir / "post_task_stated"

    if not stated_dir.exists():
        return results

    for run_dir in sorted(stated_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # Parse seed from dirname: post_task_qualitative_001_llama-3.1-8b_regex_cseed0_rseed0
        name = run_dir.name
        if "_rseed" not in name:
            continue
        seed = int(name.split("_rseed")[1])

        measurements_path = run_dir / "measurements.yaml"
        if not measurements_path.exists():
            continue

        with open(measurements_path) as f:
            data = yaml.safe_load(f)

        if not data:
            continue

        task_ids = [m["task_id"] for m in data]
        values = np.array([m["score"] for m in data])
        results[seed] = (values, task_ids)

    return results


def plot_seed_sensitivity(
    experiment_dir: Path,
    output_path: Path | None = None,
) -> None:
    """Create visualization of seed sensitivity for qualitative measurements."""
    results_by_seed = load_scores_by_seed(experiment_dir)

    if len(results_by_seed) < 2:
        print(f"Need at least 2 seeds, found {len(results_by_seed)}")
        return

    # Convert to format for compute_pairwise_correlations
    results = {str(seed): data for seed, data in results_by_seed.items()}
    correlations = compute_pairwise_correlations(results)

    seeds = sorted(results_by_seed.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Scatter of first two seeds
    ax1 = axes[0]
    if len(seeds) >= 2:
        vals1, ids1 = results_by_seed[seeds[0]]
        vals2, ids2 = results_by_seed[seeds[1]]

        common = set(ids1) & set(ids2)
        if common:
            idx1 = {tid: i for i, tid in enumerate(ids1)}
            idx2 = {tid: i for i, tid in enumerate(ids2)}
            arr1 = np.array([vals1[idx1[tid]] for tid in sorted(common)])
            arr2 = np.array([vals2[idx2[tid]] for tid in sorted(common)])

            ax1.scatter(arr1, arr2, alpha=0.7, s=50, edgecolor="white", linewidth=0.5)

            if len(arr1) > 1 and np.std(arr1) > 1e-10:
                z = np.polyfit(arr1, arr2, 1)
                p = np.poly1d(z)
                x_line = np.linspace(arr1.min(), arr1.max(), 100)
                ax1.plot(x_line, p(x_line), "r--", alpha=0.6, linewidth=1.5)

            lims = [min(arr1.min(), arr2.min()) - 0.1, max(arr1.max(), arr2.max()) + 0.1]
            ax1.plot(lims, lims, "k:", alpha=0.3, linewidth=1)

            corr = np.corrcoef(arr1, arr2)[0, 1] if np.std(arr1) > 1e-10 and np.std(arr2) > 1e-10 else 0
            ax1.set_xlabel(f"Seed {seeds[0]}", fontsize=10)
            ax1.set_ylabel(f"Seed {seeds[1]}", fontsize=10)
            ax1.set_title(f"Cross-Seed: r = {corr:.3f} (n={len(common)})", fontsize=11)

    # Panel 2: Correlation histogram
    ax2 = axes[1]
    corr_vals = [c["correlation"] for c in correlations if not np.isnan(c["correlation"])]
    if corr_vals:
        ax2.hist(corr_vals, bins=min(10, len(corr_vals)), edgecolor="white", alpha=0.8, color="steelblue")
        ax2.axvline(np.mean(corr_vals), color="red", linestyle="--", label=f"Mean: {np.mean(corr_vals):.3f}")
        ax2.set_xlabel("Correlation", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.set_title("Pairwise Seed Correlations", fontsize=11)
        ax2.legend(fontsize=9)

    # Panel 3: Distribution by seed
    ax3 = axes[2]
    data = [results_by_seed[seed][0] for seed in seeds]
    positions = list(range(len(seeds)))

    parts = ax3.violinplot(data, positions, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)

    ax3.set_xticks(positions)
    ax3.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=10)
    ax3.set_ylabel("Score", fontsize=10)
    ax3.set_title("Score Distribution by Seed", fontsize=11)
    ax3.axhline(0, color="k", linestyle=":", alpha=0.3)

    for i, seed in enumerate(seeds):
        vals = results_by_seed[seed][0]
        ax3.text(i, ax3.get_ylim()[1] - 0.1, f"μ={np.mean(vals):.2f}\nn={len(vals)}",
                 ha="center", va="top", fontsize=8)

    experiment_name = experiment_dir.name
    fig.suptitle(f"Post-Task Qualitative Seed Sensitivity: {experiment_name}", fontsize=12, y=1.02)
    plt.tight_layout()

    if output_path is None:
        date_str = datetime.now().strftime("%m%d%y")
        output_path = Path("src/analysis/correlation/plots") / f"plot_{date_str}_qualitative_seed_sensitivity.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

    # Print summary
    print(f"\n=== Seed Sensitivity Summary ===")
    print(f"Seeds: {seeds}")
    if corr_vals:
        print(f"\nPairwise correlations (n={len(corr_vals)} pairs):")
        print(f"  Mean: {np.mean(corr_vals):.3f}")
        print(f"  Min: {np.min(corr_vals):.3f}")
        print(f"  Max: {np.max(corr_vals):.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze qualitative seed sensitivity")
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to experiment results directory",
    )
    parser.add_argument("-o", "--output", type=Path)

    args = parser.parse_args()
    plot_seed_sensitivity(args.experiment_dir, args.output)
