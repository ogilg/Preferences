"""Analyze seed sensitivity of open-ended valence measurements.

Compares scores across different rating seeds to assess measurement reliability.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_open_ended_results(experiment_dir: Path) -> dict[int, dict[str, list[float]]]:
    """Load open-ended results grouped by seed and task.

    Returns:
        {seed: {task_id: [scores]}}
    """
    results_by_seed: dict[int, dict[str, list[float]]] = {}

    for filepath in experiment_dir.glob("open_ended_*.json"):
        # Parse seed from filename: open_ended_experience_reflection_rseed0.json
        name = filepath.stem
        if "_rseed" not in name:
            continue
        seed = int(name.split("_rseed")[1])

        with open(filepath) as f:
            data = json.load(f)

        by_task: dict[str, list[float]] = defaultdict(list)
        for record in data:
            by_task[record["task_id"]].append(record["semantic_valence_score"])

        results_by_seed[seed] = dict(by_task)

    return results_by_seed


def compute_seed_statistics(results_by_seed: dict[int, dict[str, list[float]]]) -> dict:
    """Compute statistics comparing scores across seeds."""
    seeds = sorted(results_by_seed.keys())
    if len(seeds) < 2:
        raise ValueError(f"Need at least 2 seeds, got {len(seeds)}")

    # Get task means per seed
    task_means: dict[int, dict[str, float]] = {}
    for seed, by_task in results_by_seed.items():
        task_means[seed] = {task_id: np.mean(scores) for task_id, scores in by_task.items()}

    # Find common tasks across all seeds
    common_tasks = set.intersection(*[set(tm.keys()) for tm in task_means.values()])

    # Build arrays for correlation
    seed_arrays = {
        seed: np.array([task_means[seed][t] for t in sorted(common_tasks)])
        for seed in seeds
    }

    # Pairwise correlations
    correlations = {}
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1:]:
            arr_a, arr_b = seed_arrays[seed_a], seed_arrays[seed_b]
            corr = np.corrcoef(arr_a, arr_b)[0, 1]
            correlations[(seed_a, seed_b)] = corr

    # Within-task variance (across samples within same seed)
    within_variances = []
    for seed, by_task in results_by_seed.items():
        for task_id, scores in by_task.items():
            if len(scores) > 1:
                within_variances.append(np.var(scores))

    # Between-seed variance (for same task, different seeds)
    between_variances = []
    for task_id in common_tasks:
        task_seed_means = [task_means[seed][task_id] for seed in seeds]
        if len(task_seed_means) > 1:
            between_variances.append(np.var(task_seed_means))

    return {
        "seeds": seeds,
        "n_common_tasks": len(common_tasks),
        "common_tasks": sorted(common_tasks),
        "seed_arrays": seed_arrays,
        "task_means": task_means,
        "correlations": correlations,
        "mean_within_variance": np.mean(within_variances) if within_variances else None,
        "mean_between_seed_variance": np.mean(between_variances) if between_variances else None,
    }


def plot_seed_sensitivity(
    experiment_dir: Path,
    output_path: Path | None = None,
) -> None:
    """Create visualization of seed sensitivity for open-ended measurements."""
    results_by_seed = load_open_ended_results(experiment_dir)
    stats = compute_seed_statistics(results_by_seed)

    seeds = stats["seeds"]
    seed_arrays = stats["seed_arrays"]
    task_means = stats["task_means"]
    common_tasks = stats["common_tasks"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Scatter plot of seed means
    ax1 = axes[0]
    if len(seeds) >= 2:
        arr_0, arr_1 = seed_arrays[seeds[0]], seed_arrays[seeds[1]]
        ax1.scatter(arr_0, arr_1, alpha=0.7, s=50, edgecolor="white", linewidth=0.5)

        # Fit line
        if len(arr_0) > 1 and np.std(arr_0) > 1e-10:
            z = np.polyfit(arr_0, arr_1, 1)
            p = np.poly1d(z)
            x_line = np.linspace(arr_0.min(), arr_0.max(), 100)
            ax1.plot(x_line, p(x_line), "r--", alpha=0.6, linewidth=1.5)

        # Identity line
        lims = [min(arr_0.min(), arr_1.min()) - 0.1, max(arr_0.max(), arr_1.max()) + 0.1]
        ax1.plot(lims, lims, "k:", alpha=0.3, linewidth=1)

        corr = stats["correlations"].get((seeds[0], seeds[1]), np.nan)
        ax1.set_xlabel(f"Seed {seeds[0]} (task mean)", fontsize=10)
        ax1.set_ylabel(f"Seed {seeds[1]} (task mean)", fontsize=10)
        ax1.set_title(f"Cross-Seed Correlation: r = {corr:.3f}", fontsize=11)

    # Panel 2: Per-task variance breakdown
    ax2 = axes[1]

    within_vars = []
    between_vars = []
    task_labels = []

    for task_id in common_tasks:
        # Within-seed variance (average across seeds)
        task_within = []
        for seed in seeds:
            if task_id in results_by_seed[seed]:
                scores = results_by_seed[seed][task_id]
                if len(scores) > 1:
                    task_within.append(np.var(scores))

        # Between-seed variance
        seed_means = [task_means[seed][task_id] for seed in seeds if task_id in task_means[seed]]

        if task_within and len(seed_means) > 1:
            within_vars.append(np.mean(task_within))
            between_vars.append(np.var(seed_means))
            task_labels.append(task_id.replace("wildchat_", ""))

    if within_vars and between_vars:
        x = np.arange(len(task_labels))
        width = 0.35

        ax2.bar(x - width/2, within_vars, width, label="Within-seed var", color="steelblue", alpha=0.8)
        ax2.bar(x + width/2, between_vars, width, label="Between-seed var", color="coral", alpha=0.8)

        ax2.set_xticks(x)
        ax2.set_xticklabels(task_labels, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Variance", fontsize=10)
        ax2.set_title("Score Variance by Task", fontsize=11)
        ax2.legend(fontsize=9)

    # Panel 3: Distribution of scores by seed
    ax3 = axes[2]

    all_scores_by_seed = {seed: [] for seed in seeds}
    for seed, by_task in results_by_seed.items():
        for scores in by_task.values():
            all_scores_by_seed[seed].extend(scores)

    positions = list(range(len(seeds)))
    data = [all_scores_by_seed[seed] for seed in seeds]

    parts = ax3.violinplot(data, positions, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)

    ax3.set_xticks(positions)
    ax3.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=10)
    ax3.set_ylabel("Valence Score", fontsize=10)
    ax3.set_title("Score Distribution by Seed", fontsize=11)
    ax3.axhline(0, color="k", linestyle=":", alpha=0.3)

    # Add summary stats
    for i, seed in enumerate(seeds):
        scores = all_scores_by_seed[seed]
        ax3.text(i, ax3.get_ylim()[1] - 0.1, f"μ={np.mean(scores):.2f}\nn={len(scores)}",
                ha="center", va="top", fontsize=8)

    # Overall title
    experiment_name = experiment_dir.name
    fig.suptitle(f"Open-Ended Valence Seed Sensitivity: {experiment_name}", fontsize=12, y=1.02)

    plt.tight_layout()

    if output_path is None:
        date_str = datetime.now().strftime("%m%d%y")
        output_path = Path("src/analysis/correlation") / f"plot_{date_str}_open_ended_seed_sensitivity.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

    # Print summary
    print(f"\n=== Seed Sensitivity Summary ===")
    print(f"Seeds: {seeds}")
    print(f"Common tasks: {len(common_tasks)}")
    for (s_a, s_b), corr in stats["correlations"].items():
        print(f"Correlation seed {s_a} vs {s_b}: {corr:.3f}")
    if stats["mean_within_variance"] is not None:
        print(f"Mean within-seed variance: {stats['mean_within_variance']:.4f}")
    if stats["mean_between_seed_variance"] is not None:
        print(f"Mean between-seed variance: {stats['mean_between_seed_variance']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze open-ended seed sensitivity")
    parser.add_argument("experiment_dir", type=Path, help="Path to experiment results directory")
    parser.add_argument("-o", "--output", type=Path, help="Output path for plot")

    args = parser.parse_args()
    plot_seed_sensitivity(args.experiment_dir, args.output)
