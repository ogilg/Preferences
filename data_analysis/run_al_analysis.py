#!/usr/bin/env python
"""Run the active learning vs full MLE comparison on synthetic and real data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from data_analysis import (
    run_synthetic_comparison,
    run_real_data_comparison,
    plot_utility_scatter,
    plot_convergence_curve,
    plot_held_out_comparison,
)
from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, PreferenceType

OUTPUT_DIR = Path("data_analysis/plots/active_learning")


def load_real_data(run_dir: Path) -> tuple[list[BinaryPreferenceMeasurement], list[Task]]:
    """Load comparison data from a results directory."""
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    with open(run_dir / "measurements.yaml") as f:
        measurements_raw = yaml.safe_load(f)

    # Build tasks from config
    task_ids = config["task_ids"]
    task_prompts = config["task_prompts"]
    tasks = [
        Task(
            prompt=task_prompts[tid],
            origin=OriginDataset.WILDCHAT,
            id=tid,
            metadata={},
        )
        for tid in task_ids
    ]
    id_to_task = {t.id: t for t in tasks}

    # Convert measurements to BinaryPreferenceMeasurement objects
    comparisons = [
        BinaryPreferenceMeasurement(
            task_a=id_to_task[m["task_a"]],
            task_b=id_to_task[m["task_b"]],
            choice=m["choice"],
            preference_type=PreferenceType.PRE_TASK_STATED,
        )
        for m in measurements_raw
    ]

    return comparisons, tasks


def run_synthetic_analysis():
    """Run analysis on synthetic data."""
    print("=" * 60)
    print("SYNTHETIC DATA ANALYSIS")
    print("=" * 60)
    print()

    result = run_synthetic_comparison(
        n_tasks=50,
        held_out_fraction=0.2,
        n_comparisons_per_pair=5,
        initial_degree=3,
        batch_size=50,
        max_iterations=20,
        seed=42,
    )

    print(f"Tasks: {len(result.scenario.tasks)}")
    print(f"Total training pairs: {result.al_vs_full_mle.total_pairs}")
    print()
    print("FULL MLE (baseline):")
    print(f"  Uses all {result.al_vs_full_mle.total_pairs} training pairs")
    print(f"  Held-out accuracy: {result.full_mle_held_out_accuracy:.1%}")
    print()
    print("ACTIVE LEARNING:")
    print(f"  Uses {result.al_vs_full_mle.pairs_queried} pairs ({result.al_vs_full_mle.efficiency:.1%} of total)")
    print(f"  Held-out accuracy: {result.al_vs_full_mle.held_out_accuracy:.1%}")
    print(f"  Spearman with Full MLE: {result.al_vs_full_mle.spearman_rho:.3f}")

    # Plots
    n_tasks = len(result.scenario.tasks)

    fig1 = plot_utility_scatter(
        result.al_final_result,
        result.full_mle_result,
        result.scenario.true_mu,
        n_tasks=n_tasks,
        figsize=(12, 5),
    )
    fig1.savefig(OUTPUT_DIR / "synthetic_01_utility_scatter.png", dpi=150, bbox_inches="tight")

    fig2 = plot_convergence_curve(
        result.trajectory,
        full_mle_accuracy=result.full_mle_held_out_accuracy,
        n_tasks=n_tasks,
        figsize=(14, 4),
    )
    fig2.savefig(OUTPUT_DIR / "synthetic_02_convergence.png", dpi=150, bbox_inches="tight")

    fig3 = plot_held_out_comparison(
        result.al_vs_full_mle.held_out_accuracy,
        result.full_mle_held_out_accuracy,
        result.al_vs_full_mle.pairs_queried,
        result.al_vs_full_mle.total_pairs,
        n_tasks=n_tasks,
        figsize=(10, 4),
    )
    fig3.savefig(OUTPUT_DIR / "synthetic_03_efficiency.png", dpi=150, bbox_inches="tight")

    print()
    print("Saved: synthetic_01_utility_scatter.png")
    print("Saved: synthetic_02_convergence.png")
    print("Saved: synthetic_03_efficiency.png")

    return result


def run_real_analysis(run_dir: Path, label: str = "real"):
    """Run analysis on real data from a results directory."""
    print()
    print("=" * 60)
    print(f"REAL DATA ANALYSIS: {run_dir.name}")
    print("=" * 60)
    print()

    comparisons, tasks = load_real_data(run_dir)
    print(f"Loaded {len(comparisons)} comparisons for {len(tasks)} tasks")

    result = run_real_data_comparison(
        comparisons=comparisons,
        tasks=tasks,
        held_out_fraction=0.2,
        initial_degree=5,
        batch_size=10,
        max_iterations=50,
        seed=42,
    )

    print()
    print(f"Total training pairs: {result.al_vs_full_mle.total_pairs}")
    print()
    print("FULL MLE (baseline):")
    print(f"  Uses all {result.al_vs_full_mle.total_pairs} training pairs")
    print(f"  Held-out accuracy: {result.full_mle_held_out_accuracy:.1%}")
    print()
    print("ACTIVE LEARNING:")
    print(f"  Uses {result.al_vs_full_mle.pairs_queried} pairs ({result.al_vs_full_mle.efficiency:.1%} of total)")
    print(f"  Held-out accuracy: {result.al_vs_full_mle.held_out_accuracy:.1%}")
    print(f"  Spearman with Full MLE: {result.al_vs_full_mle.spearman_rho:.3f}")

    # Plots (no true_mu for real data)
    n_tasks = len(tasks)

    fig1 = plot_utility_scatter(
        result.al_final_result,
        result.full_mle_result,
        true_mu=None,
        n_tasks=n_tasks,
        figsize=(7, 5),
    )
    fig1.savefig(OUTPUT_DIR / f"{label}_01_utility_scatter.png", dpi=150, bbox_inches="tight")

    fig2 = plot_convergence_curve(
        result.trajectory,
        full_mle_accuracy=result.full_mle_held_out_accuracy,
        n_tasks=n_tasks,
        figsize=(10, 4),
    )
    fig2.savefig(OUTPUT_DIR / f"{label}_02_convergence.png", dpi=150, bbox_inches="tight")

    fig3 = plot_held_out_comparison(
        result.al_vs_full_mle.held_out_accuracy,
        result.full_mle_held_out_accuracy,
        result.al_vs_full_mle.pairs_queried,
        result.al_vs_full_mle.total_pairs,
        n_tasks=n_tasks,
        figsize=(10, 4),
    )
    fig3.savefig(OUTPUT_DIR / f"{label}_03_efficiency.png", dpi=150, bbox_inches="tight")

    print()
    print(f"Saved: {label}_01_utility_scatter.png")
    print(f"Saved: {label}_02_convergence.png")
    print(f"Saved: {label}_03_efficiency.png")

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Synthetic data
    run_synthetic_analysis()

    # Real data from n50 experiment
    real_run_dir = Path("results/binary/n50/001_llama-3.1-8b")
    run_real_analysis(real_run_dir, label="real")

    plt.close("all")
    print()
    print("=" * 60)
    print(f"DONE - All plots saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
