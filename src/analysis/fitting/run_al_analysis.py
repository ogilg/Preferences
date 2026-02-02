#!/usr/bin/env python
"""Run the active learning vs full MLE comparison on synthetic and real data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.analysis.fitting import (
    run_synthetic_comparison,
    run_real_data_comparison,
    plot_utility_scatter,
    plot_convergence_curve,
    plot_held_out_comparison,
)
from src.analysis.fitting.config import N_TASKS, RESULTS_DIR
from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, PreferenceType

OUTPUT_DIR = Path("thurstonian_analysis/plots/active_learning")


def load_real_data(run_dir: Path) -> tuple[list[BinaryPreferenceMeasurement], list[Task]]:
    """Load comparison data from a results directory."""
    with open(run_dir / "measurements.yaml") as f:
        measurements_raw = yaml.load(f, Loader=yaml.CSafeLoader)

    if not measurements_raw:
        return [], []

    # Extract task IDs from measurements
    task_ids = set()
    for m in measurements_raw:
        task_ids.add(m["task_a"])
        task_ids.add(m["task_b"])

    # Build tasks (we don't have prompts, so use placeholder)
    tasks = [
        Task(
            prompt=f"Task {tid}",
            origin=OriginDataset.WILDCHAT,
            id=tid,
            metadata={},
        )
        for tid in sorted(task_ids)
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
        n_tasks=N_TASKS,
        held_out_fraction=0.2,
        n_comparisons_per_pair=5,
        initial_degree=3,
        batch_size=100,
        max_iterations=50,
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
        batch_size=100,
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


def get_all_real_data_dirs() -> list[Path]:
    """Get all measurement directories with sufficient data."""
    dirs = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "measurements.yaml").exists():
            continue
        dirs.append(d)
    return dirs


def run_real_analysis_aggregated(label: str = "real"):
    """Run analysis on real data, averaging over all prompt templates."""
    print()
    print("=" * 60)
    print("REAL DATA ANALYSIS (aggregated over prompt templates)")
    print("=" * 60)
    print()

    run_dirs = get_all_real_data_dirs()
    if not run_dirs:
        print("No measurement directories found!")
        return None

    print(f"Found {len(run_dirs)} measurement directories")

    all_results = []
    for run_dir in run_dirs:
        try:
            comparisons, tasks = load_real_data(run_dir)
            if len(tasks) < N_TASKS:
                print(f"  Skipping {run_dir.name}: only {len(tasks)} tasks (need {N_TASKS})")
                continue

            print(f"  Processing {run_dir.name} ({len(tasks)} tasks, {len(comparisons)} comparisons)")

            result = run_real_data_comparison(
                comparisons=comparisons,
                tasks=tasks,
                held_out_fraction=0.2,
                initial_degree=5,
                batch_size=100,
                max_iterations=50,
                seed=42,
            )
            all_results.append((run_dir.name, result))
        except Exception as e:
            print(f"  Error processing {run_dir.name}: {e}")

    if not all_results:
        print("No valid results!")
        return None

    print()
    print(f"Successfully processed {len(all_results)} templates")
    print()

    # Print summary statistics
    held_out_accs = [r.al_vs_full_mle.held_out_accuracy for _, r in all_results]
    full_mle_accs = [r.full_mle_held_out_accuracy for _, r in all_results]
    spearman_rhos = [r.al_vs_full_mle.spearman_rho for _, r in all_results]
    efficiencies = [r.al_vs_full_mle.efficiency for _, r in all_results]

    print("AGGREGATED RESULTS:")
    print(f"  Templates: {len(all_results)}")
    print()
    print("FULL MLE (baseline):")
    print(f"  Held-out accuracy: {np.mean(full_mle_accs):.1%} +/- {np.std(full_mle_accs):.1%}")
    print()
    print("ACTIVE LEARNING:")
    print(f"  Held-out accuracy: {np.mean(held_out_accs):.1%} +/- {np.std(held_out_accs):.1%}")
    print(f"  Spearman with Full MLE: {np.mean(spearman_rhos):.3f} +/- {np.std(spearman_rhos):.3f}")
    print(f"  Efficiency: {np.mean(efficiencies):.1%} +/- {np.std(efficiencies):.1%}")

    # Plot aggregated convergence curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, result in all_results:
        traj = result.trajectory
        axes[0].plot(traj.cumulative_pairs, traj.spearman_vs_full_mle, "o-", alpha=0.5, markersize=2, label=name)
        axes[1].plot(traj.cumulative_pairs, traj.held_out_accuracy, "o-", alpha=0.5, markersize=2, label=name)

    axes[0].set_xlabel("Pairs queried")
    axes[0].set_ylabel("Spearman rho vs Full MLE")
    axes[0].set_title(f"Convergence to Full MLE (N={N_TASKS}, {len(all_results)} templates)")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    axes[1].set_xlabel("Pairs queried")
    axes[1].set_ylabel("Held-out accuracy")
    axes[1].set_title("Held-out Prediction Accuracy")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].axhline(y=np.mean(full_mle_accs), color="red", linestyle="--", alpha=0.7, label=f"Full MLE mean: {np.mean(full_mle_accs):.3f}")
    axes[1].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{label}_convergence_all_templates.png", dpi=150, bbox_inches="tight")

    # Plot summary bar chart
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

    # Accuracy comparison
    x = np.arange(len(all_results))
    width = 0.35
    axes2[0].bar(x - width/2, [r.al_vs_full_mle.held_out_accuracy for _, r in all_results], width, label="Active Learning", alpha=0.8)
    axes2[0].bar(x + width/2, [r.full_mle_held_out_accuracy for _, r in all_results], width, label="Full MLE", alpha=0.8)
    axes2[0].set_ylabel("Held-out Accuracy")
    axes2[0].set_title(f"AL vs Full MLE by Template (N={N_TASKS})")
    axes2[0].set_xticks(x)
    axes2[0].set_xticklabels([name.replace("binary_choice_", "").replace("_llama-3.1-8b", "") for name, _ in all_results], rotation=45)
    axes2[0].legend()
    axes2[0].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    # Efficiency
    axes2[1].bar(x, [r.al_vs_full_mle.efficiency * 100 for _, r in all_results], alpha=0.8, color="C2")
    axes2[1].set_ylabel("Pairs used (%)")
    axes2[1].set_title("Active Learning Efficiency")
    axes2[1].set_xticks(x)
    axes2[1].set_xticklabels([name.replace("binary_choice_", "").replace("_llama-3.1-8b", "") for name, _ in all_results], rotation=45)

    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / f"{label}_summary_by_template.png", dpi=150, bbox_inches="tight")

    print()
    print(f"Saved: {label}_convergence_all_templates.png")
    print(f"Saved: {label}_summary_by_template.png")

    return all_results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Synthetic data
    run_synthetic_analysis()

    # Real data aggregated over all templates
    run_real_analysis_aggregated(label="real")

    plt.close("all")
    print()
    print("=" * 60)
    print(f"DONE - All plots saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
