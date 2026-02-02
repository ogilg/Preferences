"""Compute transitivity (cycle probability) for preference measurements.

Usage:
    python -m src.analysis.transitivity.run --experiment-id probe_3 --model llama-3.1-8b --type pre_stated
    python -m src.analysis.transitivity.run --list-models --experiment-id probe_3
"""
from __future__ import annotations

import argparse
from datetime import datetime
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.measurement.storage import EXPERIMENTS_DIR, list_runs
from src.analysis.transitivity.transitivity import measure_transitivity, TransitivityResult
from src.analysis.transitivity.wins_matrix import load_wins_matrix_for_run


OUTPUT_DIR = Path("src/analysis/transitivity/plots")


class MeasurementType(Enum):
    PRE_STATED = "pre_stated"
    POST_STATED = "post_stated"
    PRE_REVEALED = "pre_revealed"
    POST_REVEALED = "post_revealed"

    @property
    def experiment_subdir(self) -> str:
        return {
            MeasurementType.PRE_STATED: "pre_task_stated",
            MeasurementType.POST_STATED: "post_task_stated",
            MeasurementType.PRE_REVEALED: "pre_task_revealed",
            MeasurementType.POST_REVEALED: "post_task_revealed",
        }[self]

    @property
    def is_revealed(self) -> bool:
        return self in (MeasurementType.PRE_REVEALED, MeasurementType.POST_REVEALED)

    @property
    def display_name(self) -> str:
        return {
            MeasurementType.PRE_STATED: "Pre-task Stated",
            MeasurementType.POST_STATED: "Post-task Stated",
            MeasurementType.PRE_REVEALED: "Pre-task Revealed",
            MeasurementType.POST_REVEALED: "Post-task Revealed",
        }[self]

    def get_results_dir(self, experiment_id: str) -> Path:
        return EXPERIMENTS_DIR / experiment_id / self.experiment_subdir


def list_available_models(experiment_id: str) -> set[str]:
    """List all models with data in any measurement type."""
    models: set[str] = set()
    for mtype in MeasurementType:
        results_dir = mtype.get_results_dir(experiment_id)
        if not results_dir.exists():
            continue
        for config in list_runs(results_dir):
            models.add(config.model_short)
    return models


def load_runs_for_analysis(
    experiment_id: str,
    model: str,
    measurement_type: MeasurementType,
    min_tasks: int = 3,
) -> list[tuple[Path, str]]:
    """Load run directories for transitivity analysis.

    Returns list of (run_dir, run_name) tuples.
    """
    results_dir = measurement_type.get_results_dir(experiment_id)
    if not results_dir.exists():
        return []

    runs = []
    for config in list_runs(results_dir):
        if config.model_short != model:
            continue
        runs.append((config.run_dir, config.template_name))

    return runs


def run_transitivity_analysis(
    experiment_id: str,
    model: str,
    measurement_type: MeasurementType,
    min_tasks: int = 3,
) -> list[tuple[str, TransitivityResult]]:
    """Run transitivity analysis for a model and measurement type."""
    runs = load_runs_for_analysis(experiment_id, model, measurement_type, min_tasks)

    if not runs:
        return []

    results: list[tuple[str, TransitivityResult]] = []

    for run_dir, run_name in runs:
        try:
            wins, task_ids = load_wins_matrix_for_run(run_dir, measurement_type.is_revealed)
        except FileNotFoundError:
            continue

        if len(task_ids) < min_tasks:
            continue

        result = measure_transitivity(wins)
        results.append((run_name, result))

    return results


def plot_transitivity_results(
    results: list[tuple[str, TransitivityResult]],
    output_path: Path,
    title: str,
) -> None:
    """Plot cycle probabilities across runs."""
    if not results:
        return

    names = [name for name, _ in results]
    cycle_probs = [r.cycle_probability for _, r in results]
    n_cycles = [r.n_cycles for _, r in results]
    n_triads = [r.n_triads for _, r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(names))
    bars = ax.bar(x, cycle_probs, color="steelblue", alpha=0.8, edgecolor="white")

    # Add hard cycle counts as text
    for i, (cp, nc, nt) in enumerate(zip(cycle_probs, n_cycles, n_triads)):
        ax.text(i, cp + 0.01, f"{nc}/{nt}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.25, color="gray", linestyle=":", alpha=0.7, label="random=0.25")
    ax.set_xlabel("Run")
    ax.set_ylabel("Cycle Probability")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compute transitivity for preference measurements")
    parser.add_argument("--experiment-id", type=str, required=True, help="Experiment ID to load from")
    parser.add_argument("--model", type=str, help="Model short name (e.g., llama-3.1-8b)")
    parser.add_argument(
        "--type",
        type=str,
        choices=[t.value for t in MeasurementType],
        help="Measurement type",
    )
    parser.add_argument("--min-tasks", type=int, default=3, help="Minimum tasks required")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    if args.list_models:
        models = list_available_models(args.experiment_id)
        print(f"Available models in {args.experiment_id} ({len(models)}):")
        for m in sorted(models):
            print(f"  {m}")
        return

    if not args.model:
        parser.error("--model is required (or use --list-models)")
    if not args.type:
        parser.error("--type is required")

    measurement_type = MeasurementType(args.type)

    print(f"Running transitivity analysis:")
    print(f"  Experiment: {args.experiment_id}")
    print(f"  Model: {args.model}")
    print(f"  Type: {measurement_type.display_name}")
    print()

    results = run_transitivity_analysis(
        args.experiment_id,
        args.model,
        measurement_type,
        min_tasks=args.min_tasks,
    )

    if not results:
        print("No runs found matching criteria")
        return

    print(f"Found {len(results)} runs:")
    for name, result in results:
        sampled_str = " (sampled)" if result.sampled else ""
        print(f"  {name}: cycle_prob={result.cycle_probability:.4f}, "
              f"hard_cycles={result.n_cycles}/{result.n_triads}{sampled_str}")

    # Plot
    date_str = datetime.now().strftime("%m%d%y")
    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / f"plot_{date_str}_transitivity_{measurement_type.value}.png"
    plot_transitivity_results(
        results,
        plot_path,
        f"{args.model} [{args.experiment_id}] - {measurement_type.display_name}",
    )
    print(f"\nSaved plot to {plot_path}")

    # Save results
    summary = {
        "experiment_id": args.experiment_id,
        "model": args.model,
        "measurement_type": measurement_type.value,
        "n_runs": len(results),
        "runs": [
            {
                "name": name,
                "cycle_probability": float(r.cycle_probability),
                "n_triads": r.n_triads,
                "n_hard_cycles": r.n_cycles,
            }
            for name, r in results
        ],
    }

    yaml_path = output_dir / f"transitivity_{measurement_type.value}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print(f"Saved results to {yaml_path}")


if __name__ == "__main__":
    main()
