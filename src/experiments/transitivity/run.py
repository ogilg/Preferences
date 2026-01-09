"""Usage: python -m src.experiments.transitivity.run [--results-dir results/]"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.experiments.transitivity import measure_transitivity
from src.preferences.storage import MEASUREMENTS_DIR


def load_wins_matrix(measurements: list[dict], task_ids: list[str]) -> np.ndarray:
    """wins[i,j] = number of times task i beat task j."""
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    n = len(task_ids)
    wins = np.zeros((n, n), dtype=np.int32)

    for m in measurements:
        i = id_to_idx[m["task_a"]]
        j = id_to_idx[m["task_b"]]
        if m["choice"] == "a":
            wins[i, j] += 1
        else:
            wins[j, i] += 1

    return wins


def find_thurstonian_files(run_dir: Path) -> tuple[Path, Path]:
    """Find thurstonian YAML and CSV files, returns (yaml_path, csv_path)."""
    # Try hash-based filenames first
    for pattern in ["thurstonian_exhaustive_pairwise_*.yaml", "thurstonian_active_learning_*.yaml"]:
        matches = list(run_dir.glob(pattern))
        if matches:
            yaml_path = matches[0]
            return yaml_path, yaml_path.with_suffix(".csv")

    # Fallback to old naming
    for yaml_name in ["thurstonian_exhaustive_pairwise.yaml", "thurstonian_active_learning.yaml", "thurstonian.yaml"]:
        yaml_path = run_dir / yaml_name
        if yaml_path.exists():
            return yaml_path, yaml_path.with_suffix(".csv")

    raise FileNotFoundError(f"No thurstonian YAML found in {run_dir}")


def load_thurstonian_csv(csv_path: Path) -> tuple[list[str], list[float]]:
    """Load task_ids and sigmas from thurstonian CSV."""
    task_ids = []
    sigmas = []
    with open(csv_path) as f:
        next(f)  # Skip header
        for line in f:
            task_id, _, sigma = line.strip().split(",")
            task_ids.append(task_id)
            sigmas.append(float(sigma))
    return task_ids, sigmas


def analyze_run(run_dir: Path) -> dict:
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    with open(run_dir / "measurements.yaml") as f:
        measurements = yaml.safe_load(f)

    thurstonian_yaml, csv_path = find_thurstonian_files(run_dir)

    with open(thurstonian_yaml) as f:
        thurstonian = yaml.safe_load(f)

    task_ids, sigmas = load_thurstonian_csv(csv_path)
    sigma_max = max(sigmas)

    # Filter measurements to only include tasks in the Thurstonian fit
    task_set = set(task_ids)
    measurements = [m for m in measurements if m["task_a"] in task_set and m["task_b"] in task_set]

    wins = load_wins_matrix(measurements, task_ids)
    trans = measure_transitivity(wins)

    return {
        "run_id": run_dir.name,
        "template_name": config["template_name"],
        "model": config["model_short"],
        "n_tasks": len(task_ids),
        "n_measurements": len(measurements),
        "cycle_probability": trans.cycle_probability,
        "log_cycle_prob": trans.log_cycle_prob,
        "n_cycles": trans.n_cycles,
        "n_triads": trans.n_triads,
        "thurstonian_converged": thurstonian["converged"],
        "sigma_max": sigma_max,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute transitivity for all runs")
    parser.add_argument("--results-dir", type=Path, default=MEASUREMENTS_DIR)
    parser.add_argument("--output", type=Path, default=Path("results/transitivity.png"))
    parser.add_argument("--show", action="store_true", help="Show plot interactively instead of saving")
    args = parser.parse_args()

    results = []
    for run_dir in sorted(args.results_dir.iterdir()):
        if not (run_dir / "config.yaml").exists():
            continue
        try:
            result = analyze_run(run_dir)
            results.append(result)
            print(f"{result['run_id']}: cycle_prob={result['cycle_probability']:.4f}, "
                  f"log10={result['log_cycle_prob']:.2f}, converged={result['thurstonian_converged']}")
        except Exception as e:
            print(f"{run_dir.name}: Error - {e}")

    if not results:
        print("No results found")
        return

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Cycle probability by run
    ax = axes[0]
    x = range(len(results))
    colors = ["green" if r["thurstonian_converged"] else "red" for r in results]
    ax.bar(x, [r["cycle_probability"] for r in results], color=colors, alpha=0.7)
    ax.set_xlabel("Run")
    ax.set_ylabel("Cycle Probability")
    ax.set_title("Transitivity by Run (green=converged, red=not)")
    ax.set_xticks(x)
    ax.set_xticklabels([r["template_name"] for r in results], rotation=45, ha="right")

    # Plot 2: Cycle prob vs sigma_max
    ax = axes[1]
    cycle_probs = [r["cycle_probability"] for r in results]
    sigma_maxs = [r["sigma_max"] for r in results]
    converged = [r["thurstonian_converged"] for r in results]
    colors = ["green" if c else "red" for c in converged]
    ax.scatter(cycle_probs, sigma_maxs, c=colors, alpha=0.7, s=50)
    ax.set_xlabel("Cycle Probability")
    ax.set_ylabel("Max σ (Thurstonian)")
    ax.set_title("Transitivity vs Model Uncertainty")
    ax.set_yscale("log")

    fig.tight_layout()

    if args.show:
        plt.show()
    else:
        fig.savefig(args.output, dpi=150)
        print(f"Saved plot to {args.output}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"Runs: {len(results)}")
    print(f"Mean cycle prob: {np.mean(cycle_probs):.4f}")
    print(f"Converged: {sum(converged)}/{len(converged)}")


if __name__ == "__main__":
    main()
