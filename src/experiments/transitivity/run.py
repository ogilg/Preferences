"""Usage: python -m src.experiments.transitivity.run [--results-dir results/]"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.experiments.transitivity import measure_transitivity


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


def analyze_run(run_dir: Path) -> dict:
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    with open(run_dir / "measurements.yaml") as f:
        measurements = yaml.safe_load(f)

    # Try new filenames first, fallback to old
    thurstonian_yaml = None
    for yaml_name in ["thurstonian_exhaustive_pairwise.yaml", "thurstonian_active_learning.yaml", "thurstonian.yaml"]:
        yaml_path = run_dir / yaml_name
        if yaml_path.exists():
            with open(yaml_path) as f:
                thurstonian = yaml.safe_load(f)
            thurstonian_yaml = yaml_path
            break

    if thurstonian_yaml is None:
        raise FileNotFoundError(f"No thurstonian YAML found in {run_dir}")

    # Read sigma from CSV (use same base name as YAML)
    csv_path = thurstonian_yaml.with_suffix(".csv")
    if csv_path.exists():
        sigmas = []
        with open(csv_path) as f:
            next(f)  # Skip header
            for line in f:
                _, _, sigma = line.strip().split(",")
                sigmas.append(float(sigma))
        sigma_max = max(sigmas)
    else:
        # Fallback for old format
        sigma_max = max(thurstonian["sigma"]) if "sigma" in thurstonian else -1.0

    task_ids = config["task_ids"]
    wins = load_wins_matrix(measurements, task_ids)
    trans = measure_transitivity(wins)

    return {
        "run_id": run_dir.name,
        "template_id": config["template_id"],
        "model": config["model_short"],
        "n_tasks": config["n_tasks"],
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
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None,
                        help="Save plot to file instead of showing")
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
    ax.set_xticklabels([r["template_id"] for r in results], rotation=45, ha="right")

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

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()

    # Summary
    print(f"\n--- Summary ---")
    print(f"Runs: {len(results)}")
    print(f"Mean cycle prob: {np.mean(cycle_probs):.4f}")
    print(f"Converged: {sum(converged)}/{len(converged)}")


if __name__ == "__main__":
    main()
