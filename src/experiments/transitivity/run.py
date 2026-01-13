"""
Compute transitivity (cycle probability) across active learning measurements.

Usage: python -m src.experiments.transitivity.run
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

from src.preferences.storage import MEASUREMENTS_DIR


@dataclass
class RunData:
    run_id: str
    model: str
    comparisons: dict[tuple[str, str], list[str]]
    tasks: set[str]


def load_runs(results_dir: Path) -> list[RunData]:
    """Load comparisons grouped by run (same template/config)."""
    runs = []

    for run_dir in results_dir.iterdir():
        config_path = run_dir / "config.yaml"
        measurements_path = run_dir / "measurements.yaml"
        if not config_path.exists() or not measurements_path.exists():
            continue

        with open(config_path) as f:
            config = yaml.safe_load(f)

        with open(measurements_path) as f:
            measurements = yaml.safe_load(f)

        comparisons: dict[tuple[str, str], list[str]] = defaultdict(list)
        tasks: set[str] = set()

        for m in measurements:
            key = (m["task_a"], m["task_b"])
            comparisons[key].append(m["choice"])
            tasks.add(m["task_a"])
            tasks.add(m["task_b"])

        runs.append(RunData(
            run_id=run_dir.name,
            model=config["model_short"],
            comparisons=dict(comparisons),
            tasks=tasks,
        ))

    return runs


def get_pairwise_prob(
    comparisons: dict[tuple[str, str], list[str]],
    task_a: str,
    task_b: str,
) -> float | None:
    """Get P(task_a > task_b) from comparisons. Returns None if no data."""
    key_ab = (task_a, task_b)
    key_ba = (task_b, task_a)

    wins_a = 0
    total = 0

    if key_ab in comparisons:
        for choice in comparisons[key_ab]:
            total += 1
            if choice == "a":
                wins_a += 1

    if key_ba in comparisons:
        for choice in comparisons[key_ba]:
            total += 1
            if choice == "b":
                wins_a += 1

    if total == 0:
        return None
    return wins_a / total


def sample_transitivity(
    runs: list[RunData],
    n_samples: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Sample triads within runs (same template) and compute cycle probability.
    """
    random.seed(seed)
    np.random.seed(seed)

    cycle_probs = []
    hard_cycles = 0
    n_sampled = 0
    n_attempts = 0

    # Weight runs by number of tasks for fair sampling
    run_weights = [len(r.tasks) for r in runs]
    total_weight = sum(run_weights)
    run_probs = [w / total_weight for w in run_weights]

    pbar = tqdm(total=n_samples, desc="Sampling triads")
    while n_sampled < n_samples and n_attempts < n_samples * 100:
        n_attempts += 1

        # Sample a run
        run = random.choices(runs, weights=run_probs, k=1)[0]
        task_list = list(run.tasks)

        if len(task_list) < 3:
            continue

        # Sample three tasks from this run
        i, j, k = random.sample(task_list, 3)

        # Check all pairs have data within this run
        p_ij = get_pairwise_prob(run.comparisons, i, j)
        p_jk = get_pairwise_prob(run.comparisons, j, k)
        p_ki = get_pairwise_prob(run.comparisons, k, i)

        if p_ij is None or p_jk is None or p_ki is None:
            continue

        n_sampled += 1
        pbar.update(1)

        # Cycle probability: P(i>j>k>i) + P(j>i>k>j)
        p_clockwise = p_ij * p_jk * p_ki
        p_counter = (1 - p_ij) * (1 - p_jk) * (1 - p_ki)
        p_cycle = p_clockwise + p_counter

        cycle_probs.append(p_cycle)

        # Hard cycle: majority preference forms a cycle
        if (p_ij > 0.5 and p_jk > 0.5 and p_ki > 0.5) or \
           (p_ij < 0.5 and p_jk < 0.5 and p_ki < 0.5):
            hard_cycles += 1

    pbar.close()

    return {
        "mean_cycle_prob": float(np.mean(cycle_probs)) if cycle_probs else 0.0,
        "std_cycle_prob": float(np.std(cycle_probs)) if cycle_probs else 0.0,
        "n_sampled": n_sampled,
        "n_hard_cycles": hard_cycles,
        "hard_cycle_rate": hard_cycles / n_sampled if n_sampled > 0 else 0.0,
        "cycle_probs": cycle_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute transitivity via sampling")
    parser.add_argument("--results-dir", type=Path, default=MEASUREMENTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("results/transitivity"))
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading runs from {args.results_dir}...")
    runs = load_runs(args.results_dir)
    models = set(r.model for r in runs)
    n_pairs = sum(len(r.comparisons) for r in runs)
    print(f"Loaded {len(runs)} runs with {n_pairs} total pairs from {len(models)} model(s)")

    model_str = list(models)[0] if len(models) == 1 else f"{len(models)} models"

    print(f"Sampling {args.n_samples} triads (within same template)...")
    result = sample_transitivity(runs, n_samples=args.n_samples, seed=args.seed)

    print(f"\n--- Results ---")
    print(f"Sampled triads: {result['n_sampled']}")
    print(f"Mean cycle prob: {result['mean_cycle_prob']:.4f} ± {result['std_cycle_prob']:.4f}")
    print(f"Hard cycle rate: {result['hard_cycle_rate']:.4f} ({result['n_hard_cycles']}/{result['n_sampled']})")

    # Plot
    args.output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%m%d%y")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(result["cycle_probs"], bins=50, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(result["mean_cycle_prob"], color="red", linestyle="--",
               label=f"mean={result['mean_cycle_prob']:.3f}")
    ax.axvline(0.25, color="gray", linestyle=":", alpha=0.7, label="random=0.25")
    ax.set_xlabel("Cycle Probability")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_str} Transitivity (n={result['n_sampled']} triads)")
    ax.legend()
    plt.tight_layout()

    output_path = args.output_dir / f"plot_{date_str}_transitivity_distribution.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close()

    # Save results
    summary = {
        "model": model_str,
        "n_runs": len(runs),
        "n_pairs": n_pairs,
        "n_sampled_triads": result["n_sampled"],
        "mean_cycle_prob": result["mean_cycle_prob"],
        "std_cycle_prob": result["std_cycle_prob"],
        "hard_cycle_rate": result["hard_cycle_rate"],
    }
    yaml_path = args.output_dir / "transitivity_results.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print(f"Saved results to {yaml_path}")


if __name__ == "__main__":
    main()
