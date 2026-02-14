"""Construct task pairs for single-task steering experiment.

Bins tasks by mu, pairs within bins. Outputs a JSON list of pairs with metadata.
"""

import csv
import json
import random
import sys
from pathlib import Path

THURSTONIAN_PATH = Path(
    "results/experiments/gemma3_3k_run2/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/"
    "thurstonian_a1ebd06e.csv"
)

MU_BIN_EDGES = list(range(-10, 12, 2))  # [-10, -8, -6, ..., 8, 10]
PAIRS_PER_BIN = 30
SEED = 42

# Stress test data file is missing, exclude these task IDs
EXCLUDED_PREFIXES = ("stresstest_",)


def load_mu_values(path: Path) -> dict[str, float]:
    mu_values = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["task_id"]
            if any(tid.startswith(p) for p in EXCLUDED_PREFIXES):
                continue
            mu_values[tid] = float(row["mu"])
    return mu_values


def bin_tasks(mu_values: dict[str, float]) -> dict[str, list[str]]:
    bins: dict[str, list[str]] = {}
    for lo, hi in zip(MU_BIN_EDGES[:-1], MU_BIN_EDGES[1:]):
        bin_label = f"[{lo},{hi})"
        bins[bin_label] = [
            tid for tid, mu in mu_values.items()
            if lo <= mu < hi
        ]
    return bins


def construct_pairs(
    bins: dict[str, list[str]],
    mu_values: dict[str, float],
    pairs_per_bin: int,
    rng: random.Random,
) -> list[dict]:
    pairs = []
    for bin_label, task_ids in sorted(bins.items()):
        if len(task_ids) < 2:
            print(f"Bin {bin_label}: only {len(task_ids)} tasks, skipping")
            continue

        shuffled = task_ids.copy()
        rng.shuffle(shuffled)

        # Pair consecutive tasks from shuffled list
        n_available = len(shuffled) // 2
        n_pairs = min(pairs_per_bin, n_available)

        for i in range(n_pairs):
            task_a = shuffled[2 * i]
            task_b = shuffled[2 * i + 1]
            pairs.append({
                "pair_id": len(pairs),
                "task_a": task_a,
                "task_b": task_b,
                "mu_a": mu_values[task_a],
                "mu_b": mu_values[task_b],
                "delta_mu": abs(mu_values[task_a] - mu_values[task_b]),
                "mu_bin": bin_label,
            })

    return pairs


def main():
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scripts/single_task_steering/pairs.json")

    mu_values = load_mu_values(THURSTONIAN_PATH)
    print(f"Loaded {len(mu_values)} task mu values")

    bins = bin_tasks(mu_values)
    for bin_label, tasks in sorted(bins.items()):
        print(f"  {bin_label}: {len(tasks)} tasks")

    rng = random.Random(SEED)
    pairs = construct_pairs(bins, mu_values, PAIRS_PER_BIN, rng)

    print(f"\nConstructed {len(pairs)} pairs")
    print(f"Mean delta_mu: {sum(p['delta_mu'] for p in pairs) / len(pairs):.2f}")

    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
