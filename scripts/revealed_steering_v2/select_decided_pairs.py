"""Select 200 new pairs for the revealed steering v2 follow-up.

Draws from the full task pool (Thurstonian CSV), not restricted to borderline.
Constraints:
  - |delta_mu| < 2 (within-bin)
  - Both tasks have activations
  - Not already in the 300 existing pairs
  - Stratified by mu_bin (10 bins, 20 per bin)

Outputs: experiments/revealed_steering_v2/followup/pairs_200_new.json
"""

import csv
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

THURSTONIAN_CSV = Path(
    "results/experiments/main_probes/gemma3_10k_run1/"
    "pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/"
    "thurstonian_80fa9dc8.csv"
)
ACTIVATIONS_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
COMPLETIONS_PATH = Path("activations/gemma_3_27b/completions_with_activations.json")
EXISTING_PAIRS_PATH = Path(
    "experiments/steering/replication/fine_grained/results/pairs.json"
)
OUTPUT_DIR = Path("experiments/revealed_steering_v2/followup")
OUTPUT_PATH = OUTPUT_DIR / "pairs_200_new.json"

N_NEW_PAIRS = 200
N_BINS = 10
PAIRS_PER_BIN = N_NEW_PAIRS // N_BINS  # 20
MAX_DELTA_MU = 2.0
SEED = 42


def load_tasks() -> dict[str, float]:
    """Load task_id -> mu from Thurstonian CSV."""
    tasks = {}
    with open(THURSTONIAN_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks[row["task_id"]] = float(row["mu"])
    return tasks


def load_activation_task_ids() -> set[str]:
    data = np.load(ACTIVATIONS_PATH, allow_pickle=True)
    return set(data["task_ids"])


def load_task_prompts() -> dict[str, str]:
    with open(COMPLETIONS_PATH) as f:
        data = json.load(f)
    return {item["task_id"]: item["task_prompt"] for item in data}


def load_existing_pair_keys() -> set[frozenset[str]]:
    with open(EXISTING_PAIRS_PATH) as f:
        pairs = json.load(f)
    return {frozenset([p["task_a"], p["task_b"]]) for p in pairs}


def assign_mu_bin(mean_mu: float, bin_edges: list[float]) -> int:
    for i in range(len(bin_edges) - 1):
        if mean_mu <= bin_edges[i + 1]:
            return i
    return len(bin_edges) - 2


def main():
    random.seed(SEED)

    # Load data
    tasks = load_tasks()
    act_ids = load_activation_task_ids()
    existing_keys = load_existing_pair_keys()
    task_prompts = load_task_prompts()

    # Filter to tasks with activations
    eligible_tasks = {tid: mu for tid, mu in tasks.items() if tid in act_ids}
    print(f"Tasks in Thurstonian: {len(tasks)}")
    print(f"Tasks with activations: {len(act_ids)}")
    print(f"Eligible tasks (both): {len(eligible_tasks)}")

    # Compute bin edges from eligible tasks (deciles of mu)
    mus = sorted(eligible_tasks.values())
    bin_edges = [np.percentile(mus, p) for p in np.linspace(0, 100, N_BINS + 1)]
    bin_edges[0] = float("-inf")
    bin_edges[-1] = float("inf")

    # Group tasks by bin
    task_bins: dict[int, list[str]] = {i: [] for i in range(N_BINS)}
    for tid, mu in eligible_tasks.items():
        b = assign_mu_bin(mu, bin_edges)
        task_bins[b].append(tid)

    for b in range(N_BINS):
        print(f"  Bin {b}: {len(task_bins[b])} tasks")

    # Generate candidate pairs per bin
    print("\nGenerating candidate pairs per bin...")
    candidates_per_bin: dict[int, list[dict]] = {i: [] for i in range(N_BINS)}

    for b in range(N_BINS):
        bin_tasks = task_bins[b]
        count = 0
        for t_a, t_b in combinations(bin_tasks, 2):
            mu_a, mu_b = eligible_tasks[t_a], eligible_tasks[t_b]
            delta_mu = mu_b - mu_a
            if abs(delta_mu) >= MAX_DELTA_MU:
                continue
            key = frozenset([t_a, t_b])
            if key in existing_keys:
                continue
            mean_mu = (mu_a + mu_b) / 2
            candidates_per_bin[b].append({
                "task_a": t_a,
                "task_b": t_b,
                "mu_a": mu_a,
                "mu_b": mu_b,
                "delta_mu": delta_mu,
                "mean_mu": mean_mu,
                "mu_bin": b,
            })
            count += 1
        print(f"  Bin {b}: {count} candidate pairs")

    # Also generate cross-bin pairs where the mean falls in a bin
    # but tasks come from adjacent bins
    for b in range(N_BINS):
        adj_bins = [b - 1, b + 1]
        for adj in adj_bins:
            if adj < 0 or adj >= N_BINS:
                continue
            for t_a in task_bins[b]:
                for t_b in task_bins[adj]:
                    if t_a >= t_b:
                        continue
                    mu_a, mu_b = eligible_tasks[t_a], eligible_tasks[t_b]
                    delta_mu = mu_b - mu_a
                    if abs(delta_mu) >= MAX_DELTA_MU:
                        continue
                    key = frozenset([t_a, t_b])
                    if key in existing_keys:
                        continue
                    mean_mu = (mu_a + mu_b) / 2
                    target_bin = assign_mu_bin(mean_mu, bin_edges)
                    if target_bin != b:
                        continue
                    candidates_per_bin[b].append({
                        "task_a": t_a,
                        "task_b": t_b,
                        "mu_a": mu_a,
                        "mu_b": mu_b,
                        "delta_mu": delta_mu,
                        "mean_mu": mean_mu,
                        "mu_bin": b,
                    })

    print("\nAfter adding cross-bin pairs:")
    for b in range(N_BINS):
        print(f"  Bin {b}: {len(candidates_per_bin[b])} candidate pairs")

    # Sample PAIRS_PER_BIN from each bin
    selected = []
    for b in range(N_BINS):
        pool = candidates_per_bin[b]
        if len(pool) < PAIRS_PER_BIN:
            print(f"  WARNING: Bin {b} has only {len(pool)} candidates, taking all")
            sampled = pool
        else:
            sampled = random.sample(pool, PAIRS_PER_BIN)
        selected.extend(sampled)

    # Assign pair IDs and add task texts
    for i, pair in enumerate(selected):
        pair["pair_id"] = f"pair_{300 + i:04d}"
        pair["task_a_text"] = task_prompts[pair["task_a"]]
        pair["task_b_text"] = task_prompts[pair["task_b"]]

    print(f"\nSelected {len(selected)} new pairs")
    print(f"Bin distribution: {dict(sorted(((p['mu_bin'], 1) for p in selected), key=lambda x: x[0]))}")

    bin_counts = Counter(p["mu_bin"] for p in selected)
    print(f"Per-bin counts: {dict(sorted(bin_counts.items()))}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
