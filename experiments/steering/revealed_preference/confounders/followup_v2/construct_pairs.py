"""Construct utility-matched pairs from Thurstonian mu values.

Pairs tasks with similar mu (borderline by construction) and across Δmu bins.
"""

import csv
import json
import random
from pathlib import Path

THURSTONIAN_CSV = Path(
    "results/experiments/gemma3_3k_run2/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/"
    "thurstonian_a1ebd06e.csv"
)
TASKS_FILE = Path("activations/gemma_3_27b/completions_with_activations.json")
OUTPUT_DIR = Path("experiments/steering/revealed_preference/confounders/followup_v2")


def load_mu_values() -> dict[str, float]:
    mu_values = {}
    with open(THURSTONIAN_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mu_values[row["task_id"]] = float(row["mu"])
    return mu_values


def load_task_prompts() -> dict[str, dict]:
    with open(TASKS_FILE) as f:
        all_tasks = json.load(f)
    return {t["task_id"]: t for t in all_tasks}


def construct_pairs(
    mu_values: dict[str, float],
    task_prompts: dict[str, dict],
    seed: int = 42,
) -> list[dict]:
    """Construct pairs across Δmu bins."""
    rng = random.Random(seed)

    # Filter to tasks that have both mu and prompt
    valid_tasks = []
    for task_id, mu in mu_values.items():
        if task_id in task_prompts:
            valid_tasks.append({
                "task_id": task_id,
                "mu": mu,
                "origin": task_prompts[task_id]["origin"],
                "task_prompt": task_prompts[task_id]["task_prompt"],
            })

    print(f"Valid tasks (have both mu and prompt): {len(valid_tasks)}")

    # Sort by mu
    valid_tasks.sort(key=lambda t: t["mu"])

    # Print mu distribution
    mus = [t["mu"] for t in valid_tasks]
    print(f"Mu range: [{min(mus):.1f}, {max(mus):.1f}]")
    print(f"Mu mean: {sum(mus)/len(mus):.2f}, median: {sorted(mus)[len(mus)//2]:.2f}")

    # Δmu bins: target pairs per bin
    bins = [
        (0, 1, 30),   # borderline
        (1, 2, 20),   # slightly decided
        (2, 3, 20),   # moderately decided
        (3, 5, 20),   # firmly decided
        (5, 20, 20),  # very firm
    ]

    all_pairs = []
    used_tasks = set()

    for delta_lo, delta_hi, target_n in bins:
        bin_pairs = []
        # Shuffle to avoid always pairing the same tasks
        shuffled = list(valid_tasks)
        rng.shuffle(shuffled)

        for i in range(len(shuffled)):
            if shuffled[i]["task_id"] in used_tasks:
                continue
            for j in range(i + 1, len(shuffled)):
                if shuffled[j]["task_id"] in used_tasks:
                    continue
                delta_mu = abs(shuffled[i]["mu"] - shuffled[j]["mu"])
                if delta_lo <= delta_mu < delta_hi:
                    # Put higher-mu task as A (so positive coef should push toward A)
                    if shuffled[i]["mu"] >= shuffled[j]["mu"]:
                        task_a, task_b = shuffled[i], shuffled[j]
                    else:
                        task_a, task_b = shuffled[j], shuffled[i]

                    bin_pairs.append({
                        "task_a_id": task_a["task_id"],
                        "task_b_id": task_b["task_id"],
                        "task_a_prompt": task_a["task_prompt"],
                        "task_b_prompt": task_b["task_prompt"],
                        "task_a_origin": task_a["origin"],
                        "task_b_origin": task_b["origin"],
                        "mu_a": task_a["mu"],
                        "mu_b": task_b["mu"],
                        "delta_mu": task_a["mu"] - task_b["mu"],
                        "delta_mu_bin": f"{delta_lo}-{delta_hi}",
                    })
                    used_tasks.add(task_a["task_id"])
                    used_tasks.add(task_b["task_id"])
                    break  # Move to next i after finding a match
            if len(bin_pairs) >= target_n:
                break

        all_pairs.extend(bin_pairs)
        print(f"  Δmu [{delta_lo}, {delta_hi}): {len(bin_pairs)} pairs (target {target_n})")

    # Add pair indices
    for i, pair in enumerate(all_pairs):
        pair["pair_idx"] = i

    return all_pairs


def main():
    print("Loading mu values...")
    mu_values = load_mu_values()
    print(f"Loaded {len(mu_values)} mu values")

    print("Loading task prompts...")
    task_prompts = load_task_prompts()
    print(f"Loaded {len(task_prompts)} task prompts")

    print("\nConstructing pairs...")
    pairs = construct_pairs(mu_values, task_prompts)

    print(f"\nTotal pairs: {len(pairs)}")
    print(f"\nΔmu distribution:")
    for pair in pairs[:3]:
        print(f"  Pair {pair['pair_idx']}: Δmu={pair['delta_mu']:.2f} "
              f"(A={pair['task_a_id']}, B={pair['task_b_id']})")

    # Save
    output_path = OUTPUT_DIR / "utility_matched_pairs.json"
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"\nSaved {len(pairs)} pairs to {output_path}")

    # Summary stats
    by_bin = {}
    for p in pairs:
        b = p["delta_mu_bin"]
        if b not in by_bin:
            by_bin[b] = []
        by_bin[b].append(p["delta_mu"])

    print("\nBin summary:")
    print(f"{'Bin':<10} {'N':>4} {'Mean Δmu':>10} {'Min':>8} {'Max':>8}")
    for b in sorted(by_bin.keys()):
        vals = by_bin[b]
        print(f"{b:<10} {len(vals):>4} {sum(vals)/len(vals):>10.2f} "
              f"{min(vals):>8.2f} {max(vals):>8.2f}")


if __name__ == "__main__":
    main()
