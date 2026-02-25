"""Phase 1: Sample 1500 task IDs from existing activations, stratified by origin.

Saves:
  experiments/probe_generalization/multi_role_ablation/task_ids_all.txt
  experiments/probe_generalization/multi_role_ablation/task_ids_train.txt
  experiments/probe_generalization/multi_role_ablation/task_ids_eval.txt
  configs/measurement/active_learning/exclude_mra_non_target.txt  (all activation task IDs NOT in 1500)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

REPO = Path(__file__).parent.parent.parent
ACTIVATIONS_DIR = REPO / "activations/gemma_3_27b"
COMPLETIONS_PATH = ACTIVATIONS_DIR / "completions_with_activations.json"
OUT_DIR = REPO / "experiments/probe_generalization/multi_role_ablation"

N_TOTAL = 1500
N_TRAIN = 1000
N_EVAL = 500
SEED = 42


def main() -> None:
    with open(COMPLETIONS_PATH) as f:
        completions = json.load(f)

    # Group by origin
    by_origin: dict[str, list[str]] = defaultdict(list)
    for c in completions:
        origin = c.get("origin", "UNKNOWN").upper()
        by_origin[origin].append(c["task_id"])

    print("Task counts by origin:")
    for origin, ids in sorted(by_origin.items()):
        print(f"  {origin}: {len(ids)}")

    total_available = sum(len(ids) for ids in by_origin.values())
    print(f"Total tasks with activations: {total_available}")

    origins = ["WILDCHAT", "ALPACA", "MATH", "BAILBENCH", "STRESS_TEST"]
    rng = np.random.default_rng(SEED)

    # Stratified sampling: equal share per origin (or all if fewer available)
    sampled_ids: list[str] = []
    n_remaining = N_TOTAL
    origin_pools = {o: by_origin.get(o, []) for o in origins}

    # Shuffle each pool
    for o in origins:
        pool = origin_pools[o]
        rng.shuffle(pool)
        origin_pools[o] = pool

    # Greedy stratified: assign share to small origins first
    quotas: dict[str, int] = {}
    unassigned = set(origins)
    while unassigned:
        share = n_remaining / len(unassigned)
        small = {o for o in unassigned if len(origin_pools[o]) <= share}
        if not small:
            break
        for o in small:
            quotas[o] = len(origin_pools[o])
            n_remaining -= quotas[o]
        unassigned -= small
    per = n_remaining // len(unassigned) if unassigned else 0
    extra = n_remaining % len(unassigned) if unassigned else 0
    for i, o in enumerate(sorted(unassigned)):
        quotas[o] = per + (1 if i < extra else 0)

    print("\nSampling quotas:")
    for o in origins:
        print(f"  {o}: {quotas.get(o, 0)}")

    for o in origins:
        sampled_ids.extend(origin_pools[o][:quotas.get(o, 0)])

    assert len(sampled_ids) == N_TOTAL, f"Expected {N_TOTAL}, got {len(sampled_ids)}"

    # Shuffle for train/eval split
    rng.shuffle(sampled_ids)
    train_ids = sampled_ids[:N_TRAIN]
    eval_ids = sampled_ids[N_TRAIN:]
    assert len(eval_ids) == N_EVAL

    print(f"\nTotal sampled: {len(sampled_ids)}")
    print(f"Train: {len(train_ids)}, Eval: {len(eval_ids)}")

    # Save task ID lists
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "task_ids_all.txt").write_text("\n".join(sampled_ids))
    (OUT_DIR / "task_ids_train.txt").write_text("\n".join(train_ids))
    (OUT_DIR / "task_ids_eval.txt").write_text("\n".join(eval_ids))
    print(f"Saved task ID lists to {OUT_DIR}")

    # Create exclude file: all activation task IDs NOT in our 1500
    all_activation_ids = {c["task_id"] for c in completions}
    target_set = set(sampled_ids)
    exclude_ids = sorted(all_activation_ids - target_set)
    exclude_path = REPO / "configs/measurement/active_learning/exclude_mra_non_target.txt"
    exclude_path.write_text("\n".join(exclude_ids))
    print(f"Saved exclude file ({len(exclude_ids)} tasks) to {exclude_path}")


if __name__ == "__main__":
    main()
