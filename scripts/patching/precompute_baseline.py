"""Select 10 tasks at evenly spaced utility quantiles and precompute baseline P(A>B) for all 45 pairs."""

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from src.fitting.protocol import default_preference_probability
from src.task_data import OriginDataset
from src.task_data.loader import load_tasks

THURSTONIAN_PATH = Path(
    "results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_80fa9dc8.csv"
)
OUTPUT_DIR = Path("experiments/patching/pilot")

ALL_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]


def load_thurstonian(path: Path) -> list[dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return [
            {"task_id": row["task_id"], "mu": float(row["mu"]), "sigma": float(row["sigma"])}
            for row in reader
        ]


def select_tasks(scores: list[dict], n: int = 10) -> list[dict]:
    sorted_scores = sorted(scores, key=lambda x: x["mu"])
    n_total = len(sorted_scores)
    percentiles = [5 + 10 * i for i in range(n)]
    indices = [int(p / 100 * (n_total - 1)) for p in percentiles]
    return [sorted_scores[i] for i in indices]


def load_task_prompts(task_ids: set[str]) -> dict[str, str]:
    tasks = load_tasks(n=100_000, origins=ALL_ORIGINS)
    prompts = {}
    for t in tasks:
        if t.id in task_ids:
            prompts[t.id] = t.prompt
    missing = task_ids - set(prompts.keys())
    if missing:
        raise ValueError(f"Could not find prompts for: {missing}")
    return prompts


def compute_baseline_probs(tasks: list[dict]) -> list[dict]:
    results = []
    for i, (a, b) in enumerate(combinations(range(len(tasks)), 2)):
        ta, tb = tasks[a], tasks[b]
        p_ab = default_preference_probability(ta["mu"], tb["mu"], ta["sigma"], tb["sigma"])
        results.append({
            "pair_id": f"pair_{i:04d}",
            "task_a_id": ta["task_id"],
            "task_b_id": tb["task_id"],
            "task_a_prompt": ta["prompt"],
            "task_b_prompt": tb["prompt"],
            "mu_a": ta["mu"],
            "mu_b": tb["mu"],
            "sigma_a": ta["sigma"],
            "sigma_b": tb["sigma"],
            "delta_mu": ta["mu"] - tb["mu"],
            "p_a_over_b": p_ab,
            "p_b_over_a": 1.0 - p_ab,
        })
    return results


def main():
    scores = load_thurstonian(THURSTONIAN_PATH)
    print(f"Loaded {len(scores)} tasks from Thurstonian CSV")

    tasks = select_tasks(scores)

    # Enrich with task prompts
    task_ids = {t["task_id"] for t in tasks}
    prompts = load_task_prompts(task_ids)
    for t in tasks:
        t["prompt"] = prompts[t["task_id"]]

    print(f"\nSelected {len(tasks)} tasks at evenly spaced utility quantiles:")
    for i, t in enumerate(tasks):
        preview = t["prompt"][:60].replace("\n", " ")
        print(f"  {i}: {t['task_id']:30s}  mu={t['mu']:+.3f}  sigma={t['sigma']:.3f}  [{preview}...]")

    pairs = compute_baseline_probs(tasks)

    # Summary stats
    probs = [p["p_a_over_b"] for p in pairs]
    print(f"\n{len(pairs)} pairs")
    print(f"P(A>B) range: [{min(probs):.3f}, {max(probs):.3f}]")
    print(f"P(A>B) mean:  {np.mean(probs):.3f}")
    decisive = sum(1 for p in probs if max(p, 1 - p) > 0.8)
    print(f"Decisive pairs (max(P, 1-P) > 0.8): {decisive}/{len(pairs)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks_path = OUTPUT_DIR / "selected_tasks.json"
    with open(tasks_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"\nSaved selected tasks to {tasks_path}")

    probs_path = OUTPUT_DIR / "baseline_p_choose.json"
    with open(probs_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved baseline P(A>B) to {probs_path}")


if __name__ == "__main__":
    main()
