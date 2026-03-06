"""Select 100 tasks at evenly spaced utility quantiles from Thurstonian scores."""

import csv
import json
from pathlib import Path

import numpy as np

from src.task_data import OriginDataset
from src.task_data.loader import load_tasks

THURSTONIAN_PATH = Path(
    "results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_80fa9dc8.csv"
)
OUTPUT_DIR = Path("experiments/patching/eot_scaled")

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


def select_tasks(scores: list[dict], n: int = 100) -> list[dict]:
    sorted_scores = sorted(scores, key=lambda x: x["mu"])
    n_total = len(sorted_scores)
    # Evenly spaced percentiles: 0.5%, 1.5%, ..., 99.5%
    percentiles = [(i + 0.5) / n * 100 for i in range(n)]
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


def main():
    scores = load_thurstonian(THURSTONIAN_PATH)
    print(f"Loaded {len(scores)} tasks from Thurstonian CSV")

    tasks = select_tasks(scores, n=100)

    task_ids = {t["task_id"] for t in tasks}
    prompts = load_task_prompts(task_ids)
    for t in tasks:
        t["prompt"] = prompts[t["task_id"]]

    mus = [t["mu"] for t in tasks]
    print(f"Selected {len(tasks)} tasks")
    print(f"  mu range: [{min(mus):.2f}, {max(mus):.2f}]")
    print(f"  mu median: {np.median(mus):.2f}")

    for i, t in enumerate(tasks):
        preview = t["prompt"][:50].replace("\n", " ")
        print(f"  {i:3d}: {t['task_id']:35s}  mu={t['mu']:+7.3f}  [{preview}...]")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "selected_tasks.json"
    with open(out_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
