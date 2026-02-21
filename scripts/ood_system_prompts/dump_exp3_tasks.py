"""Dump prompt text for the 50 minimal-pairs-v7 tasks used in OOD experiment 3."""

import json
from pathlib import Path

from src.task_data.loader import load_filtered_tasks
from src.task_data.task import OriginDataset

TASKS_PATH = Path("configs/ood/tasks/minimal_pairs_v7_tasks.json")

ALL_ORIGINS = [
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
    OriginDataset.WILDCHAT,
]


def main() -> None:
    with open(TASKS_PATH) as f:
        task_ids = set(json.load(f)["task_ids"])

    tasks = load_filtered_tasks(
        n=len(task_ids),
        origins=ALL_ORIGINS,
        task_ids=task_ids,
    )

    result = {t.id: t.prompt for t in tasks}

    missing = task_ids - set(result.keys())
    if missing:
        raise ValueError(f"Missing tasks: {missing}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
