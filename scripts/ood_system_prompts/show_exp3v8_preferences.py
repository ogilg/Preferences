"""Print each target task prompt alongside its generated preference sentences."""

import json
from pathlib import Path

from src.task_data.loader import _load_origin
from src.task_data.task import OriginDataset

PREFS_PATH = Path("configs/ood/preferences/exp3_v8_preferences.json")

LOADABLE_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]


def main():
    all_tasks = {}
    for origin in LOADABLE_ORIGINS:
        for task in _load_origin(origin):
            all_tasks[task.id] = task

    with open(PREFS_PATH) as f:
        prefs = json.load(f)

    for i, p in enumerate(prefs, 1):
        tid = p["task_id"]
        task = all_tasks[tid]
        scores = p["relevance_scores"]
        status = "OK" if scores["max_off_target"] <= 0.5 else "NON-UNIQUE"

        print(f"{'='*80}")
        print(f"[{i}/20] {tid}  [{status}]  target={scores['target']:.1f}  max_off={scores['max_off_target']:.1f}")
        print(f"{'='*80}")
        print(f"\nTASK PROMPT:\n{task.prompt[:400]}")
        print(f"\nPRO:     {p['interest_sentence_pro']}")
        print(f"ANTI:    {p['interest_sentence_anti']}")
        print(f"NEUTRAL: {p['neutral_sentence']}")
        print()


if __name__ == "__main__":
    main()
