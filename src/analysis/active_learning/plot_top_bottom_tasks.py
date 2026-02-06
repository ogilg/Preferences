"""Print and export a table of the top and bottom N tasks by utility with their prompts.

Usage:
    python -m src.analysis.active_learning.plot_top_bottom_tasks --experiment-id gemma3_revealed_v1
    python -m src.analysis.active_learning.plot_top_bottom_tasks --experiment-id gemma3_revealed_v1 --n 15
"""
from __future__ import annotations

import argparse
import textwrap

from src.analysis.active_learning.utils import load_ranked_tasks


def truncate_prompt(prompt: str, max_len: int = 120) -> str:
    text = prompt.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def print_table(tasks: list[dict], label: str) -> None:
    print(f"\n{'=' * 100}")
    print(f"  {label}")
    print(f"{'=' * 100}")
    print(f"  {'Rank':>4}  {'μ':>7}  {'σ':>6}  {'Dataset':<12}  Prompt")
    print(f"  {'-' * 4}  {'-' * 7}  {'-' * 6}  {'-' * 12}  {'-' * 60}")
    for t in tasks:
        prompt_short = truncate_prompt(t["prompt"])
        print(f"  {t['rank']:>4}  {t['mu']:>+7.2f}  {t['sigma']:>6.2f}  {t['dataset']:<12}  {prompt_short}")
    print(f"{'=' * 100}")


def main():
    parser = argparse.ArgumentParser(description="Show top and bottom tasks by utility")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--n", type=int, default=10, help="Number of top/bottom tasks to show")
    args = parser.parse_args()

    tasks = load_ranked_tasks(args.experiment_id, args.run_name)
    print(f"Loaded {len(tasks)} tasks")

    sorted_tasks = sorted(tasks, key=lambda t: t["mu"], reverse=True)

    print_table(sorted_tasks[:args.n], f"TOP {args.n} MOST PREFERRED")
    print_table(sorted_tasks[-args.n:], f"BOTTOM {args.n} LEAST PREFERRED")

    # Dataset breakdown in top/bottom
    n = args.n
    top_datasets = {}
    bottom_datasets = {}
    for t in sorted_tasks[:n]:
        top_datasets[t["dataset"]] = top_datasets.get(t["dataset"], 0) + 1
    for t in sorted_tasks[-n:]:
        bottom_datasets[t["dataset"]] = bottom_datasets.get(t["dataset"], 0) + 1

    print(f"\nDataset composition:")
    print(f"  Top {n}:    {top_datasets}")
    print(f"  Bottom {n}: {bottom_datasets}")


if __name__ == "__main__":
    main()
