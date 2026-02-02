"""Transitivity analysis for ranking-based preference measurements.

Usage:
    python -m src.analysis.transitivity.ranking_transitivity --model llama-3.1-8b --template post_ranking_basic_en_v1_cseed0
    python -m src.analysis.transitivity.ranking_transitivity --model llama-3.1-8b --list-templates
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.analysis.transitivity.transitivity import measure_transitivity, TransitivityResult
from src.measurement.storage.base import load_yaml


CACHE_DIR = Path("results/cache/ranking")


def load_ranking_cache(model_short: str) -> dict[str, dict]:
    cache_path = CACHE_DIR / f"{model_short}.yaml"
    if not cache_path.exists():
        raise FileNotFoundError(f"No ranking cache found at {cache_path}")
    return load_yaml(cache_path)


def list_templates(cache_data: dict[str, dict]) -> list[str]:
    templates: set[str] = set()
    for entry in cache_data.values():
        templates.add(entry["template_name"])
    return sorted(templates)


def get_rankings_for_template(
    cache_data: dict[str, dict],
    template_name: str,
) -> list[tuple[list[str], list[int]]]:
    """Extract (task_ids, ranking) pairs for a template."""
    rankings: list[tuple[list[str], list[int]]] = []

    for entry in cache_data.values():
        if entry["template_name"] != template_name:
            continue

        task_ids = entry["task_ids"]
        for sample in entry["samples"]:
            rankings.append((task_ids, sample["ranking"]))

    return rankings


def wins_from_rankings(
    rankings: list[tuple[list[str], list[int]]],
) -> tuple[np.ndarray, list[str]]:
    """Build wins matrix from ranking data.

    Each ranking[i] = position of task i (0=best, lower=better).
    """
    all_tasks: set[str] = set()
    for task_ids, _ in rankings:
        all_tasks.update(task_ids)

    task_list = sorted(all_tasks)
    id_to_idx = {tid: i for i, tid in enumerate(task_list)}
    n = len(task_list)
    wins = np.zeros((n, n), dtype=np.int32)

    for task_ids, ranking in rankings:
        for i in range(len(task_ids)):
            for j in range(i + 1, len(task_ids)):
                ti, tj = task_ids[i], task_ids[j]
                gi, gj = id_to_idx[ti], id_to_idx[tj]
                # Lower position = preferred
                if ranking[i] < ranking[j]:
                    wins[gi, gj] += 1
                elif ranking[j] < ranking[i]:
                    wins[gj, gi] += 1

    return wins, task_list


def main():
    parser = argparse.ArgumentParser(description="Transitivity analysis for ranking data")
    parser.add_argument("--model", type=str, required=True, help="Model short name")
    parser.add_argument("--template", type=str, help="Template name to analyze")
    parser.add_argument("--list-templates", action="store_true", help="List available templates")
    args = parser.parse_args()

    cache_data = load_ranking_cache(args.model)

    if args.list_templates:
        templates = list_templates(cache_data)
        print(f"Available templates for {args.model}:")
        for t in templates:
            print(f"  {t}")
        return

    if not args.template:
        parser.error("--template is required (or use --list-templates)")

    rankings = get_rankings_for_template(cache_data, args.template)
    if not rankings:
        print(f"No rankings found for template: {args.template}")
        return

    print(f"Model: {args.model}")
    print(f"Template: {args.template}")
    print(f"Number of ranking measurements: {len(rankings)}")

    wins, task_ids = wins_from_rankings(rankings)
    print(f"Number of tasks: {len(task_ids)}")

    # Count total pairwise comparisons
    total_comparisons = np.sum(wins)
    print(f"Total pairwise comparisons: {total_comparisons}")

    result = measure_transitivity(wins)

    print()
    print("Transitivity Results:")
    print(f"  Cycle probability: {result.cycle_probability:.4f}")
    print(f"  Hard cycles: {result.n_cycles}/{result.n_triads}")
    print(f"  Hard cycle rate: {result.hard_cycle_rate:.4f}")
    if result.sampled:
        print("  (sampled)")

    # Reference: random preferences have cycle_prob = 0.25
    print()
    print(f"  Reference: random preferences would have cycle_prob = 0.25")
    print(f"  Your cycle_prob = {result.cycle_probability:.4f} -> {'more transitive' if result.cycle_probability < 0.25 else 'less transitive'} than random")


if __name__ == "__main__":
    main()
