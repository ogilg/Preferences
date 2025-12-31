from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np

from src.experiments.correlation import safe_correlation, save_correlations_yaml
from src.task_data import Task
from src.types import TaskScore


def _build_score_map(scores: list[TaskScore]) -> dict[str, float]:
    """Build task_id -> mean score map, averaging repeated samples."""
    id_to_scores: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        id_to_scores[s.task.id].append(s.score)
    return {tid: float(np.mean(vals)) for tid, vals in id_to_scores.items()}


def compute_per_task_std(scores: list[TaskScore]) -> dict[str, float]:
    """Compute std of scores across repeated samples for each task."""
    id_to_scores: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        id_to_scores[s.task.id].append(s.score)
    return {tid: float(np.std(vals)) for tid, vals in id_to_scores.items()}


def compute_mean_std_across_tasks(scores: list[TaskScore]) -> float:
    """Average std across all tasks (summary metric for sample consistency)."""
    per_task = compute_per_task_std(scores)
    return float(np.mean(list(per_task.values())))


def score_correlation(
    scores_a: list[TaskScore],
    scores_b: list[TaskScore],
    tasks: list[Task],
    method: Literal["pearson", "spearman"] = "pearson",
) -> float:
    map_a = _build_score_map(scores_a)
    map_b = _build_score_map(scores_b)
    common_ids = [t.id for t in tasks if t.id in map_a and t.id in map_b]
    vec_a = np.array([map_a[tid] for tid in common_ids])
    vec_b = np.array([map_b[tid] for tid in common_ids])
    return safe_correlation(vec_a, vec_b, method)


def compute_rating_pairwise_correlations(
    results: dict[str, list[TaskScore]],
    tasks: list[Task],
) -> list[dict]:
    correlations = []

    for (id_a, scores_a), (id_b, scores_b) in combinations(results.items(), 2):
        correlations.append({
            "template_a": id_a,
            "template_b": id_b,
            "pearson_correlation": score_correlation(scores_a, scores_b, tasks, "pearson"),
            "spearman_correlation": score_correlation(scores_a, scores_b, tasks, "spearman"),
        })

    return correlations


def save_rating_correlations(correlations: list[dict], path: Path | str) -> None:
    save_correlations_yaml(
        correlations,
        summary_keys=["pearson_correlation", "spearman_correlation"],
        path=path,
    )
