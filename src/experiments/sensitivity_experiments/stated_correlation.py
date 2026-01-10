from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from src.experiments.correlation import save_correlations_yaml
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


def scores_to_vector(scores: list[TaskScore], tasks: list[Task]) -> tuple[np.ndarray, list[str]]:
    """Convert TaskScore list to (values, task_ids) for use with compute_pairwise_correlations."""
    score_map = _build_score_map(scores)
    task_ids = [t.id for t in tasks if t.id in score_map]
    values = np.array([score_map[tid] for tid in task_ids])
    return values, task_ids


def save_stated_correlations(correlations: list[dict], path: Path | str) -> None:
    save_correlations_yaml(
        correlations,
        summary_keys=["pearson_correlation", "spearman_correlation"],
        path=path,
    )
