from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np

from src.experiments.correlation import safe_correlation, save_correlations_yaml
from src.task_data import Task
from src.types import TaskScore


def _build_score_vector(
    scores: list[TaskScore],
    tasks: list[Task],
) -> np.ndarray:
    id_to_score = {s.task.id: s.score for s in scores}
    return np.array([id_to_score[t.id] for t in tasks])


def score_correlation(
    scores_a: list[TaskScore],
    scores_b: list[TaskScore],
    tasks: list[Task],
    method: Literal["pearson", "spearman"] = "pearson",
) -> float:
    vec_a = _build_score_vector(scores_a, tasks)
    vec_b = _build_score_vector(scores_b, tasks)
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
