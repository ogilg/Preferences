from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr

from src.sensitivity_experiments.binary_correlation import save_experiment_config

if TYPE_CHECKING:
    from src.task_data import Task
    from src.types import TaskScore


def _build_score_vector(
    scores: list["TaskScore"],
    tasks: list["Task"],
) -> np.ndarray:
    id_to_score = {s.task.id: s.score for s in scores}
    return np.array([id_to_score[t.id] for t in tasks])


def score_correlation(
    scores_a: list["TaskScore"],
    scores_b: list["TaskScore"],
    tasks: list["Task"],
    method: Literal["pearson", "spearman"] = "pearson",
) -> float:
    """Returns 0.0 if insufficient variance."""
    vec_a = _build_score_vector(scores_a, tasks)
    vec_b = _build_score_vector(scores_b, tasks)

    if len(vec_a) < 2:
        return 0.0

    if np.std(vec_a) < 1e-10 or np.std(vec_b) < 1e-10:
        return 0.0

    corr_fn = pearsonr if method == "pearson" else spearmanr
    r, _ = corr_fn(vec_a, vec_b)
    return float(r) if not np.isnan(r) else 0.0


def compute_rating_pairwise_correlations(
    results: dict[str, list["TaskScore"]],
    tasks: list["Task"],
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pearson_vals = [c["pearson_correlation"] for c in correlations]
    spearman_vals = [c["spearman_correlation"] for c in correlations]

    output = {
        "summary": {
            "mean_pearson_correlation": float(np.mean(pearson_vals)) if pearson_vals else 0.0,
            "mean_spearman_correlation": float(np.mean(spearman_vals)) if spearman_vals else 0.0,
            "n_pairs": len(correlations),
        },
        "pairwise": correlations,
    }

    with open(path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)


__all__ = [
    "score_correlation",
    "compute_rating_pairwise_correlations",
    "save_rating_correlations",
    "save_experiment_config",
]
