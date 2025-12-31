"""Sensitivity analysis for rating measurements.

Functions for computing correlations between rating measurements
across different experimental conditions (e.g., phrasing variations).
"""

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
    """Build a score vector aligned to task order.

    Returns a vector where each element is the score for the corresponding task.
    """
    id_to_score = {s.task.id: s.score for s in scores}
    return np.array([id_to_score[t.id] for t in tasks])


def score_correlation(
    scores_a: list["TaskScore"],
    scores_b: list["TaskScore"],
    tasks: list["Task"],
    method: Literal["pearson", "spearman"] = "pearson",
) -> float:
    """Compute correlation between two score sets.

    Args:
        scores_a: First set of task scores.
        scores_b: Second set of task scores.
        tasks: List of all tasks (defines the ordering).
        method: Correlation method ('pearson' or 'spearman').

    Returns:
        Correlation coefficient. Returns 0.0 if correlation
        cannot be computed (e.g., not enough variance).
    """
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
    """Compute correlations for all pairs of templates.

    Args:
        results: Dict mapping template name to list of TaskScores.
        tasks: List of all tasks (for ordering).

    Returns:
        List of dicts with keys: template_a, template_b,
        pearson_correlation, spearman_correlation.
    """
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
    """Save rating correlations to YAML with summary stats."""
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
