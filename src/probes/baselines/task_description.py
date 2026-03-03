"""Task description baseline for probe benchmarking."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.probes.core.activations import load_activations
from src.probes.core.linear_probe import train_and_evaluate

from .types import BaselineResult, BaselineType


def run_task_description_baseline(
    task_description_dir: Path,
    scores: dict[str, float],
    layer: int,
    cv_folds: int,
) -> BaselineResult | None:
    """Train probe on task description activations."""
    task_ids, activations = load_activations(task_description_dir)

    if layer not in activations:
        return None

    # Match task_ids to scores
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    valid_indices = []
    valid_scores = []
    for task_id, score in scores.items():
        if task_id in id_to_idx:
            valid_indices.append(id_to_idx[task_id])
            valid_scores.append(score)

    if len(valid_indices) < cv_folds * 2:
        return None

    indices = np.array(valid_indices)
    y = np.array(valid_scores)
    X = activations[layer][indices]

    _, result, _ = train_and_evaluate(X, y, cv_folds=cv_folds)

    return BaselineResult.from_cv_result(
        result, BaselineType.TASK_DESCRIPTION, layer, len(y),
    )
