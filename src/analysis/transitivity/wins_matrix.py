"""Build wins matrices from different preference measurement types.

Wins matrix W[i,j] = number of times task i was preferred over task j.

For stated/qualitative preferences, we derive pairwise preferences from scores:
  - If score_a > score_b: a wins
  - If score_a < score_b: b wins
  - If score_a == score_b: tie (no win recorded)

For revealed preferences, we use direct pairwise comparison data.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from src.measurement_storage import (
    EXPERIMENTS_DIR,
    list_runs,
    load_yaml,
    RunConfig,
)


def wins_from_scores(scores: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    """Build wins matrix from task scores (stated/qualitative preferences).

    For each pair, the task with higher score wins. Ties add no wins.
    """
    task_ids = sorted(scores.keys())
    n = len(task_ids)
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    wins = np.zeros((n, n), dtype=np.int32)

    for i, tid_i in enumerate(task_ids):
        for j, tid_j in enumerate(task_ids):
            if i >= j:
                continue
            score_i = scores[tid_i]
            score_j = scores[tid_j]
            if score_i > score_j:
                wins[i, j] += 1
            elif score_j > score_i:
                wins[j, i] += 1
            # Ties: no win recorded

    return wins, task_ids


def wins_from_pairwise(
    comparisons: list[dict],
) -> tuple[np.ndarray, list[str]]:
    """Build wins matrix from pairwise comparison data.

    Each comparison dict has: task_a, task_b, choice ('a' or 'b').
    """
    task_ids_set: set[str] = set()
    for m in comparisons:
        task_ids_set.add(m["task_a"])
        task_ids_set.add(m["task_b"])

    task_ids = sorted(task_ids_set)
    n = len(task_ids)
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    wins = np.zeros((n, n), dtype=np.int32)

    for m in comparisons:
        i = id_to_idx[m["task_a"]]
        j = id_to_idx[m["task_b"]]
        if m["choice"] == "a":
            wins[i, j] += 1
        else:
            wins[j, i] += 1

    return wins, task_ids


def load_stated_scores(run_dir: Path) -> dict[str, float]:
    """Load scores from a stated/qualitative measurement run."""
    measurements_path = run_dir / "measurements.yaml"
    measurements = load_yaml(measurements_path)

    # Aggregate multiple samples per task (if any)
    by_task: dict[str, list[float]] = defaultdict(list)
    for m in measurements:
        by_task[m["task_id"]].append(m["score"])

    return {tid: float(np.mean(scores)) for tid, scores in by_task.items()}


def load_pairwise_comparisons(run_dir: Path) -> list[dict]:
    """Load pairwise comparisons from a revealed preference run."""
    measurements_path = run_dir / "measurements.yaml"
    return load_yaml(measurements_path)


def load_wins_matrix_for_run(run_dir: Path, is_revealed: bool) -> tuple[np.ndarray, list[str]]:
    """Load wins matrix from a run directory."""
    if is_revealed:
        comparisons = load_pairwise_comparisons(run_dir)
        return wins_from_pairwise(comparisons)
    else:
        scores = load_stated_scores(run_dir)
        return wins_from_scores(scores)


def aggregate_wins_matrices(
    matrices: list[tuple[np.ndarray, list[str]]],
) -> tuple[np.ndarray, list[str]]:
    """Aggregate multiple wins matrices over the same tasks.

    Sums wins across all matrices. Task sets must be identical.
    """
    if not matrices:
        return np.zeros((0, 0), dtype=np.int32), []

    # Use first matrix's task_ids as reference
    _, ref_task_ids = matrices[0]
    ref_set = set(ref_task_ids)

    # Find common tasks across all matrices
    common_tasks = ref_set
    for _, task_ids in matrices[1:]:
        common_tasks &= set(task_ids)

    if not common_tasks:
        return np.zeros((0, 0), dtype=np.int32), []

    task_ids = sorted(common_tasks)
    n = len(task_ids)
    aggregated = np.zeros((n, n), dtype=np.int32)

    for wins, mat_task_ids in matrices:
        mat_id_to_idx = {tid: i for i, tid in enumerate(mat_task_ids)}
        for i, tid_i in enumerate(task_ids):
            for j, tid_j in enumerate(task_ids):
                if tid_i in mat_id_to_idx and tid_j in mat_id_to_idx:
                    aggregated[i, j] += wins[mat_id_to_idx[tid_i], mat_id_to_idx[tid_j]]

    return aggregated, task_ids
