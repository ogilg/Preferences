from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from src.experiments.correlation import safe_correlation, save_correlations_yaml
from src.preferences.ranking import ThurstonianResult
from src.task_data import Task
from src.types import BinaryPreferenceMeasurement


def _build_win_rate_vector(
    measurements: list[BinaryPreferenceMeasurement],
    tasks: list[Task],
) -> np.ndarray:
    """Win rate of task i over task j for all pairs i < j, flattened."""
    n = len(tasks)
    id_to_idx = {t.id: i for i, t in enumerate(tasks)}

    wins = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.float64)

    for m in measurements:
        i = id_to_idx[m.task_a.id]
        j = id_to_idx[m.task_b.id]
        if i > j:
            i, j = j, i
            choice = "b" if m.choice == "a" else "a"
        else:
            choice = m.choice

        counts[i, j] += 1
        if choice == "a":
            wins[i, j] += 1

    rates = []
    for i in range(n):
        for j in range(i + 1, n):
            if counts[i, j] > 0:
                rates.append(wins[i, j] / counts[i, j])
            else:
                rates.append(0.5)

    return np.array(rates)


def win_rate_correlation(
    measurements_a: list[BinaryPreferenceMeasurement],
    measurements_b: list[BinaryPreferenceMeasurement],
    tasks: list[Task],
) -> float:
    """Pearson correlation of win rates."""
    rates_a = _build_win_rate_vector(measurements_a, tasks)
    rates_b = _build_win_rate_vector(measurements_b, tasks)
    return safe_correlation(rates_a, rates_b, "pearson")


def utility_correlation(
    result_a: ThurstonianResult,
    result_b: ThurstonianResult,
) -> float:
    """Pearson correlation of fitted utilities. Raises ValueError if task IDs differ."""
    ids_a = [t.id for t in result_a.tasks]
    ids_b = [t.id for t in result_b.tasks]

    if set(ids_a) != set(ids_b):
        raise ValueError("ThurstonianResult task IDs don't match")

    if ids_a != ids_b:
        idx_map = {tid: i for i, tid in enumerate(ids_b)}
        reorder = [idx_map[tid] for tid in ids_a]
        mu_b = result_b.mu[reorder]
    else:
        mu_b = result_b.mu

    return safe_correlation(result_a.mu, mu_b, "pearson")


def compute_pairwise_correlations(
    results: dict[str, tuple[list[BinaryPreferenceMeasurement], ThurstonianResult]],
    tasks: list[Task],
) -> list[dict]:
    correlations = []

    for (id_a, data_a), (id_b, data_b) in combinations(results.items(), 2):
        meas_a, thurs_a = data_a
        meas_b, thurs_b = data_b

        correlations.append({
            "template_a": id_a,
            "template_b": id_b,
            "win_rate_correlation": float(win_rate_correlation(meas_a, meas_b, tasks)),
            "utility_correlation": float(utility_correlation(thurs_a, thurs_b)),
        })

    return correlations


def save_correlations(correlations: list[dict], path: Path | str) -> None:
    save_correlations_yaml(
        correlations,
        summary_keys=["win_rate_correlation", "utility_correlation"],
        path=path,
    )
