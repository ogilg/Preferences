"""Sensitivity analysis for preference measurements.

Functions for computing correlations between preference measurements
across different experimental conditions (e.g., phrasing variations).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from scipy.stats import pearsonr

if TYPE_CHECKING:
    from src.task_data import Task
    from src.types import BinaryPreferenceMeasurement
    from src.preferences.ranking import ThurstonianResult
    from src.preferences.templates import PromptTemplate


def _build_win_rate_vector(
    measurements: list["BinaryPreferenceMeasurement"],
    tasks: list["Task"],
) -> np.ndarray:
    """Build a vector of win rates for upper-triangle pairs.

    Returns a flattened vector where each element is the win rate
    of task i over task j for all pairs i < j.
    """
    n = len(tasks)
    id_to_idx = {t.id: i for i, t in enumerate(tasks)}

    # Count wins and totals for each pair
    wins = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.float64)

    for m in measurements:
        i = id_to_idx[m.task_a.id]
        j = id_to_idx[m.task_b.id]
        # Normalize to always use smaller index first
        if i > j:
            i, j = j, i
            choice = "b" if m.choice == "a" else "a"
        else:
            choice = m.choice

        counts[i, j] += 1
        if choice == "a":
            wins[i, j] += 1

    # Compute win rates for upper triangle
    rates = []
    for i in range(n):
        for j in range(i + 1, n):
            if counts[i, j] > 0:
                rates.append(wins[i, j] / counts[i, j])
            else:
                rates.append(0.5)  # No data, assume 50/50

    return np.array(rates)


def win_rate_correlation(
    measurements_a: list["BinaryPreferenceMeasurement"],
    measurements_b: list["BinaryPreferenceMeasurement"],
    tasks: list["Task"],
) -> float:
    """Compute Pearson correlation of win rates between two measurement sets.

    For each unique task pair, computes the win rate (probability that the
    first task beats the second) from each measurement set, then computes
    the Pearson correlation between these win rate vectors.

    Args:
        measurements_a: First set of binary preference measurements.
        measurements_b: Second set of binary preference measurements.
        tasks: List of all tasks (defines the ordering).

    Returns:
        Pearson correlation coefficient. Returns 0.0 if correlation
        cannot be computed (e.g., not enough variance).
    """
    rates_a = _build_win_rate_vector(measurements_a, tasks)
    rates_b = _build_win_rate_vector(measurements_b, tasks)

    if len(rates_a) < 2:
        return 0.0

    # Check for zero variance
    if np.std(rates_a) < 1e-10 or np.std(rates_b) < 1e-10:
        return 0.0

    r, _ = pearsonr(rates_a, rates_b)
    return float(r) if not np.isnan(r) else 0.0


def utility_correlation(
    result_a: "ThurstonianResult",
    result_b: "ThurstonianResult",
) -> float:
    """Compute Pearson correlation of fitted utilities between two Thurstonian fits.

    Correlates the mu (utility mean) values across tasks. Both results must
    have tasks in the same order or with matching IDs.

    Args:
        result_a: First Thurstonian fit result.
        result_b: Second Thurstonian fit result.

    Returns:
        Pearson correlation coefficient. Returns 0.0 if correlation
        cannot be computed.

    Raises:
        ValueError: If the results have different tasks.
    """
    # Verify task IDs match
    ids_a = [t.id for t in result_a.tasks]
    ids_b = [t.id for t in result_b.tasks]

    if set(ids_a) != set(ids_b):
        raise ValueError("ThurstonianResult task IDs don't match")

    # If order differs, reorder result_b to match result_a
    if ids_a != ids_b:
        idx_map = {tid: i for i, tid in enumerate(ids_b)}
        reorder = [idx_map[tid] for tid in ids_a]
        mu_b = result_b.mu[reorder]
    else:
        mu_b = result_b.mu

    mu_a = result_a.mu

    if len(mu_a) < 2:
        return 0.0

    # Check for zero variance
    if np.std(mu_a) < 1e-10 or np.std(mu_b) < 1e-10:
        return 0.0

    r, _ = pearsonr(mu_a, mu_b)
    return float(r) if not np.isnan(r) else 0.0


def compute_pairwise_correlations(
    results: dict[str, tuple[list["BinaryPreferenceMeasurement"], "ThurstonianResult"]],
    tasks: list["Task"],
) -> list[dict]:
    """Compute correlations for all pairs of templates.

    Args:
        results: Dict mapping template/phrasing ID to (measurements, thurstonian_result).
        tasks: List of all tasks (for computing win rates).

    Returns:
        List of dicts, each with keys:
        - phrasing_a: First template ID
        - phrasing_b: Second template ID
        - win_rate_correlation: Pearson r of win rates
        - utility_correlation: Pearson r of Thurstonian utilities
    """
    correlations = []

    for (id_a, data_a), (id_b, data_b) in combinations(results.items(), 2):
        meas_a, thurs_a = data_a
        meas_b, thurs_b = data_b

        correlations.append({
            "phrasing_a": id_a,
            "phrasing_b": id_b,
            "win_rate_correlation": float(win_rate_correlation(meas_a, meas_b, tasks)),
            "utility_correlation": float(utility_correlation(thurs_a, thurs_b)),
        })

    return correlations


def save_correlations(correlations: list[dict], path: Path | str) -> None:
    """Save correlations to YAML.

    Args:
        correlations: List of correlation dicts from compute_pairwise_correlations.
        path: Path to save the YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Compute summary statistics
    win_rates = [c["win_rate_correlation"] for c in correlations]
    utilities = [c["utility_correlation"] for c in correlations]

    output = {
        "summary": {
            "mean_win_rate_correlation": float(np.mean(win_rates)),
            "mean_utility_correlation": float(np.mean(utilities)),
            "n_pairs": len(correlations),
        },
        "pairwise": correlations,
    }

    with open(path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)


def save_experiment_config(
    templates: list["PromptTemplate"],
    model_name: str,
    temperature: float,
    n_tasks: int,
    path: Path | str,
) -> None:
    """Save experiment configuration including template prompts.

    Args:
        templates: List of templates used in the experiment.
        model_name: Name of the model used.
        temperature: Temperature setting used.
        n_tasks: Number of tasks used.
        path: Path to save the config file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "model": model_name,
        "temperature": temperature,
        "n_tasks": n_tasks,
        "templates": [
            {
                "phrasing_id": t.tags_dict["phrasing"],
                "name": t.name,
                "prompt": t.template,
            }
            for t in templates
        ],
    }

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
