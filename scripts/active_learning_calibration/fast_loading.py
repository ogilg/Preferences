"""Fast loading utilities using numpy arrays instead of Python objects."""

import csv
import json
from pathlib import Path

import numpy as np

RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
MEASUREMENTS_JSON = Path("scripts/active_learning_calibration/measurements_fast.json")


def load_measurements_as_arrays() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load measurements as numpy arrays.

    Returns:
        task_a_indices: (n_measurements,) int array
        task_b_indices: (n_measurements,) int array — index of winner (a_idx if choice=='a', b_idx otherwise)
        task_id_list: ordered list of task IDs (indices correspond to the arrays)
    """
    with open(MEASUREMENTS_JSON) as f:
        raw = json.load(f)

    # Build task ID list
    task_ids_set = set()
    for m in raw:
        task_ids_set.add(m["a"])
        task_ids_set.add(m["b"])
    task_id_list = sorted(task_ids_set)
    id_to_idx = {tid: i for i, tid in enumerate(task_id_list)}

    n = len(raw)
    task_a_idx = np.empty(n, dtype=np.int32)
    task_b_idx = np.empty(n, dtype=np.int32)
    winner_idx = np.empty(n, dtype=np.int32)

    for i, m in enumerate(raw):
        a = id_to_idx[m["a"]]
        b = id_to_idx[m["b"]]
        task_a_idx[i] = a
        task_b_idx[i] = b
        winner_idx[i] = a if m["c"] == "a" else b

    return task_a_idx, task_b_idx, winner_idx, task_id_list


def build_wins_matrix(task_a_idx, task_b_idx, winner_idx, n_tasks, measurement_mask=None):
    """Build wins matrix from measurement arrays.

    Args:
        measurement_mask: optional boolean mask to select a subset of measurements
    """
    if measurement_mask is not None:
        a = task_a_idx[measurement_mask]
        b = task_b_idx[measurement_mask]
        w = winner_idx[measurement_mask]
    else:
        a = task_a_idx
        b = task_b_idx
        w = winner_idx

    wins = np.zeros((n_tasks, n_tasks), dtype=np.int32)
    # winner_idx tells us who won
    for i in range(len(a)):
        loser = b[i] if w[i] == a[i] else a[i]
        wins[w[i], loser] += 1

    return wins


def load_full_thurstonian_scores() -> dict[str, float]:
    """Load the full-data Thurstonian scores from CSV."""
    csv_path = RUN_DIR / "thurstonian_a1ebd06e.csv"
    scores = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row["task_id"]] = float(row["mu"])
    return scores


def get_pair_indices(task_a_idx, task_b_idx) -> np.ndarray:
    """Get unique pair indices from measurement arrays.

    Returns (n_unique_pairs, 2) array where each row is (min_idx, max_idx).
    """
    pairs = np.stack([
        np.minimum(task_a_idx, task_b_idx),
        np.maximum(task_a_idx, task_b_idx),
    ], axis=1)
    return np.unique(pairs, axis=0)
