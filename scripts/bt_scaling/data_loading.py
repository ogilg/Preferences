"""Shared data loading for BT scaling experiments.

Loads pairwise measurements, activations, and Thurstonian scores.
Provides task-level k-fold CV splitting.
"""
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
ACTIVATIONS_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
MEASUREMENTS_JSON = Path("scripts/active_learning_calibration/measurements_fast.json")
LAYER = 31


def load_measurements() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load measurements as numpy arrays.

    Returns:
        task_a_idx, task_b_idx, winner_idx, task_id_list
    """
    with open(MEASUREMENTS_JSON) as f:
        raw = json.load(f)

    task_ids_set: set[str] = set()
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


def load_activations_layer(task_id_list: list[str]) -> np.ndarray:
    """Load layer 31 activations, aligned to task_id_list ordering.

    Returns (n_tasks, n_features) array.
    """
    data = np.load(ACTIVATIONS_PATH, allow_pickle=True)
    act_task_ids = list(data["task_ids"])
    act_id_to_idx = {tid: i for i, tid in enumerate(act_task_ids)}

    raw_acts = data[f"layer_{LAYER}"]
    n_tasks = len(task_id_list)
    n_features = raw_acts.shape[1]
    aligned = np.zeros((n_tasks, n_features), dtype=np.float32)

    for i, tid in enumerate(task_id_list):
        if tid in act_id_to_idx:
            aligned[i] = raw_acts[act_id_to_idx[tid]]

    return aligned


def load_thurstonian_scores(task_id_list: list[str]) -> np.ndarray:
    """Load full-data Thurstonian mu, aligned to task_id_list."""
    csv_path = RUN_DIR / "thurstonian_a1ebd06e.csv"
    scores: dict[str, float] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row["task_id"]] = float(row["mu"])

    mu = np.zeros(len(task_id_list))
    for i, tid in enumerate(task_id_list):
        mu[i] = scores[tid]
    return mu


def aggregate_pairs(
    task_a_idx: np.ndarray,
    task_b_idx: np.ndarray,
    winner_idx: np.ndarray,
    measurement_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate raw measurements into unique pairs with win counts.

    Returns:
        pairs: (n_pairs, 2) with pairs[:,0] < pairs[:,1]
        wins_i: wins for the lower-indexed task
        total: total comparisons per pair
    """
    if measurement_mask is not None:
        a = task_a_idx[measurement_mask]
        b = task_b_idx[measurement_mask]
        w = winner_idx[measurement_mask]
    else:
        a = task_a_idx
        b = task_b_idx
        w = winner_idx

    # Canonical ordering: lower index first
    low = np.minimum(a, b)
    high = np.maximum(a, b)

    # Use a dict for aggregation
    pair_counts: dict[tuple[int, int], list[int]] = {}
    for i in range(len(a)):
        key = (int(low[i]), int(high[i]))
        if key not in pair_counts:
            pair_counts[key] = [0, 0]  # [wins_low, total]
        pair_counts[key][1] += 1
        if w[i] == low[i]:
            pair_counts[key][0] += 1

    keys = sorted(pair_counts.keys())
    pairs = np.array(keys, dtype=np.int32)
    wins_i = np.array([pair_counts[k][0] for k in keys], dtype=np.float64)
    total = np.array([pair_counts[k][1] for k in keys], dtype=np.float64)

    return pairs, wins_i, total


def get_task_folds(n_tasks: int, n_folds: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create task-level k-fold splits.

    Returns list of (train_task_indices, test_task_indices).
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_tasks)))


def filter_pairs_by_tasks(
    pairs: np.ndarray,
    wins_i: np.ndarray,
    total: np.ndarray,
    task_set: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only pairs where both tasks are in task_set."""
    mask = np.array([
        int(pairs[i, 0]) in task_set and int(pairs[i, 1]) in task_set
        for i in range(len(pairs))
    ])
    return pairs[mask], wins_i[mask], total[mask]


def filter_measurements_by_tasks(
    task_a_idx: np.ndarray,
    task_b_idx: np.ndarray,
    winner_idx: np.ndarray,
    task_set: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only measurements where both tasks are in task_set."""
    mask = np.isin(task_a_idx, list(task_set)) & np.isin(task_b_idx, list(task_set))
    return task_a_idx[mask], task_b_idx[mask], winner_idx[mask]


def compute_win_rates(
    task_a_idx: np.ndarray,
    task_b_idx: np.ndarray,
    winner_idx: np.ndarray,
    n_tasks: int,
) -> np.ndarray:
    """Compute per-task win rate = wins / total_comparisons.

    Returns (n_tasks,) array.
    """
    wins = np.zeros(n_tasks)
    total = np.zeros(n_tasks)
    for i in range(len(task_a_idx)):
        a, b, w = task_a_idx[i], task_b_idx[i], winner_idx[i]
        total[a] += 1
        total[b] += 1
        wins[w] += 1

    # Avoid division by zero for tasks with no comparisons
    rate = np.zeros(n_tasks)
    observed = total > 0
    rate[observed] = wins[observed] / total[observed]
    return rate


def weighted_pairwise_accuracy(
    predicted_scores: np.ndarray,
    pairs: np.ndarray,
    wins_i: np.ndarray,
    total: np.ndarray,
) -> float:
    """Weighted pairwise accuracy: fraction of individual comparisons correctly predicted."""
    wins_j = total - wins_i
    scores_i = predicted_scores[pairs[:, 0]]
    scores_j = predicted_scores[pairs[:, 1]]
    logits = scores_i - scores_j
    correct = np.where(logits > 0, wins_i, wins_j)
    return float(np.sum(correct) / np.sum(total))
