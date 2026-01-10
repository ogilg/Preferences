"""Shared utilities for Thurstonian analysis scripts."""

import numpy as np
from scipy.special import ndtr


def split_wins(
    wins: np.ndarray,
    test_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out entire pairs (edges) for test. Returns (train_wins, test_wins)."""
    n = wins.shape[0]
    train = wins.copy()
    test = np.zeros_like(wins)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if wins[i, j] + wins[j, i] > 0]

    n_test_pairs = int(len(pairs) * test_frac)
    test_pair_indices = rng.choice(len(pairs), size=n_test_pairs, replace=False)

    for idx in test_pair_indices:
        i, j = pairs[idx]
        test[i, j] = wins[i, j]
        test[j, i] = wins[j, i]
        train[i, j] = 0
        train[j, i] = 0

    return train, test


def eval_held_out_nll(
    mu: np.ndarray,
    sigma: np.ndarray,
    wins: np.ndarray,
) -> float:
    """Compute NLL on held-out pairs. Returns NLL per comparison."""
    n_comparisons = int(wins.sum())
    if n_comparisons == 0:
        return 0.0

    mu_diff = mu[:, np.newaxis] - mu[np.newaxis, :]
    scale = np.sqrt(sigma[:, np.newaxis] ** 2 + sigma[np.newaxis, :] ** 2)
    p = ndtr(mu_diff / scale)
    p = np.clip(p, 1e-10, 1 - 1e-10)

    total_nll = -float(np.sum(wins * np.log(p)))
    return total_nll / n_comparisons


def eval_held_out_accuracy(
    mu: np.ndarray,
    sigma: np.ndarray,
    wins: np.ndarray,
) -> float:
    """Compute prediction accuracy on held-out pairs."""
    n = wins.shape[0]
    correct = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            n_ij = wins[i, j] + wins[j, i]
            if n_ij == 0:
                continue

            # Model predicts i > j if mu[i] > mu[j]
            pred_i_wins = mu[i] > mu[j]
            # Empirical: i wins more often
            emp_i_wins = wins[i, j] > wins[j, i]

            if pred_i_wins == emp_i_wins:
                correct += 1
            total += 1

    return correct / total if total > 0 else 1.0
