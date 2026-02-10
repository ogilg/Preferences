from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize

from .data import PairwiseActivationData


@dataclass
class BTResult:
    weights: np.ndarray  # [coef..., 0.0] — no intercept, trailing zero for format compat
    layer: int
    train_accuracy: float
    train_loss: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    best_l2_lambda: float
    n_iterations: int
    lambda_sweep: list[dict]


def _bt_loss_and_grad(
    w: np.ndarray,
    acts: np.ndarray,
    winner_idx: np.ndarray,
    loser_idx: np.ndarray,
    l2_lambda: float,
    n_pairs: int,
) -> tuple[float, np.ndarray]:
    # Compute per-task scores once (n_tasks,), then index to get pair logits (n_pairs,)
    # No intercept: pairs are always (winner, loser), so a bias just inflates logits
    task_scores = acts @ w
    logits = task_scores[winner_idx] - task_scores[loser_idx]

    loss = np.mean(np.logaddexp(0, -logits)) + 0.5 * l2_lambda * np.sum(w ** 2)

    probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
    pair_grad = (probs - 1) / n_pairs

    # Scatter-add pair gradients back to per-task gradients
    n_tasks = len(acts)
    task_grad = (
        np.bincount(winner_idx, weights=pair_grad, minlength=n_tasks)
        - np.bincount(loser_idx, weights=pair_grad, minlength=n_tasks)
    )

    grad_w = acts.T @ task_grad + l2_lambda * w

    return loss, grad_w


def _fit_bt(
    acts: np.ndarray,
    winner_idx: np.ndarray,
    loser_idx: np.ndarray,
    l2_lambda: float,
    maxiter: int = 500,
) -> tuple[np.ndarray, int]:
    """Fit BT weights via L-BFGS-B. Returns (weights, n_iterations)."""
    n_pairs = len(winner_idx)
    w0 = np.zeros(acts.shape[1])
    n_iterations = 0

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal n_iterations
        n_iterations += 1
        return _bt_loss_and_grad(w, acts, winner_idx, loser_idx, l2_lambda, n_pairs)

    result = minimize(objective, w0, method="L-BFGS-B", jac=True, options={"maxiter": maxiter})
    return result.x, n_iterations


def _pairwise_accuracy(
    w: np.ndarray,
    acts: np.ndarray,
    winner_idx: np.ndarray,
    loser_idx: np.ndarray,
) -> float:
    task_scores = acts @ w
    logits = task_scores[winner_idx] - task_scores[loser_idx]
    return float((logits > 0).mean())


def _sweep_one_lambda(
    acts: np.ndarray,
    train_winner: np.ndarray,
    train_loser: np.ndarray,
    val_winner: np.ndarray,
    val_loser: np.ndarray,
    l2: float,
) -> dict:
    w, _ = _fit_bt(acts, train_winner, train_loser, l2)
    train_acc = _pairwise_accuracy(w, acts, train_winner, train_loser)
    val_acc = _pairwise_accuracy(w, acts, val_winner, val_loser)
    return {"l2_lambda": float(l2), "train_accuracy": train_acc, "val_accuracy": val_acc}


def train_bt(
    data: PairwiseActivationData,
    layer: int,
    lambdas: np.ndarray | None = None,
    val_fraction: float = 0.2,
    rng: np.random.Generator | None = None,
    n_jobs: int = 1,
) -> BTResult:
    if lambdas is None:
        lambdas = np.logspace(-1, 3, 5)
    if rng is None:
        rng = np.random.default_rng(0)

    acts = data.activations[layer]
    winner_idx = data.pairs[:, 0]
    loser_idx = data.pairs[:, 1]
    n_pairs = len(data.pairs)

    # Single train/val split on pairs
    val_mask = rng.random(n_pairs) < val_fraction
    train_mask = ~val_mask

    train_winner, train_loser = winner_idx[train_mask], loser_idx[train_mask]
    val_winner, val_loser = winner_idx[val_mask], loser_idx[val_mask]

    # Lambda sweep
    if n_jobs == 1:
        lambda_sweep = [
            _sweep_one_lambda(acts, train_winner, train_loser, val_winner, val_loser, l2)
            for l2 in lambdas
        ]
    else:
        lambda_sweep = Parallel(n_jobs=n_jobs)(
            delayed(_sweep_one_lambda)(acts, train_winner, train_loser, val_winner, val_loser, l2)
            for l2 in lambdas
        )
    for entry in lambda_sweep:
        print(f"    l2={entry['l2_lambda']:.4g}: train_acc={entry['train_accuracy']:.4f}, val_acc={entry['val_accuracy']:.4f}")

    best_entry = max(lambda_sweep, key=lambda x: x["val_accuracy"])
    best_l2 = best_entry["l2_lambda"]

    # Retrain on all data with best lambda
    w, n_iterations = _fit_bt(acts, winner_idx, loser_idx, best_l2)
    train_acc = _pairwise_accuracy(w, acts, winner_idx, loser_idx)
    task_scores = acts @ w
    train_loss = float(np.mean(np.logaddexp(0, -task_scores[winner_idx] + task_scores[loser_idx])))

    return BTResult(
        weights=np.append(w, 0.0),
        layer=layer,
        train_accuracy=train_acc,
        train_loss=train_loss,
        cv_accuracy_mean=best_entry["val_accuracy"],
        cv_accuracy_std=0.0,
        best_l2_lambda=best_l2,
        n_iterations=n_iterations,
        lambda_sweep=lambda_sweep,
    )


def pairwise_accuracy_from_scores(
    scores: dict[str, float],
    data: PairwiseActivationData,
    task_ids: np.ndarray,
) -> float:
    """Compute pairwise accuracy of scalar scores on BT pairs.

    Use this to evaluate Ridge probe scores on pairwise data for comparison with BT.
    """
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    score_arr = np.array([scores.get(tid, np.nan) for tid in task_ids])

    winner_scores = score_arr[data.pairs[:, 0]]
    loser_scores = score_arr[data.pairs[:, 1]]

    valid = np.isfinite(winner_scores) & np.isfinite(loser_scores)
    return float((winner_scores[valid] > loser_scores[valid]).mean())


def train_for_comparisons(
    data: PairwiseActivationData,
    lambdas: np.ndarray | None = None,
    val_fraction: float = 0.2,
    n_jobs: int = 1,
) -> tuple[list[BTResult], dict[int, np.ndarray]]:
    if len(data.pairs) == 0:
        return [], {}

    results = []
    probes = {}

    for layer in sorted(data.activations.keys()):
        result = train_bt(data, layer, lambdas=lambdas, val_fraction=val_fraction, n_jobs=n_jobs)
        results.append(result)
        probes[layer] = result.weights

    return results, probes
