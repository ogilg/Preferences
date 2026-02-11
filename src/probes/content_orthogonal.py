"""Residualize activations and scores against content embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def _align_by_task_id(
    task_ids_a: np.ndarray,
    task_ids_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return index arrays (into a, into b) for shared task IDs, preserving a's order."""
    b_lookup = {tid: i for i, tid in enumerate(task_ids_b)}
    a_indices = []
    b_indices = []
    for i, tid in enumerate(task_ids_a):
        if tid in b_lookup:
            a_indices.append(i)
            b_indices.append(b_lookup[tid])
    return np.array(a_indices), np.array(b_indices)


def residualize_activations(
    activations: np.ndarray,
    task_ids: np.ndarray,
    content_embeddings: np.ndarray,
    content_task_ids: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Regress out content-predictable variance from activations via Ridge.

    Returns (aligned_task_ids, residual_activations, stats).
    """
    act_idx, emb_idx = _align_by_task_id(task_ids, content_task_ids)
    n_aligned = len(act_idx)
    if n_aligned == 0:
        raise ValueError("No overlapping task IDs between activations and content embeddings")

    aligned_task_ids = task_ids[act_idx]
    X = content_embeddings[emb_idx]
    Y = activations[act_idx]

    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X, Y)
    predicted = ridge.predict(X)
    residuals = Y - predicted

    # R² = 1 - SS_res / SS_tot (averaged across activation dimensions)
    ss_res = np.sum((Y - predicted) ** 2)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
    content_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    stats = {
        "content_r2": round(float(content_r2), 4),
        "alpha": alpha,
        "n_tasks": n_aligned,
        "d_embed": X.shape[1],
        "d_model": Y.shape[1],
    }

    return aligned_task_ids, residuals, stats


def residualize_scores_continuous(
    scores: dict[str, float],
    content_embeddings: np.ndarray,
    content_task_ids: np.ndarray,
    alpha: float = 1.0,
) -> tuple[dict[str, float], dict]:
    """Regress out content-predictable variance from preference scores."""
    emb_lookup = {tid: i for i, tid in enumerate(content_task_ids)}

    aligned_tids = []
    aligned_scores = []
    aligned_emb_indices = []
    for tid, score in scores.items():
        if tid in emb_lookup:
            aligned_tids.append(tid)
            aligned_scores.append(score)
            aligned_emb_indices.append(emb_lookup[tid])

    if not aligned_tids:
        raise ValueError("No overlapping task IDs between scores and content embeddings")

    X = content_embeddings[aligned_emb_indices]
    y = np.array(aligned_scores)

    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X, y)
    predicted = ridge.predict(X)
    residuals = y - predicted

    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    content_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    residual_scores = dict(zip(aligned_tids, residuals.tolist()))

    stats = {
        "content_r2": round(float(content_r2), 4),
        "alpha": alpha,
        "n_tasks": len(aligned_tids),
        "d_embed": X.shape[1],
    }

    return residual_scores, stats
