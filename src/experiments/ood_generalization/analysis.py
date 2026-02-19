"""Delta computation and correlation utilities for OOD generalization experiments."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy import stats


def _split_probe(probe_path: Path) -> tuple[np.ndarray, float]:
    """Load probe .npy and split into (weights, bias)."""
    probe = np.load(probe_path)
    return probe[:-1], float(probe[-1])


def _score_activations(
    npz_path: Path,
    layer: int,
    weights: np.ndarray,
    bias: float,
) -> tuple[np.ndarray, list[str]]:
    """Load activations from npz, score with probe, return (scores, task_ids)."""
    data = np.load(npz_path, allow_pickle=True)
    acts = data[f"layer_{layer}"]
    scores = acts @ weights + bias
    task_ids = list(data["task_ids"])
    return scores, task_ids


def compute_deltas(
    conditions: dict,
    activations_dir: Path,
    probe_path: Path,
    layer: int,
    task_ids: list[str],
    baseline_key: str = "baseline",
    baseline_activations_key: str = "neutral",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute behavioral and probe deltas for all conditions vs baseline.

    Args:
        conditions: The "conditions" dict from behavioral.json — keys are
            condition IDs, values have "task_rates" with per-task p_choose.
        activations_dir: Dir with {condition_id}/activations_prompt_last.npz.
        probe_path: Path to probe .npy file.
        layer: Layer number (e.g. 31).
        task_ids: Task IDs to include.
        baseline_key: Key for the baseline condition in `conditions`.
        baseline_activations_key: Subdirectory name for baseline activations.

    Returns:
        (behavioral_deltas, probe_deltas, condition_labels) — arrays pooled
        across all non-baseline conditions × tasks.
    """
    weights, bias = _split_probe(probe_path)

    baseline_npz = activations_dir / baseline_activations_key / "activations_prompt_last.npz"
    baseline_scores, baseline_task_ids = _score_activations(baseline_npz, layer, weights, bias)
    baseline_idx = {tid: i for i, tid in enumerate(baseline_task_ids)}

    baseline_rates = conditions[baseline_key]["task_rates"]

    all_behavioral = []
    all_probe = []
    all_labels = []

    condition_ids = [k for k in conditions if k != baseline_key]
    for cid in condition_ids:
        cond_rates = conditions[cid]["task_rates"]

        cond_npz = activations_dir / cid / "activations_prompt_last.npz"
        if not cond_npz.exists():
            warnings.warn(f"Missing activations for condition '{cid}': {cond_npz}")
            continue
        cond_scores, cond_task_ids = _score_activations(cond_npz, layer, weights, bias)
        cond_idx = {tid: i for i, tid in enumerate(cond_task_ids)}

        for tid in task_ids:
            if tid not in baseline_rates or tid not in cond_rates:
                continue
            if tid not in baseline_idx or tid not in cond_idx:
                continue

            b_delta = cond_rates[tid]["p_choose"] - baseline_rates[tid]["p_choose"]
            p_delta = float(cond_scores[cond_idx[tid]] - baseline_scores[baseline_idx[tid]])

            all_behavioral.append(b_delta)
            all_probe.append(p_delta)
            all_labels.append(cid)

    return np.array(all_behavioral), np.array(all_probe), np.array(all_labels)


def correlate_deltas(
    behavioral_deltas: np.ndarray,
    probe_deltas: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Pearson r, sign agreement, permutation test."""
    r, p = stats.pearsonr(behavioral_deltas, probe_deltas)
    spearman_r, spearman_p = stats.spearmanr(behavioral_deltas, probe_deltas)

    # Sign agreement (exclude near-zero behavioral deltas)
    threshold = 0.02
    mask = np.abs(behavioral_deltas) >= threshold
    if mask.sum() > 0:
        sign_agree = float((np.sign(behavioral_deltas[mask]) == np.sign(probe_deltas[mask])).mean())
        sign_n = int(mask.sum())
    else:
        sign_agree = float("nan")
        sign_n = 0

    # Permutation test
    rng = np.random.RandomState(seed)
    perm_rs = np.array([
        stats.pearsonr(behavioral_deltas[rng.permutation(len(behavioral_deltas))], probe_deltas)[0]
        for _ in range(n_permutations)
    ])
    perm_p = float((perm_rs >= r).mean())

    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "sign_agreement": sign_agree,
        "sign_n": sign_n,
        "n": len(behavioral_deltas),
        "permutation_p": perm_p,
        "permutation_mean_r": float(perm_rs.mean()),
    }


def per_condition_correlations(
    behavioral_deltas: np.ndarray,
    probe_deltas: np.ndarray,
    condition_labels: np.ndarray,
    min_n: int = 5,
) -> dict[str, dict]:
    """Per-condition Pearson r breakdown."""
    results = {}
    for cid in np.unique(condition_labels):
        mask = condition_labels == cid
        if mask.sum() < min_n:
            continue
        b = behavioral_deltas[mask]
        p = probe_deltas[mask]
        r, pval = stats.pearsonr(b, p)
        results[cid] = {
            "pearson_r": float(r),
            "pearson_p": float(pval),
            "n": int(mask.sum()),
        }
    return results
