"""Probe evaluation and comparison utilities."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error

from src.probes.storage import load_probe, load_manifest
from src.probes.activations import load_activations
from src.measurement_storage.loading import load_pooled_scores, load_run_utilities


def evaluate_probe_on_data(
    probe_weights: np.ndarray,
    activations: np.ndarray,
    scores: np.ndarray,
    task_ids_data: np.ndarray,
    task_ids_scores: list[str],
) -> dict:
    """Evaluate probe on given activations and scores.

    Args:
        probe_weights: probe weights [coef_1, ..., coef_n, intercept]
        activations: activation matrix (n_samples, n_features)
        scores: score vector (n_samples,)
        task_ids_data: task IDs corresponding to activations
        task_ids_scores: task IDs corresponding to scores

    Returns:
        dict with r2, mse, pearson_r, n_samples, predictions
    """
    coef = probe_weights[:-1]
    intercept = probe_weights[-1]

    # Match activations to scores by task ID
    id_to_idx_data = {tid: i for i, tid in enumerate(task_ids_data)}
    valid_indices = []
    valid_scores = []
    for task_id, score in zip(task_ids_scores, scores):
        if task_id in id_to_idx_data:
            valid_indices.append(id_to_idx_data[task_id])
            valid_scores.append(score)

    if len(valid_indices) < 10:  # minimum samples for evaluation
        return {
            "r2": None,
            "mse": None,
            "pearson_r": None,
            "n_samples": len(valid_indices),
            "predictions": None,
        }

    indices = np.array(valid_indices)
    y = np.array(valid_scores)
    X_eval = activations[indices]

    # Predict
    y_pred = X_eval @ coef + intercept

    # Compute metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    pearson_r, _ = pearsonr(y, y_pred)

    return {
        "r2": float(r2),
        "mse": float(mse),
        "pearson_r": float(pearson_r),
        "n_samples": len(y),
        "predictions": y_pred.tolist(),
    }


def evaluate_probe_on_template(
    manifest_dir: Path,
    probe_id: str,
    eval_template: str,
    activations_dir: Path,
    experiment_dir: Path,
) -> dict:
    """Evaluate probe trained on one template against another template's scores.

    Args:
        manifest_dir: directory with manifest.json and probes/
        probe_id: probe identifier (e.g., "001")
        eval_template: template to evaluate on
        activations_dir: directory with activations.npz
        experiment_dir: directory with measurements (e.g., results/experiments/probe_2)

    Returns:
        dict with r2, mse, n_samples
    """
    manifest = load_manifest(manifest_dir)
    probe_meta = next((p for p in manifest["probes"] if p["id"] == probe_id), None)

    if probe_meta is None:
        raise ValueError(f"Probe {probe_id} not found in manifest")

    layer = probe_meta["layer"]
    template_trained = probe_meta["template"]

    # Load probe
    probe_weights = load_probe(manifest_dir, probe_id)
    coef = probe_weights[:-1]
    intercept = probe_weights[-1]

    # Load activations
    task_ids, activations = load_activations(activations_dir)
    X = activations[layer]

    # Load eval measurements
    scores_dict = load_pooled_scores(
        experiment_dir / "post_task_stated",
        eval_template,
        response_formats=probe_meta["response_formats"],
        seeds=probe_meta["seeds"],
    )

    if not scores_dict:
        return {"r2": None, "mse": None, "n_samples": 0}

    # Match activations to scores
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    valid_indices = []
    valid_scores = []
    for task_id, score in scores_dict.items():
        if task_id in id_to_idx:
            valid_indices.append(id_to_idx[task_id])
            valid_scores.append(score)

    if len(valid_indices) < 10:  # minimum samples for evaluation
        return {"r2": None, "mse": None, "n_samples": len(valid_indices)}

    indices = np.array(valid_indices)
    y = np.array(valid_scores)
    X_eval = X[indices]

    # Predict
    y_pred = X_eval @ coef + intercept

    # Compute metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    return {
        "r2": float(r2),
        "mse": float(mse),
        "n_samples": len(y),
        "trained_on": template_trained,
        "evaluated_on": eval_template,
    }


def compute_probe_similarity(
    manifest_dir: Path,
    probe_ids: list[str] | None = None,
) -> np.ndarray:
    """Compute cosine similarity between probe weight vectors.

    Args:
        manifest_dir: directory with manifest.json and probes/
        probe_ids: specific probes to compare, or None for all

    Returns:
        similarity matrix (len x len)
    """
    manifest = load_manifest(manifest_dir)

    if probe_ids is None:
        probe_ids = [p["id"] for p in manifest["probes"]]

    probes = []
    for probe_id in probe_ids:
        weights = load_probe(manifest_dir, probe_id)
        # Normalize to unit length
        weights_norm = weights / np.linalg.norm(weights)
        probes.append(weights_norm)

    probes_array = np.array(probes)
    # Cosine similarity = dot product of normalized vectors
    similarity = probes_array @ probes_array.T

    return similarity
