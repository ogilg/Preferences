"""Demean preference scores against categorical confounds (topic, dataset).

Fits OLS on one-hot encoded group indicators and subtracts predictions,
removing group-level mean differences from scores. This controls for the
fact that e.g. math tasks may have systematically different preference
scores than creative writing tasks.

This is distinct from content projection (see content_orthogonal.py), which
removes continuous content-predictable variance from activations via Ridge.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

def _extract_dataset_prefix(task_id: str) -> str:
    """Extract dataset origin from task_id prefix."""
    for prefix in ("competition_math_", "wildchat_", "alpaca_", "stresstest_", "bailbench_"):
        if task_id.startswith(prefix):
            return prefix.rstrip("_")
    return "unknown"


def _detect_classifier_model(topics_cache: dict) -> str:
    """Auto-detect the classifier model name from a topics cache."""
    sample_entry = next(iter(topics_cache.values()))
    models = list(sample_entry.keys())
    if len(models) != 1:
        raise ValueError(f"Expected exactly one classifier model, got: {models}")
    return models[0]


def _load_metadata_arrays(
    scores: dict[str, float],
    topics_json: Path,
) -> tuple[list[str], np.ndarray, list[str], list[str]]:
    """Load and align metadata arrays for all tasks with complete metadata.

    Returns (task_ids, y, dataset_labels, topic_labels).
    """
    with open(topics_json) as f:
        topics_cache = json.load(f)

    classifier_model = _detect_classifier_model(topics_cache)

    task_ids_ordered = []
    y_values = []
    dataset_labels = []
    topic_labels = []

    for tid, mu in scores.items():
        if tid not in topics_cache or classifier_model not in topics_cache[tid]:
            continue

        task_ids_ordered.append(tid)
        y_values.append(mu)
        dataset_labels.append(_extract_dataset_prefix(tid))
        topic_labels.append(topics_cache[tid][classifier_model]["primary"])

    return (
        task_ids_ordered,
        np.array(y_values),
        dataset_labels,
        topic_labels,
    )


def _onehot_columns(labels: list[str], prefix: str) -> tuple[list[np.ndarray], list[str]]:
    """One-hot encode labels, dropping first category for identifiability."""
    unique = sorted(set(labels))
    columns = []
    names = []
    for val in unique[1:]:
        columns.append(np.array([1.0 if l == val else 0.0 for l in labels]))
        names.append(f"{prefix}_{val}")
    return columns, names


def _fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[LinearRegression, float]:
    reg = LinearRegression().fit(X, y)
    return reg, reg.score(X, y)


def fit_metadata_models(
    scores: dict[str, float],
    topics_json: Path,
) -> dict:
    """Fit three OLS models (topic-only, dataset-only, both) and return decomposition."""
    task_ids, y, ds_labels, tp_labels = _load_metadata_arrays(scores, topics_json)

    n_total = len(scores)
    n_with_metadata = len(task_ids)
    n_dropped = n_total - n_with_metadata
    if n_dropped > 0:
        print(f"  Metadata: dropped {n_dropped}/{n_total} tasks missing metadata")

    ds_cols, ds_names = _onehot_columns(ds_labels, "dataset")
    tp_cols, tp_names = _onehot_columns(tp_labels, "topic")

    # Topic-only
    X_topic = np.column_stack(tp_cols)
    topic_features = tp_names
    reg_topic, r2_topic = _fit_ols(X_topic, y)

    # Dataset-only
    X_dataset = np.column_stack(ds_cols)
    dataset_features = ds_names
    _, r2_dataset = _fit_ols(X_dataset, y)

    # Both
    X_both = np.column_stack(ds_cols + tp_cols)
    both_features = ds_names + tp_names
    reg_both, r2_both = _fit_ols(X_both, y)

    unique_datasets = sorted(set(ds_labels))
    unique_topics = sorted(set(tp_labels))

    # Per-dataset residual means (from topic-only model)
    topic_residuals = y - reg_topic.predict(X_topic)
    ds_residual_means = {}
    for ds in unique_datasets:
        mask = [d == ds for d in ds_labels]
        ds_residual_means[ds] = round(float(np.mean(topic_residuals[mask])), 4)

    return {
        "n_tasks": n_with_metadata,
        "n_dropped": n_dropped,
        "unique_datasets": unique_datasets,
        "unique_topics": unique_topics,
        # Topic-only model (used for residualization)
        "topic_r2": round(r2_topic, 4),
        "topic_features": topic_features,
        "topic_coefs": reg_topic.coef_.tolist(),
        "topic_intercept": float(reg_topic.intercept_),
        "topic_ref": unique_topics[0],
        "topic_residual_by_dataset": ds_residual_means,
        # Dataset-only model
        "dataset_r2": round(r2_dataset, 4),
        "dataset_features": dataset_features,
        # Both model
        "both_r2": round(r2_both, 4),
        "both_features": both_features,
        "both_coefs": reg_both.coef_.tolist(),
        "both_intercept": float(reg_both.intercept_),
        "both_ref_dataset": unique_datasets[0],
        "both_ref_topic": unique_topics[0],
    }


def build_task_groups(
    task_ids: set[str],
    grouping: str,
    topics_json: Path | None = None,
) -> dict[str, str]:
    """Returns {task_id: group_label} for tasks that have group metadata.

    grouping: "topic" (requires topics_json) or "dataset" (uses task_id prefix).
    """
    if grouping == "dataset":
        return {tid: _extract_dataset_prefix(tid) for tid in task_ids}

    if grouping == "topic":
        if topics_json is None:
            raise ValueError("topics_json required for topic grouping")
        with open(topics_json) as f:
            topics_cache = json.load(f)
        classifier_model = _detect_classifier_model(topics_cache)
        result = {}
        for tid in task_ids:
            if tid in topics_cache and classifier_model in topics_cache[tid]:
                result[tid] = topics_cache[tid][classifier_model]["primary"]
        return result

    raise ValueError(f"Unknown grouping: {grouping!r}. Use 'topic' or 'dataset'.")


VALID_CONFOUNDS = {"topic", "dataset"}


def demean_scores(
    scores: dict[str, float],
    topics_json: Path,
    confounds: list[str],
) -> tuple[dict[str, float], dict]:
    """Regress out specified confounds from scores, return residuals.

    confounds: subset of {"topic", "dataset"}.
    """
    invalid = set(confounds) - VALID_CONFOUNDS
    if invalid:
        raise ValueError(f"Unknown confounds: {invalid}. Valid: {VALID_CONFOUNDS}")
    if not confounds:
        raise ValueError("confounds list cannot be empty")

    task_ids, y, ds_labels, tp_labels = _load_metadata_arrays(scores, topics_json)

    n_with_metadata = len(task_ids)
    n_dropped = len(scores) - n_with_metadata
    if n_dropped > 0:
        print(f"  Demeaning: dropped {n_dropped}/{len(scores)} tasks missing metadata")

    columns = []
    feature_names = []

    if "dataset" in confounds:
        ds_cols, ds_names = _onehot_columns(ds_labels, "dataset")
        columns.extend(ds_cols)
        feature_names.extend(ds_names)

    if "topic" in confounds:
        tp_cols, tp_names = _onehot_columns(tp_labels, "topic")
        columns.extend(tp_cols)
        feature_names.extend(tp_names)

    X = np.column_stack(columns)

    reg = LinearRegression().fit(X, y)
    residuals = y - reg.predict(X)
    metadata_r2 = reg.score(X, y)

    residual_scores = dict(zip(task_ids, residuals.tolist()))

    stats = {
        "confounds": confounds,
        "metadata_r2": round(metadata_r2, 4),
        "metadata_features": feature_names,
        "n_metadata_features": len(feature_names),
        "n_tasks_demeaned": n_with_metadata,
        "n_tasks_dropped": n_dropped,
        "topics_used": sorted(set(tp_labels)),
    }

    return residual_scores, stats
