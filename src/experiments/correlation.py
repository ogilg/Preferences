from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr


def safe_correlation(
    a: np.ndarray,
    b: np.ndarray,
    method: Literal["pearson", "spearman"] = "pearson",
) -> float:
    """Compute correlation, returning 0.0 if insufficient data or zero variance."""
    if len(a) < 2:
        return 0.0

    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0

    corr_fn = pearsonr if method == "pearson" else spearmanr
    r, _ = corr_fn(a, b)
    return float(r) if not np.isnan(r) else 0.0


def utility_vector_correlation(
    mu_a: np.ndarray,
    task_ids_a: list[str],
    mu_b: np.ndarray,
    task_ids_b: list[str],
    method: Literal["pearson", "spearman"] = "pearson",
) -> float:
    """
    Pearson/Spearman correlation of utility vectors, handling different task orderings.

    Returns NaN if task sets don't match or data is insufficient.
    """
    if set(task_ids_a) != set(task_ids_b):
        return float("nan")

    # Reorder b to match a's ordering if needed
    if task_ids_a != task_ids_b:
        id_to_idx_b = {tid: i for i, tid in enumerate(task_ids_b)}
        reorder = [id_to_idx_b[tid] for tid in task_ids_a]
        mu_b = mu_b[reorder]

    if len(mu_a) < 2 or np.std(mu_a) < 1e-10 or np.std(mu_b) < 1e-10:
        return float("nan")

    corr_fn = pearsonr if method == "pearson" else spearmanr
    r, _ = corr_fn(mu_a, mu_b)
    return float(r) if not np.isnan(r) else float("nan")


def compute_pairwise_correlations(
    results: dict[str, tuple[np.ndarray, list[str]]],
    tags: dict[str, dict[str, str]] | None = None,
    method: Literal["pearson", "spearman"] = "pearson",
) -> list[dict]:
    """
    Compute pairwise correlations between all result sets.

    Args:
        results: template_id -> (values, task_ids)
        tags: optional template_id -> tag dict for sensitivity analysis
        method: "pearson" or "spearman"
    """
    from itertools import combinations

    correlations = []
    for (id_a, (vals_a, ids_a)), (id_b, (vals_b, ids_b)) in combinations(results.items(), 2):
        corr = utility_vector_correlation(vals_a, ids_a, vals_b, ids_b, method)
        entry = {
            "template_a": id_a,
            "template_b": id_b,
            "correlation": float(corr),
        }
        if tags is not None:
            entry["tags_a"] = tags[id_a]
            entry["tags_b"] = tags[id_b]
        correlations.append(entry)
    return correlations


def save_correlations_yaml(
    correlations: list[dict],
    summary_keys: list[str],
    path: Path | str,
) -> None:
    """Save correlations to YAML with mean summary for specified keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = {"n_pairs": len(correlations)}
    for key in summary_keys:
        values = [c[key] for c in correlations]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else 0.0

    output = {
        "summary": summary,
        "pairwise": correlations,
    }

    with open(path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)


def save_experiment_config(
    templates: list,
    model_name: str,
    temperature: float,
    n_tasks: int,
    path: Path | str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "model": model_name,
        "temperature": temperature,
        "n_tasks": n_tasks,
        "templates": [
            {
                "name": t.name,
                "tags": list(t.tags),
                "prompt": t.template,
            }
            for t in templates
        ],
    }

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
