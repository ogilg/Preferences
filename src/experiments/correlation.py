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
