from __future__ import annotations

from collections import defaultdict

import numpy as np


def get_differing_tags(tags_a: dict[str, str], tags_b: dict[str, str]) -> list[str]:
    return [k for k in tags_a if tags_a[k] != tags_b.get(k)]


def compute_sensitivities(
    correlations: list[dict],
    correlation_key: str = "pearson_correlation",
) -> dict[str, dict]:
    """
    Group correlations by which single tag differs, compute mean/std.

    Returns dict mapping field name to stats:
    {
        "phrasing": {"mean": 0.95, "std": 0.02, "n_pairs": 10, "values": [...]},
        ...
    }
    """
    by_field: dict[str, list[float]] = defaultdict(list)

    for c in correlations:
        if "tags_a" not in c or "tags_b" not in c:
            continue
        diff = get_differing_tags(c["tags_a"], c["tags_b"])
        if len(diff) == 1:
            by_field[diff[0]].append(c[correlation_key])

    return {
        field: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n_pairs": len(values),
            "values": values,
        }
        for field, values in by_field.items()
    }
