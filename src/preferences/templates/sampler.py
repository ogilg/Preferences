from __future__ import annotations

from collections import Counter
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.preferences.templates.template import PromptTemplate


TEMPLATE_DIMENSIONS = [
    "phrasing",
    "instruction_position",
    "task_label_names",
    "situating_context",
    "language",
]


def sample_configurations_lhs(
    templates: list[PromptTemplate],
    response_formats: list[str],
    orders: list[str],
    n_samples: int,
    seed: int | None = None,
) -> list[tuple[PromptTemplate, str, str]]:
    """
    Sample n configurations using Latin Hypercube Sampling.

    Ensures balanced coverage: each dimension value appears
    floor(n/k) or ceil(n/k) times where k is the number of values.
    """
    rng = np.random.default_rng(seed)

    # Build full hypercube
    all_configs = list(product(templates, response_formats, orders))

    if n_samples >= len(all_configs):
        return all_configs

    # Extract dimension values for each config
    def get_dims(config: tuple[PromptTemplate, str, str]) -> dict[str, str]:
        template, resp_fmt, order = config
        dims = template.tags_dict.copy()
        dims["response_format"] = resp_fmt
        dims["order"] = order
        return dims

    config_dims = [get_dims(c) for c in all_configs]

    # Find all dimensions that actually vary
    all_dim_keys = set()
    for dims in config_dims:
        all_dim_keys.update(dims.keys())

    # For each dimension, get unique values
    dim_values: dict[str, list[str]] = {}
    for key in all_dim_keys:
        values = list({dims.get(key, "") for dims in config_dims})
        if len(values) > 1:  # Only include dimensions that vary
            dim_values[key] = values

    # Compute target counts per dimension value
    target_counts: dict[str, dict[str, int]] = {}
    for dim, values in dim_values.items():
        k = len(values)
        base = n_samples // k
        remainder = n_samples % k
        # Randomly assign which values get +1
        extras = rng.choice(values, size=remainder, replace=False).tolist()
        target_counts[dim] = {v: base + (1 if v in extras else 0) for v in values}

    # Greedy selection to match LHS targets
    selected_indices: list[int] = []
    current_counts: dict[str, Counter[str]] = {dim: Counter() for dim in dim_values}
    available = set(range(len(all_configs)))

    for _ in range(n_samples):
        # Score each available config by how much it helps balance
        best_idx = None
        best_score = float("-inf")

        for idx in available:
            dims = config_dims[idx]
            score = 0.0
            for dim, values in dim_values.items():
                val = dims.get(dim, "")
                if val in target_counts[dim]:
                    current = current_counts[dim][val]
                    target = target_counts[dim][val]
                    # Prefer values that are under-represented
                    if current < target:
                        score += (target - current) / target
                    else:
                        # Penalize over-represented
                        score -= (current - target + 1) * 10

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        available.remove(best_idx)

        # Update counts
        dims = config_dims[best_idx]
        for dim in dim_values:
            val = dims.get(dim, "")
            current_counts[dim][val] += 1

    return [all_configs[i] for i in selected_indices]


def print_sampling_balance(
    configs: list[tuple[PromptTemplate, str, str]],
) -> None:
    """Print dimension value counts for debugging."""
    counts: dict[str, Counter[str]] = {}

    for template, resp_fmt, order in configs:
        dims = template.tags_dict.copy()
        dims["response_format"] = resp_fmt
        dims["order"] = order

        for key, val in dims.items():
            if key not in counts:
                counts[key] = Counter()
            counts[key][val] += 1

    print(f"Sampling balance ({len(configs)} configurations):")
    for dim in sorted(counts.keys()):
        print(f"  {dim}: {dict(counts[dim])}")
