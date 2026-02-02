"""Activation loading and filtering utilities."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_activations(
    activations_path: Path,
    task_id_filter: set[str] | None = None,
    layers: list[int] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Load activations npz file, returning (task_ids, {layer: activations})."""
    data = np.load(activations_path, allow_pickle=True)

    task_ids = data["task_ids"]

    # Compute mask once
    if task_id_filter is not None:
        mask = np.array([tid in task_id_filter for tid in task_ids])
        task_ids = task_ids[mask]
    else:
        mask = None

    # Determine which layers to load
    available_layers = sorted(int(k.split("_")[1]) for k in data.keys() if k.startswith("layer_"))
    layers_to_load = layers if layers is not None else available_layers

    activations = {}
    for layer in layers_to_load:
        arr = data[f"layer_{layer}"]
        activations[layer] = arr[mask] if mask is not None else arr

    return task_ids, activations


def load_task_origins(activations_dir: Path) -> dict[str, set[str]]:
    """Load all task origins mapping. Returns {origin: set of task_ids}."""
    completions_path = activations_dir / "completions_with_activations.json"
    if not completions_path.exists():
        return {}

    with open(completions_path) as f:
        completions = json.load(f)

    origins: dict[str, set[str]] = defaultdict(set)
    for c in completions:
        if c.get("origin"):
            origins[c["origin"].upper()].add(c["task_id"])

    return dict(origins)


