"""Activation loading and filtering utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.measurement_storage.loading import get_activation_task_ids


def load_activations(data_dir: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Load activations.npz, returning (task_ids, {layer: activations})."""
    npz_path = data_dir / "activations.npz"
    data = np.load(npz_path, allow_pickle=True)

    task_ids = data["task_ids"]
    layer_keys = [k for k in data.keys() if k.startswith("layer_")]
    layers = sorted(int(k.split("_")[1]) for k in layer_keys)
    activations = {layer: data[f"layer_{layer}"] for layer in layers}

    return task_ids, activations


def filter_activations_by_origin(
    task_ids: np.ndarray,
    origin: str,
    activations_dir: Path,
) -> np.ndarray:
    """Return boolean mask for tasks matching origin dataset."""
    matching_ids = get_activation_task_ids(activations_dir, origin_filter=origin)
    return np.array([tid in matching_ids for tid in task_ids])
