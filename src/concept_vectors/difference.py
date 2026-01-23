"""Compute difference-in-means concept vectors from paired activation extractions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def compute_difference_in_means(
    positive_dir: Path,
    negative_dir: Path,
    layers: list[int] | None = None,
    normalize: bool = True,
) -> dict[int, np.ndarray]:
    """Compute concept direction as mean(positive) - mean(negative) per layer.

    Args:
        positive_dir: Directory with positive condition activations.npz
        negative_dir: Directory with negative condition activations.npz
        layers: Specific layers to process (None = all available)
        normalize: Whether to normalize vectors to unit length

    Returns:
        Dictionary mapping layer index to direction vector

    Raises:
        ValueError: If task IDs don't match between conditions
    """
    pos_data = np.load(positive_dir / "activations.npz", allow_pickle=True)
    neg_data = np.load(negative_dir / "activations.npz", allow_pickle=True)

    pos_task_ids = set(pos_data["task_ids"])
    neg_task_ids = set(neg_data["task_ids"])

    if pos_task_ids != neg_task_ids:
        only_pos = pos_task_ids - neg_task_ids
        only_neg = neg_task_ids - pos_task_ids
        raise ValueError(
            f"Task ID mismatch: {len(only_pos)} only in positive, {len(only_neg)} only in negative"
        )

    # Determine layers to process
    available_layers = sorted(
        int(k.split("_")[1]) for k in pos_data.keys() if k.startswith("layer_")
    )
    layers_to_process = layers if layers is not None else available_layers

    # Create mapping from task_id to index for alignment
    pos_ids = list(pos_data["task_ids"])
    neg_ids = list(neg_data["task_ids"])
    pos_idx_map = {tid: i for i, tid in enumerate(pos_ids)}
    neg_idx_map = {tid: i for i, tid in enumerate(neg_ids)}

    # Get common task order
    common_ids = sorted(pos_task_ids)
    pos_indices = [pos_idx_map[tid] for tid in common_ids]
    neg_indices = [neg_idx_map[tid] for tid in common_ids]

    directions = {}
    for layer in layers_to_process:
        pos_acts = pos_data[f"layer_{layer}"][pos_indices]
        neg_acts = neg_data[f"layer_{layer}"][neg_indices]

        direction = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)

        if normalize:
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                raise ValueError(f"Layer {layer} has near-zero norm direction")
            direction = direction / norm

        directions[layer] = direction.astype(np.float32)

    return directions


def save_concept_vectors(
    vectors: dict[int, np.ndarray],
    output_dir: Path,
    metadata: dict,
) -> None:
    """Save concept vectors and manifest.

    Creates:
        - vectors/layer_N.npy for each layer
        - manifest.json with metadata and file references
    """
    vectors_dir = output_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    vector_files = {}
    for layer, vec in vectors.items():
        filename = f"layer_{layer}.npy"
        np.save(vectors_dir / filename, vec)
        vector_files[layer] = f"vectors/{filename}"

    manifest = {
        **metadata,
        "created_at": datetime.now().isoformat(),
        "n_layers": len(vectors),
        "layers": sorted(vectors.keys()),
        "vector_files": vector_files,
        "hidden_dim": vectors[sorted(vectors.keys())[0]].shape[0] if vectors else None,
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load_concept_vector(manifest_dir: Path, layer: int) -> np.ndarray:
    """Load a single concept vector for a specific layer."""
    manifest_path = manifest_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    if layer not in manifest["layers"]:
        raise ValueError(f"Layer {layer} not available. Available: {manifest['layers']}")

    vector_path = manifest_dir / manifest["vector_files"][str(layer)]
    return np.load(vector_path)


def load_concept_vector_for_steering(
    manifest_dir: Path,
    layer: int | None = None,
) -> tuple[int, np.ndarray]:
    """Load concept vector for use with steering runner.

    Compatible with the interface of load_probe_direction() in src/probes/storage.py.

    Args:
        manifest_dir: Directory containing manifest.json
        layer: Specific layer to load (None = use middle layer)

    Returns:
        Tuple of (layer_index, unit_normalized_direction_vector)
    """
    manifest_path = manifest_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    available_layers = manifest["layers"]
    if layer is None:
        # Default to middle layer
        layer = available_layers[len(available_layers) // 2]

    if layer not in available_layers:
        raise ValueError(f"Layer {layer} not available. Available: {available_layers}")

    vector_path = manifest_dir / manifest["vector_files"][str(layer)]
    direction = np.load(vector_path)
    return layer, direction
