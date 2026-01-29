"""Compute difference-in-means concept vectors from paired activation extractions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def compute_difference_in_means(
    positive_dir: Path,
    negative_dir: Path,
    selector_name: str,
    layers: list[int] | None = None,
    normalize: bool = False,
) -> dict[int, np.ndarray]:
    """Compute concept direction as mean(positive) - mean(negative) per layer.

    Uses intersection of task IDs if some tasks failed in one condition.

    Args:
        positive_dir: Directory with positive condition activations
        negative_dir: Directory with negative condition activations
        selector_name: Which token selector to use ('last', 'first', 'mean')
        layers: Specific layers to process (None = all available)
        normalize: Whether to normalize vectors to unit length

    Returns:
        Dictionary mapping layer index to direction vector
    """
    pos_data = np.load(positive_dir / f"activations_{selector_name}.npz", allow_pickle=True)
    neg_data = np.load(negative_dir / f"activations_{selector_name}.npz", allow_pickle=True)

    pos_task_ids = set(pos_data["task_ids"])
    neg_task_ids = set(neg_data["task_ids"])

    common_task_ids = pos_task_ids & neg_task_ids
    only_pos = pos_task_ids - neg_task_ids
    only_neg = neg_task_ids - pos_task_ids

    if only_pos or only_neg:
        print(f"Warning: {len(only_pos)} tasks only in positive, {len(only_neg)} only in negative")
        print(f"Using intersection of {len(common_task_ids)} tasks")

    if len(common_task_ids) == 0:
        raise ValueError("No common task IDs between conditions")

    available_layers = sorted(
        int(k.split("_")[1]) for k in pos_data.keys() if k.startswith("layer_")
    )
    layers_to_process = layers if layers is not None else available_layers

    pos_ids = list(pos_data["task_ids"])
    neg_ids = list(neg_data["task_ids"])
    pos_idx_map = {tid: i for i, tid in enumerate(pos_ids)}
    neg_idx_map = {tid: i for i, tid in enumerate(neg_ids)}

    common_ids = sorted(common_task_ids)
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


def compute_all_concept_vectors(
    positive_dir: Path,
    negative_dir: Path,
    selector_names: list[str],
    layers: list[int] | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Compute concept vectors for specified token selectors.

    Returns:
        {selector_name: {layer: direction_vector}}
    """
    results = {}
    for selector_name in selector_names:
        print(f"Computing {selector_name} token vectors...")
        results[selector_name] = compute_difference_in_means(
            positive_dir, negative_dir, selector_name, layers
        )
    return results


def save_concept_vectors(
    vectors_by_selector: dict[str, dict[int, np.ndarray]],
    output_dir: Path,
    metadata: dict,
) -> None:
    """Save concept vectors for all selectors.

    Creates:
        - vectors/{selector}/layer_N.npy for each selector and layer
        - manifest.json with metadata and file references
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    vector_files: dict[str, dict[int, str]] = {}
    layers = None
    hidden_dim = None

    for selector_name, vectors in vectors_by_selector.items():
        vectors_dir = output_dir / "vectors" / selector_name
        vectors_dir.mkdir(parents=True, exist_ok=True)

        vector_files[selector_name] = {}
        for layer, vec in vectors.items():
            filename = f"layer_{layer}.npy"
            np.save(vectors_dir / filename, vec)
            vector_files[selector_name][layer] = f"vectors/{selector_name}/{filename}"

        if layers is None:
            layers = sorted(vectors.keys())
            hidden_dim = vectors[layers[0]].shape[0] if vectors else None

    manifest = {
        **metadata,
        "created_at": datetime.now().isoformat(),
        "selectors": list(vectors_by_selector.keys()),
        "n_layers": len(layers) if layers else 0,
        "layers": layers,
        "vector_files": vector_files,
        "hidden_dim": hidden_dim,
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load_concept_vector(
    manifest_dir: Path,
    layer: int,
    selector: str = "last",
) -> np.ndarray:
    """Load a single concept vector for a specific layer and selector."""
    manifest_path = manifest_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    if layer not in manifest["layers"]:
        raise ValueError(f"Layer {layer} not available. Available: {manifest['layers']}")
    if selector not in manifest["selectors"]:
        raise ValueError(f"Selector {selector} not available. Available: {manifest['selectors']}")

    vector_path = manifest_dir / manifest["vector_files"][selector][str(layer)]
    return np.load(vector_path)


def load_concept_vector_for_steering(
    manifest_dir: Path,
    layer: int | None = None,
    selector: str = "last",
) -> tuple[int, np.ndarray]:
    """Load concept vector for use with steering runner.

    Compatible with the interface of load_probe_direction() in src/probes/storage.py.

    Args:
        manifest_dir: Directory containing manifest.json
        layer: Specific layer to load (None = use middle layer)
        selector: Which token selector to use ('last', 'first', 'mean')

    Returns:
        Tuple of (layer_index, unit_normalized_direction_vector)
    """
    manifest_path = manifest_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    available_layers = manifest["layers"]
    if layer is None:
        layer = available_layers[len(available_layers) // 2]

    if layer not in available_layers:
        raise ValueError(f"Layer {layer} not available. Available: {available_layers}")
    if selector not in manifest["selectors"]:
        raise ValueError(f"Selector {selector} not available. Available: {manifest['selectors']}")

    vector_path = manifest_dir / manifest["vector_files"][selector][str(layer)]
    direction = np.load(vector_path)
    return layer, direction
