"""Probe storage and manifest management."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_probe(
    weights: np.ndarray,
    output_dir: Path,
    probe_id: str,
) -> str:
    """Save probe weights, return relative path."""
    probes_dir = output_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probes_dir / f"probe_{probe_id}.npy"
    np.save(probe_path, weights)
    return str(probe_path.relative_to(output_dir))


def load_probe(output_dir: Path, probe_id: str) -> np.ndarray:
    """Load probe weights from disk."""
    return np.load(output_dir / "probes" / f"probe_{probe_id}.npy")


def save_manifest(manifest_data: dict, output_dir: Path) -> None:
    """Save manifest to JSON."""
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)


def load_manifest(output_dir: Path) -> dict:
    """Load manifest from JSON."""
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path) as f:
        return json.load(f)
