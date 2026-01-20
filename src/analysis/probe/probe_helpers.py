"""Helper functions for probe analysis."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from src.probes.storage import load_manifest
from src.probes.evaluate import compute_probe_similarity


def filter_probes(manifest: dict, template: str | None = None, layer: int | None = None, dataset: str | None = None) -> list[dict]:
    """Filter probes by template, layer, dataset.

    Args:
        manifest: manifest dict from load_manifest()
        template: filter by template name (substring match)
        layer: filter by layer number
        dataset: filter by dataset name (substring match)

    Returns:
        filtered list of probe dicts
    """
    probes = manifest["probes"]

    if template is not None:
        probes = [p for p in probes if template in p["template"]]

    if layer is not None:
        probes = [p for p in probes if p["layer"] == layer]

    if dataset is not None:
        probes = [p for p in probes if dataset in (p.get("datasets") or [])]

    return probes


def make_probe_label(probe: dict) -> str:
    """Create short label for a probe."""
    template_short = probe["template"].replace("post_task_", "").replace("_", "")
    layer_num = probe["layer"]
    datasets = "-".join(probe["datasets"]) if probe["datasets"] else "all"
    return f"{template_short} L{layer_num} {datasets}"


def get_probe_similarity(manifest_dir: Path, probes: list[dict]) -> np.ndarray:
    """Get cosine similarity matrix for probes."""
    probe_ids = [p["id"] for p in probes]
    return compute_probe_similarity(manifest_dir, probe_ids)
