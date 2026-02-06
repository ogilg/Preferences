"""Helper functions for probe analysis."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from src.probes.core.storage import load_manifest
from src.probes.core.evaluate import compute_probe_similarity


def get_template(probe: dict) -> str:
    """Get template name from probe (handles both 'template' and 'templates' keys)."""
    if "template" in probe:
        return probe["template"]
    return probe["templates"][0]


def filter_probes(
    manifest: dict,
    template: str | None = None,
    layer: int | None = None,
    dataset: str | None = None,
    probe_ids: list[str] | None = None,
) -> list[dict]:
    """Filter probes by template, layer, dataset, or specific IDs."""
    probes = manifest["probes"]

    if probe_ids is not None:
        probes = [p for p in probes if p["id"] in probe_ids]

    if template is not None:
        probes = [p for p in probes if template in get_template(p)]

    if layer is not None:
        probes = [p for p in probes if p["layer"] == layer]

    if dataset is not None:
        probes = [p for p in probes if dataset in (p.get("datasets") or [])]

    return probes


def make_probe_label(probe: dict) -> str:
    """Create short label for a probe."""
    template_short = get_template(probe).replace("post_task_", "").replace("_", "")
    layer_num = probe["layer"]
    datasets = "-".join(probe["datasets"]) if probe.get("datasets") else "all"
    return f"{template_short} L{layer_num} {datasets}"


def get_probe_similarity(manifest_dir: Path, probes: list[dict]) -> np.ndarray:
    """Get cosine similarity matrix for probes."""
    probe_ids = [p["id"] for p in probes]
    return compute_probe_similarity(manifest_dir, probe_ids)
