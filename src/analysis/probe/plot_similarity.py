"""Plot probe similarity heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.probe.helpers import (
    get_probe_similarity,
    load_and_filter,
    output_path,
    probe_label,
)


def run(
    manifest_dir: Path,
    method: str | None = None,
    layer: int | None = None,
    probe_ids: list[str] | None = None,
    output: Path | None = None,
    no_plot: bool = False,
) -> None:
    _, probes = load_and_filter(manifest_dir, method, layer, probe_ids)

    if len(probes) < 2:
        print("Need at least 2 probes to compare")
        return

    similarity = get_probe_similarity(manifest_dir, probes)
    labels = [probe_label(p) for p in probes]

    # Text output: similarity matrix
    n = len(labels)
    max_label = max(len(l) for l in labels)
    header = " " * (max_label + 2) + "  ".join(f"{l:>8}" for l in labels)
    print("Similarity Matrix:")
    print(header)
    for i, row_label in enumerate(labels):
        row_vals = "  ".join(f"{similarity[i, j]:>8.3f}" for j in range(n))
        print(f"  {row_label:<{max_label}}  {row_vals}")

    # Mean off-diagonal similarity
    mask = ~np.eye(n, dtype=bool)
    mean_sim = similarity[mask].mean()
    print(f"\nMean off-diagonal similarity: {mean_sim:.3f}")

    if no_plot:
        return

    out = output or output_path("similarity", "all")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity", rotation=270, labelpad=20)

    for i in range(n):
        for j in range(n):
            color = "white" if abs(similarity[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title(f"Probe Similarity Matrix ({n} probes)")
    plt.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out}")
