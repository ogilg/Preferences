"""Plot probe similarity heatmap."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.probes.storage import load_manifest
from src.analysis.probe.probe_helpers import filter_probes, make_probe_label, get_probe_similarity


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot similarity heatmap")
    parser.add_argument("manifest_dir", type=Path, help="Directory with manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--template", type=str, help="Filter by template (substring match)")
    parser.add_argument("--layer", type=int, help="Filter by layer")
    parser.add_argument("--dataset", type=str, help="Filter by dataset")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest_dir)
    probes = filter_probes(manifest, args.template, args.layer, args.dataset)

    if len(probes) < 2:
        print("Need at least 2 probes to compare")
        return

    similarity = get_probe_similarity(args.manifest_dir, probes)
    labels = [make_probe_label(p) for p in probes]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity", rotation=270, labelpad=20)

    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center",
                          color="white" if abs(similarity[i, j]) > 0.5 else "black", fontsize=8)

    ax.set_title(f"Probe Similarity Matrix ({len(probes)} probes)")
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
