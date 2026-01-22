"""Plot R² comparison bar chart."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.probes.storage import load_manifest
from src.analysis.probe.probe_helpers import filter_probes, make_probe_label, get_template


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot R² comparison")
    parser.add_argument("manifest_dir", type=Path, help="Directory with manifest.json")
    parser.add_argument("--output", type=Path, help="Output PNG path (default: src/analysis/probe/plots/plot_MMDDYY_r2_*.png)")
    parser.add_argument("--template", type=str, help="Filter by template (substring match)")
    parser.add_argument("--layer", type=int, help="Filter by layer")
    parser.add_argument("--dataset", type=str, help="Filter by dataset")
    parser.add_argument("--probes", type=str, help="Comma-separated probe IDs (e.g., '0001,0002,0003')")
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        from datetime import datetime
        date_str = datetime.now().strftime("%m%d%y")
        filters = []
        if args.template:
            filters.append(args.template[:10])
        if args.layer is not None:
            filters.append(f"L{args.layer}")
        if args.dataset:
            filters.append(args.dataset[:10])
        filter_suffix = "_".join(filters) if filters else "all"
        args.output = Path(f"src/analysis/probe/plots/plot_{date_str}_r2_{filter_suffix}.png")

    manifest = load_manifest(args.manifest_dir)
    probe_ids = args.probes.split(",") if args.probes else None
    probes = filter_probes(manifest, args.template, args.layer, args.dataset, probe_ids)

    if not probes:
        print("No probes match filters")
        return

    probes = sorted(probes, key=lambda p: (get_template(p), p["layer"]))

    labels = [make_probe_label(p) for p in probes]
    r2_means = [p["cv_r2_mean"] for p in probes]
    r2_stds = [p["cv_r2_std"] for p in probes]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=5, alpha=0.7, color="steelblue", edgecolor="black")

    ax.set_xlabel("Probe", fontsize=11)
    ax.set_ylabel("R² (CV Mean ± Std)", fontsize=11)
    ax.set_title(f"Probe Performance Comparison ({len(probes)} probes)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for i, (bar, mean, std) in enumerate(zip(bars, r2_means, r2_stds)):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.01, f"{mean:.3f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
