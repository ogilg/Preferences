"""Print probe summary table."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.probes.storage import load_manifest
from src.analysis.probe.probe_helpers import filter_probes


def main() -> None:
    parser = argparse.ArgumentParser(description="Print summary table")
    parser.add_argument("manifest_dir", type=Path, help="Directory with manifest.json")
    parser.add_argument("--template", type=str, help="Filter by template (substring match)")
    parser.add_argument("--layer", type=int, help="Filter by layer")
    parser.add_argument("--dataset", type=str, help="Filter by dataset")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest_dir)
    probes = filter_probes(manifest, args.template, args.layer, args.dataset)

    if not probes:
        print("No probes match filters")
        return

    probes = sorted(probes, key=lambda x: (x["template"], x["layer"]))

    print("\nProbe Performance Summary:")
    print("-" * 110)
    print(f"{'ID':<4} {'Template':<30} {'Layer':<6} {'Datasets':<20} {'R²':<8} {'±':<6} {'N':<6}")
    print("-" * 110)

    for p in probes:
        template_name = p["template"][:29]
        layer_num = p["layer"]
        datasets = ", ".join(p["datasets"]) if p["datasets"] else "all"
        r2_mean = p["cv_r2_mean"]
        r2_std = p["cv_r2_std"]
        n_samples = p["n_samples"]

        print(f"{p['id']:<4} {template_name:<30} {layer_num:<6} {datasets:<20} {r2_mean:>7.3f} {r2_std:>6.3f} {n_samples:>6}")

    print("-" * 110)


if __name__ == "__main__":
    main()
