"""Run all baseline experiments for probe benchmarking."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.probes.baselines import run_all_baselines, aggregate_noise_baselines, BaselineType
from src.probes.config import ProbeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiments for probe benchmarking")
    parser.add_argument(
        "--reference-config",
        type=Path,
        required=True,
        help="Path to probe config YAML (defines templates, layers, etc.)",
    )
    parser.add_argument(
        "--task-description-dir",
        type=Path,
        help="Directory containing task description activations (optional)",
    )
    parser.add_argument(
        "--n-noise-seeds",
        type=int,
        default=5,
        help="Number of random seeds for noise baselines",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for baseline results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ProbeConfig.from_yaml(args.reference_config)

    print(f"Baseline Experiments")
    print(f"Reference config: {args.reference_config}")
    print(f"Template combinations: {config.training_data.template_combinations}")
    print(f"Layers: {config.layers}")
    print(f"Task description dir: {args.task_description_dir}")
    print(f"Noise seeds: {args.n_noise_seeds}")
    print(f"Output dir: {args.output_dir}")
    print()

    # Run all baselines
    results = run_all_baselines(
        config,
        args.task_description_dir,
        args.n_noise_seeds,
    )

    print(f"\nCollected {len(results)} baseline results")

    # Aggregate noise baselines
    aggregated_noise = aggregate_noise_baselines(results)

    # Extract task description results (no aggregation needed)
    task_description_results = [
        r.to_dict() for r in results
        if r.baseline_type == BaselineType.TASK_DESCRIPTION
    ]

    # Build manifest
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment_name": args.output_dir.name,
        "reference_config": str(args.reference_config),
        "reference_experiment": config.experiment_name,
        "created_at": datetime.now().isoformat(),
        "n_noise_seeds": args.n_noise_seeds,
        "task_description_dir": str(args.task_description_dir) if args.task_description_dir else None,
        "baselines": {
            "shuffled_labels": [
                r for r in aggregated_noise
                if r["baseline_type"] == "shuffled_labels"
            ],
            "random_activations": [
                r for r in aggregated_noise
                if r["baseline_type"] == "random_activations"
            ],
            "task_description": task_description_results,
        },
    }

    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest to {manifest_path}")

    # Print summary
    print("\n=== Baseline Summary ===")

    for baseline_type in ["shuffled_labels", "random_activations", "task_description"]:
        baselines = manifest["baselines"][baseline_type]
        if not baselines:
            print(f"\n{baseline_type}: no results")
            continue

        print(f"\n{baseline_type}:")
        r2_values = [b["cv_r2_mean"] for b in baselines]
        print(f"  Count: {len(baselines)}")
        print(f"  R² mean: {sum(r2_values) / len(r2_values):.4f}")
        print(f"  R² range: [{min(r2_values):.4f}, {max(r2_values):.4f}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
