"""Run held-one-out (leave-one-out) validation for probes."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.probes.config import ProbeTrainingConfig
from src.probes.storage import load_manifest, save_manifest


def get_available_datasets(activations_path: Path) -> set[str]:
    """Load available datasets from activations metadata."""
    # Ensure we have a directory
    if activations_path.name == "activations.npz":
        activations_path = activations_path.parent

    completions_path = activations_path / "completions_with_activations.json"
    if not completions_path.exists():
        raise FileNotFoundError(f"Cannot find {completions_path}")

    with open(completions_path) as f:
        completions = json.load(f)

    origins = set(c.get("origin") for c in completions if c.get("origin"))
    return {o.lower() for o in origins}  # Normalize to lowercase


def consolidate_hoo_metadata(
    folds: list[dict],
    unified_manifest_dir: Path,
    datasets: set[str],
    hold_out_size: int,
) -> None:
    """Consolidate probe metadata from all folds and add HOO tracking.

    Probes are already stored in unified_manifest_dir/probes/ from training.
    This function updates the manifest to track which fold each probe came from.
    """
    # Load the existing unified manifest that was created during training
    unified_manifest = load_manifest(unified_manifest_dir)

    # Track which probes belong to which fold (by mapping probe IDs)
    # All probes written during training are already in the manifest
    # We just need to add HOO metadata

    probe_to_fold = {}
    for fold in folds:
        fold_eval_datasets = fold["eval_datasets"]
        fold_train_datasets = sorted(fold.get("training_datasets", []))

        # Map probes from this fold
        for probe in unified_manifest["probes"]:
            # Check if probe belongs to this fold by comparing training datasets
            probe_train_datasets = sorted(probe.get("datasets", []))
            if probe_train_datasets == fold_train_datasets:
                if probe["id"] not in probe_to_fold:
                    probe_to_fold[probe["id"]] = {
                        "eval_datasets": fold_eval_datasets,
                        "train_datasets": fold_train_datasets,
                    }

    # Update probes with HOO metadata
    for probe in unified_manifest["probes"]:
        if probe["id"] in probe_to_fold:
            fold_info = probe_to_fold[probe["id"]]
            probe["hoo_fold_eval_datasets"] = fold_info["eval_datasets"]
            probe["hoo_fold_train_datasets"] = fold_info["train_datasets"]

    # Add consolidation metadata
    unified_manifest["hoo_consolidation"] = {
        "created_at": datetime.now().isoformat(),
        "num_folds": len(folds),
        "datasets": sorted(datasets),
        "hold_out_size": hold_out_size,
        "total_probes": len(unified_manifest["probes"]),
    }

    save_manifest(unified_manifest, unified_manifest_dir)
    print(f"Added HOO metadata to {len(unified_manifest['probes'])} probes in {unified_manifest_dir}")


def create_training_config(
    base_config: dict,
    exclude_datasets: set[str],
    datasets: set[str],
) -> ProbeTrainingConfig:
    """Create training config excluding specified datasets.

    All folds write probes to the same unified manifest directory.
    Individual probes are tracked with metadata indicating which fold they came from.
    """
    train_datasets = sorted(datasets - exclude_datasets)

    config_dict = dict(base_config)
    config_dict["dataset_combinations"] = [train_datasets]

    if exclude_datasets:
        exclude_str = "_".join(sorted(exclude_datasets))
        config_dict["experiment_name"] = f"{base_config['experiment_name']}_exclude_{exclude_str}"
        config_dict["experiment_dir"] = Path(str(base_config["experiment_dir"]).replace(
            "DATASET", exclude_str
        ))
        # Keep manifest_dir unified (remove _DATASET placeholder)
        config_dict["manifest_dir"] = Path(str(base_config["manifest_dir"]).replace(
            "_DATASET", ""
        ))
    else:
        # Remove DATASET placeholder if present
        config_dict["manifest_dir"] = Path(str(base_config["manifest_dir"]).replace(
            "_DATASET", ""
        ))

    config_dict["experiment_dir"] = Path(config_dict["experiment_dir"])
    config_dict["activations_path"] = Path(config_dict["activations_path"])
    config_dict["manifest_dir"] = Path(config_dict["manifest_dir"])

    return ProbeTrainingConfig(**config_dict)


def run_training(training_config: ProbeTrainingConfig) -> dict:
    """Run probe training via subprocess."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_dict = {
            "experiment_name": training_config.experiment_name,
            "experiment_dir": str(training_config.experiment_dir),
            "activations_path": str(training_config.activations_path),
            "template_combinations": training_config.template_combinations,
            "dataset_combinations": training_config.dataset_combinations,
            "seed_combinations": training_config.seed_combinations,
            "layers": training_config.layers,
            "cv_folds": training_config.cv_folds,
            "alpha_sweep_size": training_config.alpha_sweep_size,
            "manifest_dir": str(training_config.manifest_dir),
        }
        yaml.dump(config_dict, f)
        temp_config_path = f.name

    try:
        subprocess.run(
            [sys.executable, "-m", "src.probes.train_probe_experiment", "--config", temp_config_path],
            check=True,
            cwd=Path.cwd(),
        )
    finally:
        Path(temp_config_path).unlink()

    # Load and return manifest
    return load_manifest(training_config.manifest_dir)


def run_held_one_out(
    base_training_config: dict,
    datasets: set[str] | None = None,
    hold_out_size: int = 1,
    output_dir: Path | None = None,
) -> dict:
    """Run held-out training across all datasets.

    Trains one probe set per fold, excluding hold_out_size datasets at a time.
    All probes are written to the same unified manifest directory.

    Args:
        base_training_config: dict with base config (experiment_name, experiment_dir, etc.)
        datasets: set of datasets to validate on, or None to auto-detect
        hold_out_size: number of datasets to hold out per fold (default 1 = leave-one-out)
        output_dir: directory to save results

    Returns:
        dict with fold information for later evaluation
    """
    # Determine datasets
    if datasets is None:
        activations_path = Path(base_training_config["activations_path"])
        # If path points to activations.npz file, use parent directory
        if activations_path.name == "activations.npz":
            activations_path = activations_path.parent
        datasets = get_available_datasets(activations_path)

    if hold_out_size >= len(datasets):
        raise ValueError(f"hold_out_size ({hold_out_size}) must be less than number of datasets ({len(datasets)})")

    print("=" * 80)
    print(f"HELD-OUT TRAINING (hold_out_size={hold_out_size})")
    print("=" * 80)
    print(f"Datasets: {sorted(datasets)}\n")

    results = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "datasets": sorted(datasets),
            "hold_out_size": hold_out_size,
        },
        "folds": [],
    }

    for eval_combo in combinations(sorted(datasets), hold_out_size):
        eval_datasets = set(eval_combo)
        train_datasets = datasets - eval_datasets
        eval_str = ", ".join(sorted(eval_datasets)).upper()

        print("=" * 80)
        print(f"FOLD: Hold out {eval_str}")
        print("=" * 80)

        # Create training config
        training_config = create_training_config(base_training_config, eval_datasets, datasets)
        print(f"\nTraining on: {sorted(train_datasets)}")
        print(f"Held out: {sorted(eval_datasets)}\n")

        # Train
        print("Training...")
        try:
            training_manifest = run_training(training_config)
            fold_result = {
                "eval_datasets": sorted(eval_datasets),
                "training_datasets": sorted(train_datasets),
                "manifest_dir": str(training_config.manifest_dir),
                "n_probes": len(training_manifest["probes"]),
            }
            results["folds"].append(fold_result)
            print(f"Trained {len(training_manifest['probes'])} probes\n")
        except Exception as e:
            import traceback
            print(f"Error during training: {e}")
            print(traceback.format_exc())
            continue

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if not results["folds"]:
        print("No folds completed")
        return results

    print(f"\n{'Hold-out Datasets':<30} {'# Trained Probes':<20}")
    print("-" * 50)

    for fold in results["folds"]:
        eval_str = ", ".join(fold["eval_datasets"])
        print(f"{eval_str:<30} {fold['n_probes']:<20}")

    total_probes = sum(f["n_probes"] for f in results["folds"])
    print("-" * 50)
    print(f"{'Total':<30} {total_probes:<20}")
    print(f"\nAll probes stored in: {base_training_config['manifest_dir'].replace('_DATASET', '')}")

    # Add HOO metadata to unified manifest
    if results["folds"]:
        unified_manifest_dir = Path(str(base_training_config["manifest_dir"]).replace("_DATASET", ""))
        consolidate_hoo_metadata(results["folds"], unified_manifest_dir, datasets, hold_out_size)

    # Save summary
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "hoo_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to {summary_path}")

    return results


def main():
    """CLI entry point for HOO training."""
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run held-one-out training for probes across datasets"
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="Base config YAML (e.g., configs/probe_training/hoo_base.yaml)",
    )
    parser.add_argument("--datasets", type=str, nargs="+", help="Datasets to validate on (auto-detect if omitted)")
    parser.add_argument("--hold-out-size", type=int, default=1, help="Number of datasets to hold out per fold (default: 1)")
    parser.add_argument("--output-dir", type=Path, help="Directory for saving results (hoo_summary.json)")

    args = parser.parse_args()

    # Load base config
    with open(args.base_config) as f:
        base_config = yaml.safe_load(f)

    # Normalize datasets
    datasets = {d.lower() for d in args.datasets} if args.datasets else None

    # Run held-out training
    results = run_held_one_out(
        base_config,
        datasets=datasets,
        hold_out_size=args.hold_out_size,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 80)
    print("Held-one-out training complete!")
    print("Run 'run_hoo_evaluation.py' to evaluate probes on their held-out datasets")
    print("=" * 80)


if __name__ == "__main__":
    main()
