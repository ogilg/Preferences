"""Main training script for probe experiment."""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.measurement_storage.loading import load_raw_scores
from src.probes.config import ProbeTrainingConfig
from src.probes.storage import load_manifest, save_manifest, save_probe
from src.probes.activations import filter_activations_by_origin, load_activations
from src.probes.training import train_for_scores


def determine_training_combinations(config: ProbeTrainingConfig) -> list[dict]:
    """Returns list of dicts with keys: templates, datasets, response_formats, seeds, layer."""
    dataset_combos = config.dataset_combinations or [None]

    return [
        {
            "templates": templates,
            "datasets": datasets,
            "response_formats": response_formats,
            "seeds": seeds,
            "layer": layer,
        }
        for templates, response_formats, seeds, datasets, layer in product(
            config.template_combinations,
            config.response_format_combinations,
            config.seed_combinations,
            dataset_combos,
            config.layers,
        )
    ]


def train_probe_combination(
    config: ProbeTrainingConfig,
    templates: list[str],
    datasets: list[str] | None,
    response_formats: list[str],
    seeds: list[int],
    layer: int,
    probe_id: str,
) -> dict | None:
    """Train a single probe. Returns metadata dict or None if insufficient data."""
    import sys
    print(f"  [{probe_id}] Loading activations...", file=sys.stderr, flush=True)
    task_ids, activations = load_activations(config.activations_path.parent)

    print(f"  [{probe_id}] Loading measurements...", file=sys.stderr, flush=True)
    # Load raw measurements (without averaging) for all templates
    # Determine measurement directories and collect all measurements
    all_measurements = []
    for template in templates:
        task_type = "pre_task" if template.startswith("pre_task") else "post_task"
        measurement_dir = config.experiment_dir / f"{task_type}_stated"

        measurements = load_raw_scores(
            measurement_dir,
            [template],
            response_formats,
            seeds,
        )
        all_measurements.extend(measurements)

    if not all_measurements:
        print(f"  [{probe_id}] No measurements found", file=sys.stderr, flush=True)
        return None

    # Filter by dataset if needed
    if datasets:
        print(f"  [{probe_id}] Filtering by datasets: {datasets}...", file=sys.stderr, flush=True)
        # Filter activations by all specified datasets
        mask = np.zeros(len(task_ids), dtype=bool)
        for dataset in datasets:
            dataset_mask = filter_activations_by_origin(
                task_ids,
                dataset,
                config.activations_path.parent,
            )
            mask |= dataset_mask
        filtered_task_ids = task_ids[mask]
        filtered_activations = {l: a[mask] for l, a in activations.items()}
        print(f"  [{probe_id}] Kept {mask.sum()} / {len(task_ids)} samples", file=sys.stderr, flush=True)
    else:
        filtered_task_ids = task_ids
        filtered_activations = activations

    print(f"  [{probe_id}] Training probe...", file=sys.stderr, flush=True)
    # Train probe
    results, probes = train_for_scores(
        filtered_task_ids,
        filtered_activations,
        all_measurements,
        config.cv_folds,
        config.alpha_sweep_size,
    )

    if not results:
        print(f"  [{probe_id}] Training failed", file=sys.stderr, flush=True)
        return None

    layer_result = next((r for r in results if r["layer"] == layer), None)
    if layer_result is None:
        return None

    print(f"  [{probe_id}] Saving probe...", file=sys.stderr, flush=True)
    probe_weights = probes[layer]
    relative_path = save_probe(probe_weights, config.manifest_dir, probe_id)
    unique_tasks = len(set(tid for tid, _ in all_measurements))

    return {
        "id": probe_id,
        "file": relative_path,
        "templates": templates,
        "layer": layer,
        "datasets": datasets,  # None means "all datasets"
        "response_formats": response_formats,
        "seeds": seeds,
        "cv_r2_mean": layer_result["cv_r2_mean"],
        "cv_r2_std": layer_result["cv_r2_std"],
        "n_measurement_instances": layer_result["n_samples"],
        "n_unique_tasks": unique_tasks,
        "best_alpha": layer_result["best_alpha"],
        "condition_number": layer_result["condition_number"],
        "train_test_gap": layer_result["train_test_gap"],
        "cv_stability": layer_result["cv_stability"],
        "trained_at": datetime.now().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train probes for experiment")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML path")
    args = parser.parse_args()

    config = ProbeTrainingConfig.from_yaml(args.config)

    config.manifest_dir.mkdir(parents=True, exist_ok=True)
    probes_dir = config.manifest_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)

    print(f"Probe Training: {config.experiment_name}")
    print(f"Output: {probes_dir}")

    # Initialize or load manifest
    manifest_path = config.manifest_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(config.manifest_dir)
        next_id = len(manifest["probes"]) + 1
    else:
        manifest = {
            "experiment_name": config.experiment_name,
            "experiment_dir": str(config.experiment_dir),
            "created_at": datetime.now().isoformat(),
            "probes": [],
        }
        next_id = 1

    # Determine combinations
    combinations = determine_training_combinations(config)
    print(f"Training {len(combinations)} probes...\n")

    trained_count = 0
    for combo in tqdm(combinations, desc="Training probes", unit="probe"):
        probe_id = f"{next_id:04d}"

        result = train_probe_combination(
            config,
            combo["templates"],
            combo["datasets"],
            combo["response_formats"],
            combo["seeds"],
            combo["layer"],
            probe_id,
        )

        if result:
            manifest["probes"].append(result)
            next_id += 1
            trained_count += 1

    save_manifest(manifest, config.manifest_dir)
    print(f"\nTrained {trained_count}/{len(combinations)} probes")


if __name__ == "__main__":
    main()
