"""Main training script for probe experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.measurement_storage.loading import load_pooled_scores
from src.probes.config import ProbeTrainingConfig
from src.probes.storage import load_manifest, load_probe, save_manifest, save_probe
from src.probes.activations import filter_activations_by_origin, load_activations
from src.probes.training import train_for_scores


def determine_training_combinations(config: ProbeTrainingConfig) -> list[dict]:
    """Determine (template, dataset, layer) combinations to train.

    Returns:
        list of dicts with keys: template, datasets, layer
    """
    combinations = []

    for template in config.templates:
        if config.datasets:
            # Train separate probe per dataset
            for dataset in config.datasets:
                for layer in config.layers:
                    combinations.append({
                        "template": template,
                        "datasets": [dataset],
                        "layer": layer,
                    })
        else:
            # Pool all datasets
            for layer in config.layers:
                combinations.append({
                    "template": template,
                    "datasets": None,  # signals to use all
                    "layer": layer,
                })

    return combinations


def train_probe_combination(
    config: ProbeTrainingConfig,
    template: str,
    datasets: list[str] | None,
    layer: int,
    probe_id: str,
) -> dict | None:
    """Train a single probe for given template/dataset/layer combination.

    Returns:
        dict with probe metadata or None if insufficient data
    """
    print(f"\n  Training probe_{probe_id}: {template} layer={layer}", end="")
    if datasets:
        print(f" datasets={datasets}", end="")
    else:
        print(f" datasets=all", end="")
    print()

    # Load activations
    task_ids, activations = load_activations(config.activations_path.parent)

    # Determine measurement directory (pre_task_stated or post_task_stated)
    task_type = "pre_task" if template.startswith("pre_task") else "post_task"
    measurement_dir = config.experiment_dir / f"{task_type}_stated"

    # Load measurements
    scores = load_pooled_scores(
        measurement_dir,
        template,
        config.response_formats,
        config.seeds,
    )

    if not scores:
        print(f"    Skipped: no measurements found for {template}")
        return None

    # Filter by dataset if needed
    if datasets:
        mask = filter_activations_by_origin(
            task_ids,
            datasets[0],
            config.activations_path.parent,
        )
        filtered_task_ids = task_ids[mask]
        filtered_activations = {l: a[mask] for l, a in activations.items()}
    else:
        filtered_task_ids = task_ids
        filtered_activations = activations

    # Train probe
    results, probes = train_for_scores(
        filtered_task_ids,
        filtered_activations,
        scores,
        config.cv_folds,
    )

    if not results:
        print(f"    Skipped: insufficient samples")
        return None

    # Extract layer result
    layer_result = None
    for r in results:
        if r["layer"] == layer:
            layer_result = r
            break

    if layer_result is None:
        return None

    # Save probe
    probe_weights = probes[layer]
    relative_path = save_probe(probe_weights, config.output_dir, probe_id)

    print(f"    R² = {layer_result['cv_r2_mean']:.3f} ± {layer_result['cv_r2_std']:.3f}")
    print(f"    n_samples = {layer_result['n_samples']}, α = {layer_result['best_alpha']}")

    return {
        "id": probe_id,
        "file": relative_path,
        "template": template,
        "layer": layer,
        "datasets": datasets or [],
        "response_formats": config.response_formats,
        "seeds": config.seeds,
        "cv_r2_mean": layer_result["cv_r2_mean"],
        "cv_r2_std": layer_result["cv_r2_std"],
        "n_samples": layer_result["n_samples"],
        "best_alpha": layer_result["best_alpha"],
        "trained_at": datetime.now().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train probes for experiment")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML path")
    args = parser.parse_args()

    config = ProbeTrainingConfig.from_yaml(args.config)

    print(f"Probe Training Experiment: {config.experiment_name}")
    print(f"Output directory: {config.output_dir}")
    print(f"Templates: {config.templates}")
    if config.datasets:
        print(f"Datasets: {config.datasets}")
    else:
        print("Datasets: all (pooled)")
    print(f"Layers: {config.layers}")
    print()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize or load manifest
    manifest_path = config.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(config.output_dir)
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

    # Train probes with progress bar
    trained_count = 0
    pbar = tqdm(combinations, desc="Training probes", unit="probe")
    for combo in pbar:
        probe_id = f"{next_id:03d}"

        # Update progress bar description with current probe
        template_short = combo["template"].replace("post_task_", "")[:15]
        pbar.set_description(f"Training {template_short} L{combo['layer']}")

        result = train_probe_combination(
            config,
            combo["template"],
            combo["datasets"],
            combo["layer"],
            probe_id,
        )

        if result:
            manifest["probes"].append(result)
            next_id += 1
            trained_count += 1

    pbar.close()

    # Save manifest
    save_manifest(manifest, config.output_dir)
    print(f"\nTrained {trained_count}/{len(combinations)} probes")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
