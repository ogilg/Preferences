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
from src.probes.activations import load_activations, load_task_origins
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


def load_measurements_for_combo(
    config: ProbeTrainingConfig,
    templates: list[str],
    response_formats: list[str],
    seeds: list[int],
) -> dict[str, float]:
    """Load measurements and average scores per task_id."""
    raw_measurements = []
    for template in templates:
        task_type = "pre_task" if template.startswith("pre_task") else "post_task"
        measurement_dir = config.experiment_dir / f"{task_type}_stated"
        raw_measurements.extend(load_raw_scores(measurement_dir, [template], response_formats, seeds))

    # Average scores per task_id
    scores_by_task: dict[str, list[float]] = {}
    for task_id, score in raw_measurements:
        if task_id not in scores_by_task:
            scores_by_task[task_id] = []
        scores_by_task[task_id].append(score)

    return {tid: np.mean(scores) for tid, scores in scores_by_task.items()}


def filter_task_ids_by_datasets(
    task_ids: set[str],
    datasets: list[str] | None,
    origins_cache: dict[str, set[str]],
) -> set[str]:
    """Filter task IDs to only those in specified datasets. Returns all if datasets is None."""
    if not datasets:
        return task_ids
    target_ids = set()
    for dataset in datasets:
        target_ids.update(origins_cache.get(dataset.upper(), set()))
    return task_ids & target_ids


def collect_all_measurements(
    config: ProbeTrainingConfig,
    combinations: list[dict],
) -> dict[tuple, dict[str, float]]:
    """Load and cache averaged measurements for all unique template/format/seed combinations."""
    cache: dict[tuple, dict[str, float]] = {}
    for combo in combinations:
        key = (tuple(combo["templates"]), tuple(combo["response_formats"]), tuple(combo["seeds"]))
        if key not in cache:
            cache[key] = load_measurements_for_combo(
                config, combo["templates"], combo["response_formats"], combo["seeds"]
            )
    return cache


def collect_all_needed_task_ids(
    measurements_cache: dict[tuple, dict[str, float]],
    combinations: list[dict],
    origins_cache: dict[str, set[str]],
) -> set[str]:
    """Collect union of all task IDs needed across all combinations."""
    all_task_ids: set[str] = set()

    for combo in combinations:
        key = (tuple(combo["templates"]), tuple(combo["response_formats"]), tuple(combo["seeds"]))
        task_ids = set(measurements_cache[key].keys())
        task_ids = filter_task_ids_by_datasets(task_ids, combo["datasets"], origins_cache)
        all_task_ids.update(task_ids)

    return all_task_ids


def train_probe_combination(
    config: ProbeTrainingConfig,
    combo: dict,
    probe_id: str,
    all_task_ids: np.ndarray,
    all_activations: dict[int, np.ndarray],
    origins_cache: dict[str, set[str]],
    measurements_cache: dict[tuple, dict[str, float]],
) -> dict | None:
    """Train a single probe. Returns metadata dict or None if insufficient data."""
    key = (tuple(combo["templates"]), tuple(combo["response_formats"]), tuple(combo["seeds"]))
    scores_by_task = measurements_cache[key]

    if not scores_by_task:
        return None

    measurement_task_ids = set(scores_by_task.keys())
    measurement_task_ids = filter_task_ids_by_datasets(
        measurement_task_ids, combo["datasets"], origins_cache
    )

    # Filter scores to only include tasks in measurement_task_ids
    filtered_scores = {tid: scores_by_task[tid] for tid in measurement_task_ids if tid in scores_by_task}

    mask = np.array([tid in measurement_task_ids for tid in all_task_ids])
    filtered_task_ids = all_task_ids[mask]
    filtered_activations = {l: a[mask] for l, a in all_activations.items()}

    # Train probe
    results, probes = train_for_scores(
        filtered_task_ids,
        filtered_activations,
        filtered_scores,
        config.cv_folds,
        config.alpha_sweep_size,
    )

    if not results:
        return None

    layer = combo["layer"]
    layer_result = next((r for r in results if r["layer"] == layer), None)
    if layer_result is None:
        return None

    probe_weights = probes[layer]
    relative_path = save_probe(probe_weights, config.manifest_dir, probe_id)

    return {
        "id": probe_id,
        "file": relative_path,
        "templates": combo["templates"],
        "layer": layer,
        "datasets": combo["datasets"],
        "response_formats": combo["response_formats"],
        "seeds": combo["seeds"],
        "cv_r2_mean": layer_result["cv_r2_mean"],
        "cv_r2_std": layer_result["cv_r2_std"],
        "n_measurement_instances": layer_result["n_samples"],
        "n_unique_tasks": len(measurement_task_ids),
        "best_alpha": layer_result["best_alpha"],
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

    # Load data
    print("Loading data...", flush=True)
    origins_cache = load_task_origins(config.activations_path.parent)
    measurements_cache = collect_all_measurements(config, combinations)
    needed_task_ids = collect_all_needed_task_ids(measurements_cache, combinations, origins_cache)
    all_task_ids, all_activations = load_activations(
        config.activations_path.parent,
        task_id_filter=needed_task_ids,
        layers=config.layers,
    )
    print(f"Loaded {len(needed_task_ids)} tasks, {len(all_activations)} layers\n", flush=True)

    trained_count = 0
    for combo in tqdm(combinations, desc="Training probes", unit="probe"):
        probe_id = f"{next_id:04d}"

        result = train_probe_combination(
            config,
            combo,
            probe_id,
            all_task_ids,
            all_activations,
            origins_cache,
            measurements_cache,
        )

        if result:
            manifest["probes"].append(result)
            next_id += 1
            trained_count += 1

    save_manifest(manifest, config.manifest_dir)
    print(f"\nTrained {trained_count}/{len(combinations)} probes")


if __name__ == "__main__":
    main()
