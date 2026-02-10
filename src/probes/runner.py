"""Unified probe training and evaluation entry point."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.probes.config import ProbeConfig, ProbeType, DataSpec
from src.probes.core.activations import load_activations, load_task_origins
from src.probes.core.evaluate import evaluate_probe_on_data
from src.probes.core.linear_probe import train_and_evaluate
from src.probes.core.storage import save_probe, load_probe, save_manifest, load_manifest
from src.probes.data_loading import (
    load_measurements_for_templates,
    filter_task_ids_by_datasets,
    expand_training_combinations,
    collect_measurements_cache,
    collect_needed_task_ids,
)
from src.measurement.storage.loading import load_run_utilities


def _train_single_probe(
    combo: dict,
    probe_id: str,
    all_task_ids: np.ndarray,
    all_activations: dict[int, np.ndarray],
    origins_cache: dict[str, set[str]],
    measurements_cache: dict[tuple, dict[str, float]],
    output_dir: Path,
    cv_folds: int,
    alpha_sweep_size: int,
) -> dict | None:
    """Train a single probe. Returns metadata dict or None if insufficient data."""
    key = (tuple(combo["templates"]), tuple(combo["seeds"]))
    scores_by_task = measurements_cache[key]

    if not scores_by_task:
        return None

    measurement_task_ids = set(scores_by_task.keys())
    measurement_task_ids = filter_task_ids_by_datasets(
        measurement_task_ids, combo["datasets"], origins_cache
    )

    filtered_scores = {tid: scores_by_task[tid] for tid in measurement_task_ids if tid in scores_by_task}

    mask = np.array([tid in measurement_task_ids for tid in all_task_ids], dtype=bool)
    filtered_task_ids = all_task_ids[mask]
    layer = combo["layer"]
    X = all_activations[layer][mask]

    # Build indices/y mapping scores to activation rows
    id_to_idx = {tid: i for i, tid in enumerate(filtered_task_ids)}
    valid_indices = []
    valid_scores = []
    for task_id, score in filtered_scores.items():
        if task_id in id_to_idx:
            valid_indices.append(id_to_idx[task_id])
            valid_scores.append(score)

    if len(valid_indices) < cv_folds * 2:
        return None

    indices = np.array(valid_indices)
    y = np.array(valid_scores)

    probe, eval_results, _ = train_and_evaluate(
        X[indices], y, cv_folds=cv_folds, alpha_sweep_size=alpha_sweep_size,
    )

    probe_weights = np.append(probe.coef_, probe.intercept_)
    relative_path = save_probe(probe_weights, output_dir, probe_id)

    return {
        "id": probe_id,
        "file": relative_path,
        "templates": combo["templates"],
        "layer": layer,
        "datasets": combo["datasets"],
        "seeds": combo["seeds"],
        "cv_r2_mean": eval_results["cv_r2_mean"],
        "cv_r2_std": eval_results["cv_r2_std"],
        "n_measurement_instances": len(y),
        "n_unique_tasks": len(measurement_task_ids),
        "best_alpha": eval_results["best_alpha"],
        "train_test_gap": eval_results["train_test_gap"],
        "cv_stability": eval_results["cv_stability"],
        "trained_at": datetime.now().isoformat(),
    }


def run_training(config: ProbeConfig) -> dict:
    """Train probes according to config, returning manifest dict."""
    if config.probe_type == ProbeType.BRADLEY_TERRY:
        raise NotImplementedError(
            "Bradley-Terry training via the grid runner is not supported because BT "
            "requires pairwise data. Use experiments/run_dir_probes.py instead."
        )

    data = config.training_data

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "probes").mkdir(parents=True, exist_ok=True)

    print(f"Probe Training: {config.experiment_name}")
    print(f"Output: {config.output_dir}")

    # Initialize or load manifest
    manifest_path = config.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(config.output_dir)
        next_id = len(manifest["probes"]) + 1
    else:
        manifest = {
            "experiment_name": config.experiment_name,
            "experiment_dir": str(data.experiment_dir),
            "created_at": datetime.now().isoformat(),
            "probes": [],
        }
        next_id = 1

    combinations = expand_training_combinations(
        data.template_combinations,
        data.seed_combinations,
        config.layers,
        data.dataset_combinations,
    )
    print(f"Training {len(combinations)} probes...\n")

    # Load data
    print("Loading data...", flush=True)
    origins_cache = load_task_origins(config.activations_path.parent)
    measurements_cache = collect_measurements_cache(data.experiment_dir, combinations)
    needed_task_ids = collect_needed_task_ids(measurements_cache, combinations, origins_cache)
    all_task_ids, all_activations = load_activations(
        config.activations_path,
        task_id_filter=needed_task_ids,
        layers=config.layers,
    )
    print(f"Loaded {len(needed_task_ids)} tasks, {len(all_activations)} layers\n", flush=True)

    trained_count = 0
    for combo in tqdm(combinations, desc="Training probes", unit="probe"):
        probe_id = f"{next_id:04d}"

        result = _train_single_probe(
            combo,
            probe_id,
            all_task_ids,
            all_activations,
            origins_cache,
            measurements_cache,
            config.output_dir,
            config.cv_folds,
            config.alpha_sweep_size,
        )

        if result:
            manifest["probes"].append(result)
            next_id += 1
            trained_count += 1

    save_manifest(manifest, config.output_dir)
    print(f"\nTrained {trained_count}/{len(combinations)} probes")

    return manifest


def run_evaluation(config: ProbeConfig, manifest: dict | None = None) -> list[dict]:
    """Evaluate probes on evaluation_data spec."""
    if config.evaluation_data is None:
        raise ValueError("evaluation_data must be specified in config for evaluation")

    if manifest is None:
        manifest = load_manifest(config.output_dir)

    eval_data = config.evaluation_data

    # Load activations
    print(f"Loading activations from {config.activations_path}")
    task_ids, activations_dict = load_activations(config.activations_path)

    eval_combinations = expand_training_combinations(
        eval_data.template_combinations,
        eval_data.seed_combinations,
        config.layers,
        eval_data.dataset_combinations,
    )

    # Load origins for dataset filtering
    origins_cache = load_task_origins(config.activations_path.parent)

    all_results = []

    for eval_combo in eval_combinations:
        template = eval_combo["templates"][0]
        seeds = eval_combo["seeds"]
        layer = eval_combo["layer"]
        datasets = eval_combo["datasets"]

        # Find measurement run directory
        if template.startswith("pre_task"):
            search_dir = eval_data.experiment_dir / "pre_task_stated"
        else:
            search_dir = eval_data.experiment_dir / "post_task_stated"

        run_dir = None
        if search_dir.exists():
            for seed in seeds:
                for child in search_dir.iterdir():
                    if not child.is_dir():
                        continue
                    if template in child.name and f"seed{seed}" in child.name:
                        run_dir = child
                        break
                if run_dir:
                    break

        if run_dir is None:
            print(f"Warning: No measurement run found for {template}, seeds={seeds}")
            continue

        try:
            scores, task_ids_scores = load_run_utilities(run_dir)
        except FileNotFoundError as e:
            print(f"Warning: Could not load utilities from {run_dir}: {e}")
            continue

        # Filter by dataset if requested
        if datasets is not None:
            valid_task_ids = set()
            for dataset in datasets:
                valid_task_ids.update(origins_cache.get(dataset.upper(), set()))
            mask = np.array([tid in valid_task_ids for tid in task_ids_scores])
            scores = scores[mask]
            task_ids_scores = [tid for tid, m in zip(task_ids_scores, mask) if m]

        # Evaluate each probe at this layer
        for probe_meta in manifest["probes"]:
            if probe_meta["layer"] != layer:
                continue

            probe_id = probe_meta["id"]
            probe_weights = load_probe(config.output_dir, probe_id)
            X = activations_dict[layer]

            eval_result = evaluate_probe_on_data(
                probe_weights=probe_weights,
                activations=X,
                scores=scores,
                task_ids_data=task_ids,
                task_ids_scores=task_ids_scores,
            )

            all_results.append({
                "probe_id": probe_id,
                "layer": layer,
                "eval_template": template,
                "eval_seeds": seeds,
                "eval_datasets": datasets,
                "eval_metrics": eval_result,
            })

            r2_str = f"{eval_result['r2']:.4f}" if eval_result['r2'] is not None else "N/A"
            print(f"  Probe {probe_id} (L{layer}): R²={r2_str}, n={eval_result['n_samples']}")

    return all_results


def run(config: ProbeConfig) -> dict:
    """Train and evaluate probes."""
    manifest = run_training(config)

    evaluation_results = []
    if config.evaluation_data is not None:
        evaluation_results = run_evaluation(config, manifest)

    return {
        "training": manifest,
        "evaluation": evaluation_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified probe training and evaluation")
    parser.add_argument("config", type=Path, help="Config YAML path")
    parser.add_argument("--skip-training", action="store_true", help="Skip training phase")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation phase")
    args = parser.parse_args()

    config = ProbeConfig.from_yaml(args.config)

    if not args.skip_training:
        manifest = run_training(config)
    else:
        manifest = load_manifest(config.output_dir)

    if not args.skip_evaluation and config.evaluation_data is not None:
        run_evaluation(config, manifest)


if __name__ == "__main__":
    main()
