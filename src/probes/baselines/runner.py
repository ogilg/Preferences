"""Unified runner for all baseline types."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.measurement.storage.loading import load_pooled_scores
from src.probes.core.activations import load_activations
from src.probes.config import ProbeConfig

from .noise import run_random_activations_baseline, run_shuffled_labels_baseline
from .task_description import run_task_description_baseline
from .types import BaselineResult, BaselineType


def run_all_baselines(
    config: ProbeConfig,
    task_description_dir: Path | None,
    n_noise_seeds: int = 5,
    alpha_sweep_size: int = 10,
    standardize: bool = False,
) -> list[BaselineResult]:
    """Run all baseline types for a probe training config.

    Args:
        config: ProbeConfig with training_data, layers, etc.
        task_description_dir: Directory containing task description activations.
            If None, skips task_description baseline.
        n_noise_seeds: Number of random seeds for noise baselines.

    Returns:
        List of all baseline results.
    """
    results: list[BaselineResult] = []

    # Load completion activations for noise baselines
    task_ids, activations = load_activations(config.activations_path.parent)

    # Flatten template combinations to unique templates
    templates = set()
    for combo in config.training_data.template_combinations:
        templates.update(combo)

    # Flatten seed combinations to list of seeds for loading
    seeds = []
    for combo in config.training_data.seed_combinations:
        seeds.extend(combo)
    seeds = list(set(seeds))

    # Build combinations to test
    combinations = []
    for template in templates:
        for layer in config.layers:
            combinations.append((template, layer))

    print(f"Running baselines for {len(combinations)} template/layer combinations...")
    print(f"  Noise seeds: {n_noise_seeds}")
    print(f"  Task description dir: {task_description_dir}")

    for template, layer in tqdm(combinations, desc="Baselines"):
        # Load scores for this template
        task_type = "pre_task" if template.startswith("pre_task") else "post_task"
        measurement_dir = config.training_data.experiment_dir / f"{task_type}_stated"

        scores = load_pooled_scores(measurement_dir, template, seeds)

        if not scores:
            tqdm.write(f"  Skipping {template} layer {layer}: no measurements found")
            continue

        # Filter activations to tasks with scores
        id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
        valid_indices = []
        valid_scores = []
        for task_id, score in scores.items():
            if task_id in id_to_idx:
                valid_indices.append(id_to_idx[task_id])
                valid_scores.append(score)

        if len(valid_indices) < config.cv_folds * 2:
            tqdm.write(f"  Skipping {template} layer {layer}: insufficient samples")
            continue

        indices = np.array(valid_indices)
        y = np.array(valid_scores)
        X = activations[layer][indices]

        # Run noise baselines
        for seed in range(n_noise_seeds):
            # Shuffled labels
            result = run_shuffled_labels_baseline(
                X, y, template, layer, config.cv_folds, seed,
                alpha_sweep_size=alpha_sweep_size, standardize=standardize,
            )
            results.append(result)

            # Random activations
            result = run_random_activations_baseline(
                X, y, template, layer, config.cv_folds, seed,
                alpha_sweep_size=alpha_sweep_size, standardize=standardize,
            )
            results.append(result)

        # Run task description baseline if available
        if task_description_dir is not None:
            result = run_task_description_baseline(
                task_description_dir, scores, template, layer, config.cv_folds
            )
            if result is not None:
                results.append(result)

    return results


def aggregate_noise_baselines(results: list[BaselineResult]) -> list[dict]:
    """Aggregate noise baselines across seeds, computing mean/std.

    Returns one entry per (baseline_type, template, layer) with aggregated stats.
    """
    from collections import defaultdict

    # Group by (baseline_type, template, layer)
    grouped: dict[tuple, list[BaselineResult]] = defaultdict(list)
    for r in results:
        if r.baseline_type in (BaselineType.SHUFFLED_LABELS, BaselineType.RANDOM_ACTIVATIONS):
            key = (r.baseline_type, r.template, r.layer)
            grouped[key].append(r)

    aggregated = []
    for (baseline_type, template, layer), items in grouped.items():
        r2_means = [r.cv_r2_mean for r in items]
        mse_means = [r.cv_mse_mean for r in items]

        aggregated.append({
            "baseline_type": baseline_type.value,
            "template": template,
            "layer": layer,
            "cv_r2_mean": float(np.mean(r2_means)),
            "cv_r2_std": float(np.std(r2_means)),
            "cv_mse_mean": float(np.mean(mse_means)),
            "cv_mse_std": float(np.std(mse_means)),
            "n_samples": items[0].n_samples,
            "n_seeds": len(items),
        })

    return aggregated
