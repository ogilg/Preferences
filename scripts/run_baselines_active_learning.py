"""Run noise baselines for run-dir probe experiment.

Loads the same config and data as the run-dir probe training,
using identical alpha sweep and standardization settings.

Usage:
    python -m scripts.run_baselines_active_learning --config configs/probes/gemma3_completion_preference.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from src.probes.core.activations import load_activations
from src.probes.baselines.noise import (
    run_shuffled_labels_baseline,
    run_random_activations_baseline,
)
from src.probes.data_loading import load_thurstonian_scores
from src.probes.experiments.run_dir_probes import RunDirProbeConfig

N_SEEDS = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    config = RunDirProbeConfig.from_yaml(args.config)
    output_dir = Path("results/baselines") / config.output_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running baselines for: {config.experiment_name}")
    print(f"  Alpha sweep size: {config.alpha_sweep_size}")
    print(f"  Standardize: {config.standardize}")
    print(f"  CV folds: {config.cv_folds}")

    scores = load_thurstonian_scores(config.run_dir)
    print(f"  Loaded {len(scores)} task scores")

    task_id_filter = set(scores.keys())
    results = []

    for layer in config.layers:
        print(f"\n--- Layer {layer} ---")
        task_ids, activations = load_activations(
            config.activations_path,
            task_id_filter=task_id_filter,
            layers=[layer],
        )

        id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
        valid_indices = []
        valid_scores = []
        for task_id, score in scores.items():
            if task_id in id_to_idx:
                valid_indices.append(id_to_idx[task_id])
                valid_scores.append(score)

        indices = np.array(valid_indices)
        y = np.array(valid_scores)
        X = activations[layer][indices]
        print(f"  {len(y)} samples, {X.shape[1]} features")

        for seed in range(args.n_seeds):
            sl = run_shuffled_labels_baseline(
                X, y, "completion_preference", layer, config.cv_folds, seed,
                alpha_sweep_size=config.alpha_sweep_size, standardize=config.standardize,
            )
            results.append(sl)
            print(f"  Seed {seed} | shuffled_labels R²={sl.cv_r2_mean:.4f} (best_alpha={sl.best_alpha:.0f})")

            ra = run_random_activations_baseline(
                X, y, "completion_preference", layer, config.cv_folds, seed,
                alpha_sweep_size=config.alpha_sweep_size, standardize=config.standardize,
            )
            results.append(ra)
            print(f"  Seed {seed} | random_activations R²={ra.cv_r2_mean:.4f} (best_alpha={ra.best_alpha:.0f})")

        del activations
        gc.collect()

    # Aggregate across seeds
    grouped: dict[tuple, list] = defaultdict(list)
    for r in results:
        grouped[(r.baseline_type.value, r.layer)].append(r)

    aggregated = []
    for (btype, layer), items in sorted(grouped.items()):
        r2s = [r.cv_r2_mean for r in items]
        mses = [r.cv_mse_mean for r in items]
        entry = {
            "baseline_type": btype,
            "layer": layer,
            "cv_r2_mean": float(np.mean(r2s)),
            "cv_r2_std": float(np.std(r2s)),
            "cv_mse_mean": float(np.mean(mses)),
            "cv_mse_std": float(np.std(mses)),
            "best_alpha_mode": float(max(set(r.best_alpha for r in items), key=lambda a: sum(1 for r in items if r.best_alpha == a))),
            "n_samples": items[0].n_samples,
            "n_seeds": len(items),
            "per_seed_r2": [float(r) for r in r2s],
        }
        aggregated.append(entry)
        print(f"\n{btype} L{layer}: R²={entry['cv_r2_mean']:.4f} ± {entry['cv_r2_std']:.4f}")

    manifest = {
        "experiment_name": f"{config.experiment_name}_baselines",
        "reference_config": str(args.config),
        "alpha_sweep_size": config.alpha_sweep_size,
        "standardize": config.standardize,
        "cv_folds": config.cv_folds,
        "created_at": datetime.now().isoformat(),
        "n_noise_seeds": args.n_seeds,
        "layers": config.layers,
        "baselines": aggregated,
    }

    manifest_path = output_dir / "baselines_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved to {manifest_path}")


if __name__ == "__main__":
    main()
