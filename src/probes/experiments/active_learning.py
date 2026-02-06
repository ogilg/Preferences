"""Train probes from active learning experiment results.

Supports two training modes:
1. Ridge regression on Thurstonian mu values (utility scores)
2. Bradley-Terry on pairwise comparison data

Usage:
    python -m src.probes.experiments.active_learning --config configs/probes/example.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.measurement.storage.loading import load_run_utilities, load_yaml
from src.probes.core.activations import load_activations
from src.probes.core.storage import save_probe, save_manifest
from src.probes.core.training import train_for_scores
from src.probes.bradley_terry.data import PairwiseActivationData
from src.probes.bradley_terry.training import train_for_comparisons
from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, PreferenceType


@dataclass
class ActiveLearningConfig:
    experiment_name: str
    run_dir: Path
    activations_path: Path
    output_dir: Path
    layers: list[int]
    cv_folds: int = 5
    alpha_sweep_size: int = 50
    bt_lr: float = 0.01
    bt_l2_lambda: float = 1.0
    bt_batch_size: int = 64
    bt_max_epochs: int = 1000
    bt_patience: int = 10

    @classmethod
    def from_yaml(cls, path: Path) -> ActiveLearningConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            experiment_name=data["experiment_name"],
            run_dir=Path(data["run_dir"]),
            activations_path=Path(data["activations_path"]),
            output_dir=Path(data["output_dir"]),
            layers=data["layers"],
            cv_folds=data.get("cv_folds", 5),
            alpha_sweep_size=data.get("alpha_sweep_size", 50),
            bt_lr=data.get("bt_lr", 0.01),
            bt_l2_lambda=data.get("bt_l2_lambda", 1.0),
            bt_batch_size=data.get("bt_batch_size", 64),
            bt_max_epochs=data.get("bt_max_epochs", 1000),
            bt_patience=data.get("bt_patience", 10),
        )


def _load_thurstonian_scores(run_dir: Path) -> dict[str, float]:
    """Load task_id -> mu mapping from Thurstonian fit."""
    mu_array, task_ids = load_run_utilities(run_dir)
    return dict(zip(task_ids, mu_array))


def _load_pairwise_measurements(run_dir: Path) -> list[BinaryPreferenceMeasurement]:
    """Load measurements and reconstruct as BinaryPreferenceMeasurement objects."""
    measurements_path = run_dir / "measurements.yaml"
    raw = load_yaml(measurements_path)

    measurements = []
    for m in raw:
        task_a = Task(
            id=m["task_a"],
            prompt="",
            origin=OriginDataset[m["origin_a"]],
            metadata={},
        )
        task_b = Task(
            id=m["task_b"],
            prompt="",
            origin=OriginDataset[m["origin_b"]],
            metadata={},
        )
        measurements.append(BinaryPreferenceMeasurement(
            task_a=task_a,
            task_b=task_b,
            choice=m["choice"],
            preference_type=PreferenceType.POST_TASK_REVEALED,
        ))

    return measurements


def _train_ridge_probes(
    config: ActiveLearningConfig,
    task_ids: np.ndarray,
    activations: dict[int, np.ndarray],
    scores: dict[str, float],
) -> list[dict]:
    """Train Ridge regression probes on Thurstonian mu values."""
    results, probes = train_for_scores(
        task_ids=task_ids,
        activations=activations,
        scores=scores,
        cv_folds=config.cv_folds,
        alpha_sweep_size=config.alpha_sweep_size,
    )

    probe_entries = []
    for result in results:
        layer = result["layer"]
        probe_id = f"ridge_L{layer:02d}"
        relative_path = save_probe(probes[layer], config.output_dir, probe_id)

        probe_entries.append({
            "id": probe_id,
            "file": relative_path,
            "method": "ridge",
            "layer": layer,
            "cv_r2_mean": result["cv_r2_mean"],
            "cv_r2_std": result["cv_r2_std"],
            "cv_mse_mean": result["cv_mse_mean"],
            "cv_mse_std": result["cv_mse_std"],
            "best_alpha": result["best_alpha"],
            "n_samples": result["n_samples"],
            "train_test_gap": result["train_test_gap"],
            "cv_stability": result["cv_stability"],
        })

    return probe_entries


def _train_bt_probes(
    config: ActiveLearningConfig,
    task_ids: np.ndarray,
    activations: dict[int, np.ndarray],
    measurements: list[BinaryPreferenceMeasurement],
) -> tuple[list[dict], int]:
    """Train Bradley-Terry probes on pairwise comparisons.

    Returns (probe_entries, n_pairs_used).
    """
    data = PairwiseActivationData.from_measurements(measurements, task_ids, activations)
    n_pairs_used = len(data.pairs)

    if n_pairs_used == 0:
        return [], 0

    results, probes = train_for_comparisons(
        task_ids=task_ids,
        activations=activations,
        measurements=measurements,
        lr=config.bt_lr,
        l2_lambda=config.bt_l2_lambda,
        batch_size=config.bt_batch_size,
        max_epochs=config.bt_max_epochs,
        patience=config.bt_patience,
    )

    probe_entries = []
    for result in results:
        layer = result.layer
        probe_id = f"bt_L{layer:02d}"
        relative_path = save_probe(probes[layer], config.output_dir, probe_id)

        probe_entries.append({
            "id": probe_id,
            "file": relative_path,
            "method": "bradley_terry",
            "layer": layer,
            "train_accuracy": result.train_accuracy,
            "train_loss": result.train_loss,
            "n_epochs": result.n_epochs,
            "n_pairs": n_pairs_used,
        })

    return probe_entries, n_pairs_used


def run_active_learning_probes(config: ActiveLearningConfig) -> dict:
    """Train Ridge and Bradley-Terry probes from active learning data."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training probes: {config.experiment_name}")
    print(f"Run dir: {config.run_dir}")
    print(f"Output: {config.output_dir}")

    # Load data
    print("\nLoading data...")
    scores = _load_thurstonian_scores(config.run_dir)
    measurements = _load_pairwise_measurements(config.run_dir)
    print(f"  Loaded {len(scores)} task scores from Thurstonian fit")
    print(f"  Loaded {len(measurements)} pairwise comparisons")

    task_id_filter = set(scores.keys())
    task_ids, activations = load_activations(
        config.activations_path,
        task_id_filter=task_id_filter,
        layers=config.layers,
    )
    print(f"  Loaded activations for {len(task_ids)} tasks, {len(activations)} layers")

    manifest = {
        "experiment_name": config.experiment_name,
        "run_dir": str(config.run_dir),
        "activations_path": str(config.activations_path),
        "created_at": datetime.now().isoformat(),
        "n_tasks_in_experiment": len(scores),
        "n_tasks_with_activations": len(task_ids),
        "n_comparisons_in_experiment": len(measurements),
        "probes": [],
    }

    # Train Ridge probes
    print("\nTraining Ridge probes on Thurstonian mu...")
    ridge_entries = _train_ridge_probes(config, task_ids, activations, scores)
    manifest["probes"].extend(ridge_entries)
    print(f"  Trained {len(ridge_entries)} Ridge probes")

    if ridge_entries:
        best_ridge = max(ridge_entries, key=lambda x: x["cv_r2_mean"])
        print(f"  Best layer: {best_ridge['layer']} (R²={best_ridge['cv_r2_mean']:.4f})")

    # Train Bradley-Terry probes
    print("\nTraining Bradley-Terry probes on pairwise comparisons...")
    bt_entries, n_pairs_used = _train_bt_probes(config, task_ids, activations, measurements)
    manifest["probes"].extend(bt_entries)
    print(f"  Trained {len(bt_entries)} BT probes on {n_pairs_used} pairs")

    if bt_entries:
        best_bt = max(bt_entries, key=lambda x: x["train_accuracy"])
        print(f"  Best layer: {best_bt['layer']} (accuracy={best_bt['train_accuracy']:.4f})")

    save_manifest(manifest, config.output_dir)
    print(f"\nSaved manifest with {len(manifest['probes'])} probes")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train probes from active learning results")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML path")
    args = parser.parse_args()

    config = ActiveLearningConfig.from_yaml(args.config)
    run_active_learning_probes(config)


if __name__ == "__main__":
    main()
