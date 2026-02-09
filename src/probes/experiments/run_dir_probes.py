"""Train probes from a measurement run directory.

Supports two training modes:
1. Ridge regression on Thurstonian mu values (utility scores)
2. Bradley-Terry on pairwise comparison data

Usage:
    python -m src.probes.experiments.run_dir_probes --config configs/probes/example.yaml
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import yaml

from src.probes.core.activations import load_activations
from src.probes.core.storage import save_probe, save_manifest
from src.probes.core.training import train_for_scores
from src.probes.bradley_terry.data import PairwiseActivationData
from src.probes.bradley_terry.training import train_for_comparisons
from src.probes.data_loading import load_thurstonian_scores, load_pairwise_measurements
from src.probes.residualization import residualize_scores


class ProbeMode(Enum):
    RIDGE = "ridge"
    BRADLEY_TERRY = "bradley_terry"


@dataclass
class RunDirProbeConfig:
    experiment_name: str
    run_dir: Path
    activations_path: Path
    output_dir: Path
    layers: list[int]
    modes: list[ProbeMode]
    cv_folds: int = 5
    alpha_sweep_size: int = 50
    standardize: bool = False
    bt_l2_lambda: float = 1.0
    residualize: bool = False
    residualize_include_dataset: bool = False
    topics_json: Path | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> RunDirProbeConfig:
        with open(path) as f:
            data = yaml.safe_load(f)

        modes = [ProbeMode(m) for m in data.get("modes", ["ridge", "bradley_terry"])]
        topics_json = Path(data["topics_json"]) if "topics_json" in data else None

        # Only pass optional keys present in YAML; let dataclass defaults handle the rest
        optional = {}
        for key in ("cv_folds", "alpha_sweep_size", "standardize", "bt_l2_lambda", "residualize", "residualize_include_dataset"):
            if key in data:
                optional[key] = data[key]

        return cls(
            experiment_name=data["experiment_name"],
            run_dir=Path(data["run_dir"]),
            activations_path=Path(data["activations_path"]),
            output_dir=Path(data["output_dir"]),
            layers=data["layers"],
            modes=modes,
            topics_json=topics_json,
            **optional,
        )


def _train_ridge_probes(
    config: RunDirProbeConfig,
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
        standardize=config.standardize,
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
            "train_r2": result["train_r2"],
            "n_samples": result["n_samples"],
            "train_test_gap": result["train_test_gap"],
            "cv_stability": result["cv_stability"],
            "alpha_sweep": result["alpha_sweep"],
        })

    return probe_entries


def _train_bt_probes(
    config: RunDirProbeConfig,
    data: PairwiseActivationData,
) -> tuple[list[dict], int]:
    """Train Bradley-Terry probes on pairwise comparisons.

    Returns (probe_entries, n_pairs_used).
    """
    n_pairs_used = len(data.pairs)

    if n_pairs_used == 0:
        return [], 0

    results, probes = train_for_comparisons(
        data=data,
        l2_lambda=config.bt_l2_lambda,
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
            "n_iterations": result.n_iterations,
            "n_pairs": n_pairs_used,
        })

    return probe_entries, n_pairs_used


def run_probes(config: RunDirProbeConfig) -> dict:
    """Train Ridge and Bradley-Terry probes from run directory data.

    Processes one layer at a time to limit memory usage.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    run_ridge = ProbeMode.RIDGE in config.modes
    run_bt = ProbeMode.BRADLEY_TERRY in config.modes
    mode_names = [m.value for m in config.modes]

    print(f"Training probes: {config.experiment_name}")
    print(f"Modes: {mode_names}")
    print(f"Run dir: {config.run_dir}")
    print(f"Output: {config.output_dir}")

    # Load measurement data
    print("\nLoading measurement data...")
    scores = load_thurstonian_scores(config.run_dir) if run_ridge else {}
    measurements = load_pairwise_measurements(config.run_dir) if run_bt else []
    if scores:
        print(f"  Loaded {len(scores)} task scores from Thurstonian fit")
    if measurements:
        print(f"  Loaded {len(measurements)} pairwise comparisons")

    # Optionally residualize scores against metadata confounds
    metadata_stats = None
    if config.residualize and scores:
        assert config.topics_json is not None, "topics_json required when residualize=True"
        print("\nResidualizing scores against metadata confounds...")
        scores, metadata_stats = residualize_scores(scores, config.topics_json)
        print(f"  Metadata R²={metadata_stats['metadata_r2']:.4f} "
              f"({metadata_stats['n_metadata_features']} features)")
        print(f"  {metadata_stats['n_tasks_residualized']} tasks retained")

    task_id_filter = set(scores.keys()) if scores else None

    # Load one layer to get task_ids and count
    task_ids, first_layer_acts = load_activations(
        config.activations_path,
        task_id_filter=task_id_filter,
        layers=[config.layers[0]],
    )
    n_tasks = len(task_ids)
    del first_layer_acts
    gc.collect()

    manifest = {
        "experiment_name": config.experiment_name,
        "run_dir": str(config.run_dir),
        "activations_path": str(config.activations_path),
        "modes": mode_names,
        "created_at": datetime.now().isoformat(),
        "n_tasks_in_experiment": len(scores),
        "n_tasks_with_activations": n_tasks,
        "n_comparisons_in_experiment": len(measurements),
        "residualized": config.residualize,
        "probes": [],
    }
    if metadata_stats is not None:
        manifest["metadata_r2"] = metadata_stats["metadata_r2"]
        manifest["metadata_features"] = metadata_stats["metadata_features"]
        manifest["n_tasks_residualized"] = metadata_stats["n_tasks_residualized"]
        manifest["n_tasks_dropped"] = metadata_stats["n_tasks_dropped"]

    # Process one layer at a time
    for layer in config.layers:
        print(f"\n--- Layer {layer} ---")
        task_ids, activations = load_activations(
            config.activations_path,
            task_id_filter=task_id_filter,
            layers=[layer],
        )

        if run_ridge:
            print(f"  Training Ridge probe...")
            ridge_entries = _train_ridge_probes(config, task_ids, activations, scores)
            manifest["probes"].extend(ridge_entries)
            if ridge_entries:
                print(f"  Ridge R²={ridge_entries[0]['cv_r2_mean']:.4f}")

        if run_bt:
            print(f"  Training BT probe...")
            bt_data = PairwiseActivationData.from_measurements(measurements, task_ids, activations)
            bt_entries, n_pairs = _train_bt_probes(config, bt_data)
            manifest["probes"].extend(bt_entries)
            if bt_entries:
                print(f"  BT accuracy={bt_entries[0]['train_accuracy']:.4f} ({n_pairs} pairs)")

        del activations
        gc.collect()

    # Summary
    ridge_probes = [p for p in manifest["probes"] if p["method"] == "ridge"]
    bt_probes = [p for p in manifest["probes"] if p["method"] == "bradley_terry"]

    if ridge_probes:
        best = max(ridge_probes, key=lambda x: x["cv_r2_mean"])
        print(f"\nBest Ridge: layer {best['layer']} (R²={best['cv_r2_mean']:.4f})")
    if bt_probes:
        best = max(bt_probes, key=lambda x: x["train_accuracy"])
        print(f"Best BT: layer {best['layer']} (accuracy={best['train_accuracy']:.4f})")

    save_manifest(manifest, config.output_dir)
    print(f"\nSaved manifest with {len(manifest['probes'])} probes")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train probes from run directory results")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML path")
    args = parser.parse_args()

    config = RunDirProbeConfig.from_yaml(args.config)
    run_probes(config)


if __name__ == "__main__":
    main()
