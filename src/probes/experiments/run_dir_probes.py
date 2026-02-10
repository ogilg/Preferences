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
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from src.analysis.probe.plot_hoo import plot_hoo_summary
from src.probes.core.activations import load_activations
from src.probes.core.linear_probe import train_and_evaluate
from src.probes.core.storage import save_probe, save_manifest
from src.probes.bradley_terry.data import PairwiseActivationData
from src.probes.bradley_terry.training import train_bt
from src.probes.data_loading import load_thurstonian_scores, load_pairwise_measurements
from src.probes.experiments import hoo_ridge, hoo_bt
from src.probes.experiments.hoo_ridge import build_ridge_xy
from src.probes.residualization import build_task_groups, residualize_scores


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
    residualize_confounds: list[str] | None = None
    topics_json: Path | None = None
    n_jobs: int = 1  # parallel workers for lambda sweep (1=sequential, -1=all cores)
    # HOO settings — if hoo_grouping is set, runs HOO instead of standard training
    hoo_grouping: str | None = None  # "topic" | "dataset"
    hoo_hold_out_size: int = 1
    hoo_groups: list[str] | None = None  # specific groups, or None for all

    @classmethod
    def from_yaml(cls, path: Path) -> RunDirProbeConfig:
        with open(path) as f:
            data = yaml.safe_load(f)

        modes = [ProbeMode(m) for m in data.get("modes", ["ridge", "bradley_terry"])]
        topics_json = Path(data["topics_json"]) if "topics_json" in data else None

        optional = {}
        for key in (
            "cv_folds", "alpha_sweep_size", "standardize", "residualize_confounds",
            "n_jobs", "hoo_grouping", "hoo_hold_out_size", "hoo_groups",
        ):
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


def _train_ridge_probe(
    config: RunDirProbeConfig,
    layer: int,
    task_ids: np.ndarray,
    activations: np.ndarray,
    scores: dict[str, float],
) -> dict | None:
    """Train a single Ridge probe. Returns entry dict or None if insufficient data."""
    indices, y = build_ridge_xy(task_ids, scores)
    if len(indices) < config.cv_folds * 2:
        return None

    X = activations[indices]
    probe, eval_results, alpha_sweep = train_and_evaluate(
        X, y, cv_folds=config.cv_folds,
        alpha_sweep_size=config.alpha_sweep_size,
        standardize=config.standardize,
    )
    weights = np.append(probe.coef_, probe.intercept_)
    probe_id = f"ridge_L{layer:02d}"
    relative_path = save_probe(weights, config.output_dir, probe_id)

    return {
        "id": probe_id,
        "file": relative_path,
        "method": "ridge",
        "layer": layer,
        "cv_r2_mean": eval_results["cv_r2_mean"],
        "cv_r2_std": eval_results["cv_r2_std"],
        "cv_mse_mean": eval_results["cv_mse_mean"],
        "cv_mse_std": eval_results["cv_mse_std"],
        "best_alpha": eval_results["best_alpha"],
        "train_r2": eval_results["train_r2"],
        "n_samples": len(y),
        "train_test_gap": eval_results["train_test_gap"],
        "cv_stability": eval_results["cv_stability"],
        "alpha_sweep": alpha_sweep,
    }


def _train_bt_probe(
    config: RunDirProbeConfig,
    data: PairwiseActivationData,
    layer: int,
) -> dict | None:
    """Train a single BT probe. Returns entry dict or None if no pairs."""
    if len(data.pairs) == 0:
        return None

    result = train_bt(data, layer, n_jobs=config.n_jobs)
    probe_id = f"bt_L{layer:02d}"
    relative_path = save_probe(result.weights, config.output_dir, probe_id)

    return {
        "id": probe_id,
        "file": relative_path,
        "method": "bradley_terry",
        "layer": layer,
        "train_accuracy": result.train_accuracy,
        "train_loss": result.train_loss,
        "cv_accuracy_mean": result.cv_accuracy_mean,
        "cv_accuracy_std": result.cv_accuracy_std,
        "best_l2_lambda": result.best_l2_lambda,
        "n_iterations": result.n_iterations,
        "n_pairs": len(data.pairs),
        "lambda_sweep": result.lambda_sweep,
    }


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
    if config.residualize_confounds and scores:
        assert config.topics_json is not None, "topics_json required for residualization"
        print(f"\nResidualizing scores against: {config.residualize_confounds}")
        scores, metadata_stats = residualize_scores(
            scores, config.topics_json, confounds=config.residualize_confounds,
        )
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
        "residualize_confounds": config.residualize_confounds,
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
            ridge_entry = _train_ridge_probe(config, layer, task_ids, activations[layer], scores)
            if ridge_entry:
                manifest["probes"].append(ridge_entry)
                print(f"  Ridge cv_R²={ridge_entry['cv_r2_mean']:.4f}")

        if run_bt:
            print(f"  Training BT probe...")
            bt_data = PairwiseActivationData.from_measurements(measurements, task_ids, activations)
            bt_entry = _train_bt_probe(config, bt_data, layer)
            if bt_entry:
                manifest["probes"].append(bt_entry)
                print(f"  BT cv_acc={bt_entry['cv_accuracy_mean']:.4f} "
                      f"(best l2={bt_entry['best_l2_lambda']:.4g}, {bt_entry['n_pairs']} pairs)")

        del activations
        gc.collect()

    # Summary
    ridge_probes = [p for p in manifest["probes"] if p["method"] == "ridge"]
    bt_probes = [p for p in manifest["probes"] if p["method"] == "bradley_terry"]

    if ridge_probes:
        best = max(ridge_probes, key=lambda x: x["cv_r2_mean"])
        print(f"\nBest Ridge: layer {best['layer']} (cv_R²={best['cv_r2_mean']:.4f})")
    if bt_probes:
        best = max(bt_probes, key=lambda x: x["cv_accuracy_mean"])
        print(f"Best BT: layer {best['layer']} (cv_acc={best['cv_accuracy_mean']:.4f}, "
              f"l2={best['best_l2_lambda']:.4g})")

    save_manifest(manifest, config.output_dir)
    print(f"\nSaved manifest with {len(manifest['probes'])} probes")

    return manifest


def run_hoo(config: RunDirProbeConfig) -> dict:
    """Held-one-out training and evaluation by group (topic or dataset)."""
    assert config.hoo_grouping is not None
    config.output_dir.mkdir(parents=True, exist_ok=True)

    run_ridge = ProbeMode.RIDGE in config.modes
    run_bt = ProbeMode.BRADLEY_TERRY in config.modes

    print(f"HOO Probes: {config.experiment_name}")
    print(f"Grouping: {config.hoo_grouping}, hold_out_size: {config.hoo_hold_out_size}")
    print(f"Modes: {[m.value for m in config.modes]}")
    print(f"Run dir: {config.run_dir}")
    print(f"Output: {config.output_dir}")

    # Load scores
    scores = load_thurstonian_scores(config.run_dir) if run_ridge else {}
    measurements = load_pairwise_measurements(config.run_dir) if run_bt else []
    if scores:
        print(f"\nLoaded {len(scores)} task scores")
    if measurements:
        print(f"Loaded {len(measurements)} pairwise comparisons")

    # Collect all task IDs from scores and/or measurements
    all_task_ids = set(scores.keys())
    for m in measurements:
        all_task_ids.add(m.task_a.id)
        all_task_ids.add(m.task_b.id)

    # Build task_id -> group mapping
    task_groups = build_task_groups(
        task_ids=all_task_ids,
        grouping=config.hoo_grouping,
        topics_json=config.topics_json,
    )
    # Keep only tasks that have group labels (and scores if running ridge)
    if run_ridge:
        scored_and_grouped = set(scores.keys()) & set(task_groups.keys())
        n_dropped = len(scores) - len(scored_and_grouped)
    else:
        scored_and_grouped = all_task_ids & set(task_groups.keys())
        n_dropped = len(all_task_ids) - len(scored_and_grouped)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} tasks without group metadata")

    all_groups = sorted(set(task_groups[tid] for tid in scored_and_grouped))
    if config.hoo_groups is not None:
        all_groups = [g for g in all_groups if g in config.hoo_groups]
    print(f"  Groups ({len(all_groups)}): {all_groups}")

    group_sizes = {}
    for g in all_groups:
        group_sizes[g] = sum(1 for tid in scored_and_grouped if task_groups[tid] == g)
    for g, n in sorted(group_sizes.items(), key=lambda x: -x[1]):
        print(f"    {g}: {n}")

    if config.hoo_hold_out_size >= len(all_groups):
        raise ValueError(
            f"hoo_hold_out_size ({config.hoo_hold_out_size}) must be < number of groups ({len(all_groups)})"
        )

    # Pre-load all activations (filtered to scored+grouped tasks)
    task_ids_arr, activations = load_activations(
        config.activations_path,
        task_id_filter=scored_and_grouped,
        layers=config.layers,
    )
    print(f"  {len(task_ids_arr)} tasks with activations")

    # Build full pairwise data if running BT
    bt_data = None
    if run_bt:
        bt_data = PairwiseActivationData.from_measurements(measurements, task_ids_arr, activations)
        print(f"  {len(bt_data.pairs)} unique pairs ({bt_data.n_measurements} measurements)")

    folds = list(combinations(all_groups, config.hoo_hold_out_size))
    print(f"\n{len(folds)} folds\n")

    # Method factory configs: (name, factory_fn, kwargs)
    methods_config = []
    if run_ridge:
        methods_config.append(("ridge", hoo_ridge.make_method, {
            "config": config,
            "task_ids": task_ids_arr,
            "activations": activations,
            "scores": scores,
            "task_groups": task_groups,
            "scored_and_grouped": scored_and_grouped,
        }))
    if run_bt and bt_data is not None:
        methods_config.append(("bradley_terry", hoo_bt.make_method, {
            "bt_data": bt_data,
            "task_ids": task_ids_arr,
            "task_groups": task_groups,
        }))

    best_hps: dict[str, float | None] = {name: None for name, _, _ in methods_config}
    all_fold_results = []

    for fold_idx, held_out in enumerate(folds):
        held_out_set = set(held_out)
        train_groups = [g for g in all_groups if g not in held_out_set]

        fold_metrics = {
            "fold_idx": fold_idx,
            "held_out_groups": list(held_out),
            "train_groups": train_groups,
            "layers": {},
        }

        for name, factory, kwargs in methods_config:
            method = factory(
                fold_idx=fold_idx,
                held_out_set=held_out_set,
                best_hp=best_hps[name],
                **kwargs,
            )
            if method is None:
                continue

            for layer in config.layers:
                weights, method.best_hp = method.train(layer, method.best_hp)
                probe_id = f"hoo_fold{fold_idx}_{method.name}_L{layer:02d}"
                save_probe(weights, config.output_dir, probe_id)
                metrics = method.evaluate(layer, weights)
                metrics.update(method=method.name, probe_id=probe_id, layer=layer)
                fold_metrics["layers"][f"{method.name}_L{layer}"] = metrics

            best_hps[name] = method.best_hp

        all_fold_results.append(fold_metrics)

    # Build summary
    summary = {
        "experiment_name": config.experiment_name,
        "created_at": datetime.now().isoformat(),
        "grouping": config.hoo_grouping,
        "hold_out_size": config.hoo_hold_out_size,
        "all_groups": all_groups,
        "group_sizes": group_sizes,
        "n_folds": len(all_fold_results),
        "layers": config.layers,
        "residualize_confounds": config.residualize_confounds,
        "folds": all_fold_results,
    }

    # Aggregate across folds per layer
    def _collect(key: str, field: str) -> list:
        return [
            f["layers"][key][field]
            for f in all_fold_results
            if key in f["layers"] and f["layers"][key].get(field) is not None
        ]

    layer_summary = {}
    for layer in config.layers:
        entry = {}
        if run_ridge:
            k = f"ridge_L{layer}"
            val_rs = _collect(k, "val_r")
            hoo_rs = _collect(k, "hoo_r")
            entry["ridge"] = {
                "mean_val_r": float(np.mean(val_rs)) if val_rs else None,
                "mean_hoo_r": float(np.mean(hoo_rs)) if hoo_rs else None,
                "std_hoo_r": float(np.std(hoo_rs)) if hoo_rs else None,
                "n_folds": len(hoo_rs),
            }
        if run_bt:
            k = f"bradley_terry_L{layer}"
            val_accs = _collect(k, "val_acc")
            hoo_accs = _collect(k, "hoo_acc")
            entry["bradley_terry"] = {
                "mean_val_acc": float(np.mean(val_accs)) if val_accs else None,
                "mean_hoo_acc": float(np.mean(hoo_accs)) if hoo_accs else None,
                "std_hoo_acc": float(np.std(hoo_accs)) if hoo_accs else None,
                "n_folds": len(hoo_accs),
            }
        layer_summary[layer] = entry
    summary["layer_summary"] = {str(k): v for k, v in layer_summary.items()}

    # Print summary
    print("\n" + "=" * 60)
    print("HOO Summary")
    print("=" * 60)
    for layer in config.layers:
        ls = layer_summary[layer]
        if "ridge" in ls and ls["ridge"]["mean_val_r"] is not None:
            r = ls["ridge"]
            gap = r["mean_val_r"] - r["mean_hoo_r"] if r["mean_hoo_r"] is not None else None
            gap_str = f", gap={gap:.4f}" if gap is not None else ""
            print(f"  Ridge L{layer}: val_r={r['mean_val_r']:.4f}, hoo_r={r['mean_hoo_r']:.4f}{gap_str} "
                  f"({r['n_folds']} folds)")
        if "bradley_terry" in ls and ls["bradley_terry"]["mean_val_acc"] is not None:
            b = ls["bradley_terry"]
            gap = b["mean_val_acc"] - b["mean_hoo_acc"] if b["mean_hoo_acc"] is not None else None
            gap_str = f", gap={gap:.4f}" if gap is not None else ""
            print(f"  BT    L{layer}: val_acc={b['mean_val_acc']:.4f}, hoo_acc={b['mean_hoo_acc']:.4f}{gap_str} "
                  f"({b['n_folds']} folds)")

    # Save
    summary_path = config.output_dir / "hoo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {summary_path}")

    plot_hoo_summary(summary, config.output_dir)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train probes from run directory results")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML path")
    args = parser.parse_args()

    config = RunDirProbeConfig.from_yaml(args.config)

    if config.hoo_grouping is not None:
        run_hoo(config)
    else:
        run_probes(config)


if __name__ == "__main__":
    main()
