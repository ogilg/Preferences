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

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.probes.core.activations import load_activations
from src.probes.core.evaluate import evaluate_probe_on_data
from src.probes.core.storage import save_probe, save_manifest
from src.probes.core.training import train_for_scores
from src.probes.bradley_terry.data import PairwiseActivationData
from src.probes.bradley_terry.training import train_for_comparisons
from src.probes.data_loading import load_thurstonian_scores, load_pairwise_measurements
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

    results, probes = train_for_comparisons(data=data, n_jobs=config.n_jobs)

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
            "cv_accuracy_mean": result.cv_accuracy_mean,
            "cv_accuracy_std": result.cv_accuracy_std,
            "best_l2_lambda": result.best_l2_lambda,
            "n_iterations": result.n_iterations,
            "n_pairs": n_pairs_used,
            "lambda_sweep": result.lambda_sweep,
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
            ridge_entries = _train_ridge_probes(config, task_ids, activations, scores)
            manifest["probes"].extend(ridge_entries)
            if ridge_entries:
                print(f"  Ridge cv_R²={ridge_entries[0]['cv_r2_mean']:.4f}")

        if run_bt:
            print(f"  Training BT probe...")
            bt_data = PairwiseActivationData.from_measurements(measurements, task_ids, activations)
            bt_entries, n_pairs = _train_bt_probes(config, bt_data)
            manifest["probes"].extend(bt_entries)
            if bt_entries:
                print(f"  BT cv_acc={bt_entries[0]['cv_accuracy_mean']:.4f} "
                      f"(best l2={bt_entries[0]['best_l2_lambda']:.4g}, {n_pairs} pairs)")

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
    """Held-one-out training and evaluation by group (topic or dataset).

    For each fold, trains on all groups except the held-out ones,
    then evaluates on the held-out groups. Only supports Ridge mode.
    """
    assert config.hoo_grouping is not None
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"HOO Probes: {config.experiment_name}")
    print(f"Grouping: {config.hoo_grouping}, hold_out_size: {config.hoo_hold_out_size}")
    print(f"Run dir: {config.run_dir}")
    print(f"Output: {config.output_dir}")

    # Load scores
    scores = load_thurstonian_scores(config.run_dir)
    print(f"\nLoaded {len(scores)} task scores")

    # Build task_id -> group mapping
    task_groups = build_task_groups(
        task_ids=set(scores.keys()),
        grouping=config.hoo_grouping,
        topics_json=config.topics_json,
    )
    # Keep only tasks that have both scores and group labels
    scored_and_grouped = set(scores.keys()) & set(task_groups.keys())
    n_dropped = len(scores) - len(scored_and_grouped)
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
    task_id_set = set(task_ids_arr)
    print(f"  {len(task_ids_arr)} tasks with activations")

    folds = list(combinations(all_groups, config.hoo_hold_out_size))
    print(f"\n{len(folds)} folds\n")

    all_fold_results = []

    for fold_idx, held_out in enumerate(folds):
        held_out_set = set(held_out)
        train_groups = [g for g in all_groups if g not in held_out_set]
        eval_label = ", ".join(held_out)

        # Split scores
        train_scores = {
            tid: scores[tid]
            for tid in scored_and_grouped
            if task_groups[tid] not in held_out_set and tid in task_id_set
        }
        eval_scores = {
            tid: scores[tid]
            for tid in scored_and_grouped
            if task_groups[tid] in held_out_set and tid in task_id_set
        }

        if len(train_scores) < config.cv_folds * 2 or len(eval_scores) < 10:
            print(f"Fold {fold_idx}: skip (train={len(train_scores)}, eval={len(eval_scores)})")
            continue

        print(f"Fold {fold_idx}: hold out [{eval_label}] "
              f"(train={len(train_scores)}, eval={len(eval_scores)})")

        # Optionally residualize train targets only — eval stays raw
        if config.residualize_confounds:
            assert config.topics_json is not None
            train_scores, res_stats = residualize_scores(
                train_scores, config.topics_json, confounds=config.residualize_confounds,
            )
            print(f"  Residualized train (R²={res_stats['metadata_r2']:.4f})")

        # Train probes on train split
        results, probes = train_for_scores(
            task_ids=task_ids_arr,
            activations=activations,
            scores=train_scores,
            cv_folds=config.cv_folds,
            alpha_sweep_size=config.alpha_sweep_size,
            standardize=config.standardize,
        )

        fold_metrics = {
            "fold_idx": fold_idx,
            "held_out_groups": list(held_out),
            "train_groups": train_groups,
            "n_train": len(train_scores),
            "n_eval": len(eval_scores),
            "layers": {},
        }

        for result in results:
            layer = result["layer"]
            probe_weights = probes[layer]

            # Save probe
            probe_id = f"hoo_fold{fold_idx}_L{layer:02d}"
            save_probe(probe_weights, config.output_dir, probe_id)

            # Evaluate on held-out tasks
            eval_task_ids_list = list(eval_scores.keys())
            eval_scores_arr = np.array([eval_scores[tid] for tid in eval_task_ids_list])

            eval_result = evaluate_probe_on_data(
                probe_weights=probe_weights,
                activations=activations[layer],
                scores=eval_scores_arr,
                task_ids_data=task_ids_arr,
                task_ids_scores=eval_task_ids_list,
            )

            fold_metrics["layers"][layer] = {
                "probe_id": probe_id,
                "train_cv_r2": result["cv_r2_mean"],
                "train_cv_r2_std": result["cv_r2_std"],
                "best_alpha": result["best_alpha"],
                "eval_r2": eval_result["r2"],
                "eval_r2_adjusted": eval_result["r2_adjusted"],
                "eval_pearson_r": eval_result["pearson_r"],
                "eval_n_samples": eval_result["n_samples"],
            }

            eval_r2 = eval_result["r2"]
            eval_r2_str = f"{eval_r2:.4f}" if eval_r2 is not None else "N/A"
            eval_pr = eval_result["pearson_r"]
            eval_pr_str = f"{eval_pr:.4f}" if eval_pr is not None else "N/A"
            print(f"  L{layer}: train_cv_R²={result['cv_r2_mean']:.4f}, "
                  f"eval_R²={eval_r2_str}, eval_r={eval_pr_str}")

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
    layer_summary = {}
    for layer in config.layers:
        eval_r2s = [
            f["layers"][layer]["eval_r2"]
            for f in all_fold_results
            if layer in f["layers"] and f["layers"][layer]["eval_r2"] is not None
        ]
        eval_r2_adj = [
            f["layers"][layer]["eval_r2_adjusted"]
            for f in all_fold_results
            if layer in f["layers"] and f["layers"][layer]["eval_r2_adjusted"] is not None
        ]
        eval_pearson = [
            f["layers"][layer]["eval_pearson_r"]
            for f in all_fold_results
            if layer in f["layers"] and f["layers"][layer]["eval_pearson_r"] is not None
        ]
        train_r2s = [
            f["layers"][layer]["train_cv_r2"]
            for f in all_fold_results
            if layer in f["layers"]
        ]
        layer_summary[layer] = {
            "mean_eval_r2": float(np.mean(eval_r2s)) if eval_r2s else None,
            "std_eval_r2": float(np.std(eval_r2s)) if eval_r2s else None,
            "mean_eval_r2_adjusted": float(np.mean(eval_r2_adj)) if eval_r2_adj else None,
            "mean_eval_pearson_r": float(np.mean(eval_pearson)) if eval_pearson else None,
            "std_eval_pearson_r": float(np.std(eval_pearson)) if eval_pearson else None,
            "mean_train_cv_r2": float(np.mean(train_r2s)) if train_r2s else None,
            "n_folds_evaluated": len(eval_r2s),
        }
    summary["layer_summary"] = {str(k): v for k, v in layer_summary.items()}

    # Print summary
    print("\n" + "=" * 60)
    print("HOO Summary")
    print("=" * 60)
    for layer in config.layers:
        ls = layer_summary[layer]
        mean_r2 = ls["mean_eval_r2"]
        std_r2 = ls["std_eval_r2"]
        if mean_r2 is not None:
            pr = ls["mean_eval_pearson_r"]
            pr_str = f", eval_r={pr:.4f}" if pr is not None else ""
            print(f"  L{layer}: eval_R²={mean_r2:.4f} ± {std_r2:.4f}{pr_str} "
                  f"(train_cv_R²={ls['mean_train_cv_r2']:.4f}, {ls['n_folds_evaluated']} folds)")
        else:
            print(f"  L{layer}: no valid folds")

    # Save
    summary_path = config.output_dir / "hoo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {summary_path}")

    _plot_hoo_summary(summary, config.output_dir)

    return summary


def _plot_hoo_summary(summary: dict, output_dir: Path) -> None:
    """Plot per-fold eval R² for each layer, grouped by held-out group."""
    folds = summary["folds"]
    layers = summary["layers"]
    if not folds:
        return

    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 5), squeeze=False)
    axes = axes[0]

    for ax, layer in zip(axes, layers):
        fold_labels = []
        r2_values = []
        r2_adj_values = []
        for f in folds:
            label = ", ".join(f["held_out_groups"])
            if len(label) > 20:
                label = label[:17] + "..."
            fold_labels.append(label)
            layer_data = f["layers"].get(str(layer)) or f["layers"].get(layer)
            if layer_data:
                r2_values.append(layer_data["eval_r2"])
                r2_adj_values.append(layer_data["eval_r2_adjusted"])
            else:
                r2_values.append(None)
                r2_adj_values.append(None)

        x = np.arange(len(fold_labels))
        valid_r2 = [v if v is not None else 0 for v in r2_values]
        valid_adj = [v if v is not None else 0 for v in r2_adj_values]

        bars = ax.bar(x - 0.15, valid_r2, 0.3, label="R²", color="#3498db", alpha=0.8)
        ax.bar(x + 0.15, valid_adj, 0.3, label="R² (mean-adj)", color="#2ecc71", alpha=0.8)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("R²")
        ax.set_title(f"Layer {layer}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(f"HOO Eval R² by Held-Out Group ({summary['grouping']})", fontweight="bold")
    plt.tight_layout()
    date_str = datetime.now().strftime("%m%d%y")
    plot_path = output_dir / f"plot_{date_str}_hoo_{summary['grouping']}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")


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
