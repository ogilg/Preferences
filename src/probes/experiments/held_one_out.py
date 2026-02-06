"""Held-one-out (leave-one-out) validation for probes.

Replaces run_held_one_out.py, run_hoo_evaluation.py, run_hoo_validation.py.
Usage:
    python -m src.probes.experiments.held_one_out train --config base.yaml
    python -m src.probes.experiments.held_one_out evaluate --manifest-dir ... --activations ...
    python -m src.probes.experiments.held_one_out validate --manifest-dir ... --activations ...
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

import yaml

from src.probes.config import ProbeConfig, ProbeType, DataSpec
from src.probes.core.activations import load_activations, load_task_origins
from src.probes.core.evaluate import evaluate_probe_on_data
from src.probes.core.storage import load_manifest, load_probe, save_manifest
from src.probes.runner import run_training
from src.measurement.storage.loading import load_run_utilities


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def _get_available_datasets(activations_path: Path) -> set[str]:
    """Load available datasets from activations metadata."""
    if activations_path.name == "activations.npz":
        activations_path = activations_path.parent

    completions_path = activations_path / "completions_with_activations.json"
    if not completions_path.exists():
        raise FileNotFoundError(f"Cannot find {completions_path}")

    with open(completions_path) as f:
        completions = json.load(f)

    origins = set(c.get("origin") for c in completions if c.get("origin"))
    return {o.lower() for o in origins}


def run_hoo_training(
    base_config_path: Path,
    datasets: set[str] | None = None,
    hold_out_size: int = 1,
    output_dir: Path | None = None,
) -> dict:
    """Train probes with held-one-out folds.

    Trains one probe set per fold, excluding hold_out_size datasets at a time.
    All probes are written to the same unified output directory.
    """
    with open(base_config_path) as f:
        base_data = yaml.safe_load(f)

    activations_path = Path(base_data["activations_path"])
    if activations_path.name == "activations.npz":
        activations_dir = activations_path.parent
    else:
        activations_dir = activations_path

    if datasets is None:
        datasets = _get_available_datasets(activations_dir)

    if hold_out_size >= len(datasets):
        raise ValueError(f"hold_out_size ({hold_out_size}) must be less than number of datasets ({len(datasets)})")

    # Determine unified output directory
    unified_output_dir = Path(base_data["output_dir"])

    print("=" * 80)
    print(f"HELD-OUT TRAINING (hold_out_size={hold_out_size})")
    print("=" * 80)
    print(f"Datasets: {sorted(datasets)}\n")

    results = {
        "created_at": datetime.now().isoformat(),
        "config": {"datasets": sorted(datasets), "hold_out_size": hold_out_size},
        "folds": [],
    }

    for eval_combo in combinations(sorted(datasets), hold_out_size):
        eval_datasets = set(eval_combo)
        train_datasets = sorted(datasets - eval_datasets)
        eval_str = ", ".join(sorted(eval_datasets)).upper()

        print("=" * 80)
        print(f"FOLD: Hold out {eval_str}")
        print("=" * 80)

        # Build a ProbeConfig for this fold
        training_data = DataSpec(
            experiment_dir=Path(base_data["training_data"]["experiment_dir"]),
            template_combinations=base_data["training_data"]["template_combinations"],
            seed_combinations=base_data["training_data"]["seed_combinations"],
            dataset_combinations=[train_datasets],
        )

        fold_config = ProbeConfig(
            experiment_name=f"{base_data['experiment_name']}_exclude_{'_'.join(sorted(eval_datasets))}",
            activations_path=activations_path,
            output_dir=unified_output_dir,
            layers=base_data["layers"],
            probe_type=ProbeType(base_data.get("probe_type", "ridge")),
            training_data=training_data,
            cv_folds=base_data.get("cv_folds", 5),
            alpha_sweep_size=base_data.get("alpha_sweep_size", 17),
        )

        print(f"Training on: {train_datasets}")
        print(f"Held out: {sorted(eval_datasets)}\n")

        try:
            training_manifest = run_training(fold_config)
            fold_result = {
                "eval_datasets": sorted(eval_datasets),
                "training_datasets": train_datasets,
                "n_probes": len(training_manifest["probes"]),
            }
            results["folds"].append(fold_result)
            print(f"Trained {len(training_manifest['probes'])} probes\n")
        except Exception as e:
            import traceback
            print(f"Error during training: {e}")
            print(traceback.format_exc())
            continue

    # Add HOO metadata to probes in unified manifest
    if results["folds"]:
        _consolidate_hoo_metadata(results["folds"], unified_output_dir, datasets, hold_out_size)

    # Save summary
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "hoo_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to {summary_path}")

    return results


def _consolidate_hoo_metadata(
    folds: list[dict],
    unified_manifest_dir: Path,
    datasets: set[str],
    hold_out_size: int,
) -> None:
    """Add HOO fold metadata to probes in the unified manifest."""
    unified_manifest = load_manifest(unified_manifest_dir)

    probe_to_fold = {}
    for fold in folds:
        fold_eval_datasets = fold["eval_datasets"]
        fold_train_datasets = sorted(fold.get("training_datasets", []))

        for probe in unified_manifest["probes"]:
            probe_train_datasets = sorted(probe.get("datasets", []))
            if probe_train_datasets == fold_train_datasets:
                if probe["id"] not in probe_to_fold:
                    probe_to_fold[probe["id"]] = {
                        "eval_datasets": fold_eval_datasets,
                        "train_datasets": fold_train_datasets,
                    }

    for probe in unified_manifest["probes"]:
        if probe["id"] in probe_to_fold:
            fold_info = probe_to_fold[probe["id"]]
            probe["hoo_fold_eval_datasets"] = fold_info["eval_datasets"]
            probe["hoo_fold_train_datasets"] = fold_info["train_datasets"]

    unified_manifest["hoo_consolidation"] = {
        "created_at": datetime.now().isoformat(),
        "num_folds": len(folds),
        "datasets": sorted(datasets),
        "hold_out_size": hold_out_size,
        "total_probes": len(unified_manifest["probes"]),
    }

    save_manifest(unified_manifest, unified_manifest_dir)
    print(f"Added HOO metadata to {len(unified_manifest['probes'])} probes")


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


def run_hoo_evaluation(
    manifest_dir: Path,
    activations_path: Path,
    template: str | None = None,
    seeds: list[int] | None = None,
    experiment_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict:
    """Evaluate HOO probes on their held-out datasets."""
    manifest = load_manifest(manifest_dir)

    # Infer defaults from manifest
    if template is None:
        if manifest.get("probes"):
            templates = manifest["probes"][0].get("templates") or manifest["probes"][0].get("template")
            template = templates[0] if isinstance(templates, list) else templates
        if not template:
            raise ValueError("Template not found in manifest and not provided")

    if seeds is None:
        seeds = manifest["probes"][0].get("seeds", [0]) if manifest.get("probes") else [0]

    if experiment_dir is None:
        experiment_dir_str = manifest.get("experiment_dir")
        if not experiment_dir_str:
            raise ValueError("experiment_dir not found in manifest and not provided")
        experiment_dir = Path(experiment_dir_str)

    if results_dir is None:
        results_dir = manifest_dir.parent
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HELD-ONE-OUT EVALUATION")
    print("=" * 80)
    print(f"Manifest: {manifest_dir}")
    print(f"Template: {template}")
    print(f"Seeds: {seeds}\n")

    # Load activations
    task_ids, activations_dict = load_activations(activations_path)

    # Group probes by their held-out datasets
    probes_by_holdout: dict[tuple[str, ...], list[str]] = {}
    for probe in manifest["probes"]:
        holdout_datasets = probe.get("hoo_fold_eval_datasets") or probe.get("hoo_fold_eval_dataset")
        if holdout_datasets:
            if isinstance(holdout_datasets, str):
                holdout_datasets = (holdout_datasets,)
            else:
                holdout_datasets = tuple(sorted(holdout_datasets))
            if holdout_datasets not in probes_by_holdout:
                probes_by_holdout[holdout_datasets] = []
            probes_by_holdout[holdout_datasets].append(probe["id"])

    if not probes_by_holdout:
        print("No HOO probes found in manifest (missing hoo_fold_eval_datasets metadata)")
        return {"folds": []}

    # Load origins for dataset filtering
    origins_cache = load_task_origins(activations_path.parent if activations_path.name == "activations.npz" else activations_path)

    # Find measurement run directory
    if template.startswith("pre_task"):
        search_dir = experiment_dir / "pre_task_stated"
    else:
        search_dir = experiment_dir / "post_task_stated"

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
        raise FileNotFoundError(
            f"Could not find measurement run for template={template}, seeds={seeds} in {experiment_dir}"
        )

    all_results = {"created_at": datetime.now().isoformat(), "folds": []}

    for eval_datasets_tuple in sorted(probes_by_holdout.keys()):
        probe_ids = probes_by_holdout[eval_datasets_tuple]
        eval_datasets_list = list(eval_datasets_tuple)
        eval_str = ", ".join(d.upper() for d in eval_datasets_list)

        print(f"\nFOLD: Evaluate on {eval_str} ({len(probe_ids)} probes)")

        fold_probe_results = []

        for eval_dataset in eval_datasets_list:
            # Load scores
            try:
                scores, task_ids_scores = load_run_utilities(run_dir)
            except FileNotFoundError as e:
                print(f"  Warning: {e}")
                continue

            # Filter by dataset
            valid_task_ids = origins_cache.get(eval_dataset.upper(), set())
            mask = np.array([tid in valid_task_ids for tid in task_ids_scores])
            ds_scores = scores[mask]
            ds_task_ids_scores = [tid for tid, m in zip(task_ids_scores, mask) if m]

            if len(ds_scores) == 0:
                continue

            for pid in probe_ids:
                probe_meta = next(p for p in manifest["probes"] if p["id"] == pid)
                layer = probe_meta["layer"]
                probe_weights = load_probe(manifest_dir, pid)

                eval_result = evaluate_probe_on_data(
                    probe_weights=probe_weights,
                    activations=activations_dict[layer],
                    scores=ds_scores,
                    task_ids_data=task_ids,
                    task_ids_scores=ds_task_ids_scores,
                )
                # Strip predictions for storage
                eval_result = {k: v for k, v in eval_result.items() if k != "predictions"}

                fold_probe_results.append({
                    "id": pid,
                    "layer": layer,
                    "eval_dataset": eval_dataset,
                    "eval_metrics": eval_result,
                })

        # Aggregate per-probe across datasets
        probes_aggregated = _aggregate_probe_results(probe_ids, fold_probe_results)

        fold_result = {
            "eval_datasets": eval_datasets_list,
            "probe_ids": probe_ids,
            "n_probes": len(probe_ids),
            "probes": probes_aggregated,
        }
        all_results["folds"].append(fold_result)

    # Save results
    eval_dir = manifest_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_dir / "hoo_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {summary_path}")

    _plot_hoo_r2(all_results, eval_dir)

    return all_results


def _aggregate_probe_results(probe_ids: list[str], raw_results: list[dict]) -> list[dict]:
    """Aggregate per-probe metrics across datasets."""
    aggregated = []
    for pid in probe_ids:
        per_dataset = [r for r in raw_results if r["id"] == pid]
        if not per_dataset:
            continue

        by_dataset = {r["eval_dataset"]: r["eval_metrics"] for r in per_dataset}

        all_r2 = [m["r2"] for m in by_dataset.values() if m.get("r2") is not None]
        all_r2_adj = [m["r2_adjusted"] for m in by_dataset.values() if m.get("r2_adjusted") is not None]
        all_pearson = [m["pearson_r"] for m in by_dataset.values() if m.get("pearson_r") is not None]

        aggregated.append({
            "id": pid,
            "eval_metrics": {
                "r2": sum(all_r2) / len(all_r2) if all_r2 else None,
                "r2_adjusted": sum(all_r2_adj) / len(all_r2_adj) if all_r2_adj else None,
                "pearson_r": sum(all_pearson) / len(all_pearson) if all_pearson else None,
                "by_dataset": by_dataset,
            },
        })

    return aggregated


def _plot_hoo_r2(results: dict, output_dir: Path) -> None:
    """Plot R² and mean-adjusted R² by hold-out dataset."""
    folds = results.get("folds", [])
    if not folds:
        return

    eval_datasets = []
    r2_values = []
    r2_adj_values = []

    for fold in folds:
        label = ", ".join(d.upper() for d in fold["eval_datasets"])
        eval_datasets.append(label)

        r2_scores = [p["eval_metrics"]["r2"] for p in fold["probes"] if p["eval_metrics"]["r2"] is not None]
        r2_adj_scores = [
            p["eval_metrics"].get("r2_adjusted")
            for p in fold["probes"]
            if p["eval_metrics"].get("r2_adjusted") is not None
        ]

        r2_values.append(np.median(r2_scores) if r2_scores else None)
        r2_adj_values.append(np.median(r2_adj_scores) if r2_adj_scores else None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("HOO Probe Performance by Dataset", fontsize=14, fontweight="bold")

    colors = ["#2ecc71" if r2 and r2 > 0 else "#e74c3c" for r2 in r2_values]
    bars1 = ax1.bar(eval_datasets, r2_values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax1.set_ylabel("Median R²")
    ax1.set_xlabel("Held-Out Dataset")
    ax1.set_title("Standard R²")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, val in zip(bars1, r2_values):
        if val is not None:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height, f"{val:.3f}",
                     ha="center", va="bottom" if height >= 0 else "top", fontsize=10, fontweight="bold")

    colors_adj = ["#3498db" if r2 and r2 > 0 else "#e67e22" for r2 in r2_adj_values]
    bars2 = ax2.bar(eval_datasets, r2_adj_values, color=colors_adj, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_ylabel("Median R² (Mean-Adjusted)")
    ax2.set_xlabel("Held-Out Dataset")
    ax2.set_title("Mean-Adjusted R²")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, val in zip(bars2, r2_adj_values):
        if val is not None:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, height, f"{val:.3f}",
                     ha="center", va="bottom" if height >= 0 else "top", fontsize=10, fontweight="bold")

    plt.tight_layout()
    date_str = datetime.now().strftime("%m%d%y")
    plot_path = output_dir / f"plot_{date_str}_hoo_r2.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")


# ---------------------------------------------------------------------------
# Validate (in-distribution)
# ---------------------------------------------------------------------------


def run_hoo_validation(
    manifest_dir: Path,
    activations_path: Path,
    template: str | None = None,
    seeds: list[int] | None = None,
    experiment_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict:
    """Evaluate HOO probes on their training data (in-distribution)."""
    manifest = load_manifest(manifest_dir)

    # Infer defaults from manifest
    if template is None:
        if manifest.get("probes"):
            templates = manifest["probes"][0].get("templates") or manifest["probes"][0].get("template")
            template = templates[0] if isinstance(templates, list) else templates
        if not template:
            raise ValueError("Template not found in manifest and not provided")

    if seeds is None:
        seeds = manifest["probes"][0].get("seeds", [0]) if manifest.get("probes") else [0]

    if experiment_dir is None:
        experiment_dir_str = manifest.get("experiment_dir")
        if not experiment_dir_str:
            raise ValueError("experiment_dir not found in manifest and not provided")
        experiment_dir = Path(experiment_dir_str)

    if results_dir is None:
        results_dir = manifest_dir.parent
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HELD-ONE-OUT VALIDATION (In-Distribution)")
    print("=" * 80)

    # Load activations
    if activations_path.name == "activations.npz":
        activations_path = activations_path.parent
    task_ids, activations_dict = load_activations(activations_path)

    # Find measurement run directory
    if template.startswith("pre_task"):
        search_dir = experiment_dir / "pre_task_stated"
    else:
        search_dir = experiment_dir / "post_task_stated"

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
        raise FileNotFoundError(
            f"Could not find measurement run for template={template}, seeds={seeds} in {experiment_dir}"
        )

    try:
        all_scores, all_task_ids_scores = load_run_utilities(run_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {"folds": [], "created_at": datetime.now().isoformat()}

    all_results = {"created_at": datetime.now().isoformat(), "folds": []}

    for probe in manifest["probes"]:
        probe_id = probe["id"]
        layer = probe["layer"]
        train_datasets = probe.get("hoo_fold_train_datasets", [])
        eval_datasets = probe.get("hoo_fold_eval_datasets") or probe.get("hoo_fold_eval_dataset")
        if isinstance(eval_datasets, str):
            eval_datasets = [eval_datasets]
        elif not eval_datasets:
            eval_datasets = ["unknown"]

        probe_weights = load_probe(manifest_dir, probe_id)
        coef = probe_weights[:-1]
        intercept = probe_weights[-1]

        X = activations_dict[layer]
        id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
        valid_indices = []
        valid_scores = []
        for task_id, score in zip(all_task_ids_scores, all_scores):
            if task_id in id_to_idx:
                valid_indices.append(id_to_idx[task_id])
                valid_scores.append(score)

        if len(valid_indices) < 10:
            continue

        indices = np.array(valid_indices)
        y = np.array(valid_scores)
        X_eval = X[indices]
        y_pred = X_eval @ coef + intercept

        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        pearson_r_val, _ = pearsonr(y, y_pred)

        y_pred_adjusted = y_pred - np.mean(y_pred) + np.mean(y)
        r2_adjusted = r2_score(y, y_pred_adjusted)

        print(f"Probe {probe_id} (L{layer}): R²={r2:.4f}, Adj R²={r2_adjusted:.4f}, Pearson r={pearson_r_val:.4f}")

        all_results["folds"].append({
            "probe_id": probe_id,
            "layer": layer,
            "train_datasets": train_datasets,
            "eval_datasets": eval_datasets,
            "metrics": {
                "r2": float(r2),
                "r2_adjusted": float(r2_adjusted),
                "mse": float(mse),
                "pearson_r": float(pearson_r_val),
                "n_samples": len(y),
            },
        })

    results_path = results_dir / "hoo_validation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Held-one-out probe experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_parser = subparsers.add_parser("train", help="Train probes with held-one-out folds")
    train_parser.add_argument("--config", type=Path, required=True, help="Base config YAML")
    train_parser.add_argument("--datasets", type=str, nargs="+", help="Datasets to validate on")
    train_parser.add_argument("--hold-out-size", type=int, default=1)
    train_parser.add_argument("--output-dir", type=Path)

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate probes on held-out datasets")
    eval_parser.add_argument("--manifest-dir", type=Path, required=True)
    eval_parser.add_argument("--activations", type=Path, required=True)
    eval_parser.add_argument("--template", type=str)
    eval_parser.add_argument("--seeds", type=int, nargs="+")
    eval_parser.add_argument("--experiment-dir", type=Path)
    eval_parser.add_argument("--output-dir", type=Path)

    # validate
    val_parser = subparsers.add_parser("validate", help="Evaluate probes on training data")
    val_parser.add_argument("--manifest-dir", type=Path, required=True)
    val_parser.add_argument("--activations", type=Path, required=True)
    val_parser.add_argument("--template", type=str)
    val_parser.add_argument("--seeds", type=int, nargs="+")
    val_parser.add_argument("--experiment-dir", type=Path)
    val_parser.add_argument("--output-dir", type=Path)

    args = parser.parse_args()

    if args.command == "train":
        datasets = {d.lower() for d in args.datasets} if args.datasets else None
        run_hoo_training(args.config, datasets=datasets, hold_out_size=args.hold_out_size, output_dir=args.output_dir)
    elif args.command == "evaluate":
        run_hoo_evaluation(
            manifest_dir=args.manifest_dir,
            activations_path=args.activations,
            template=args.template,
            seeds=args.seeds,
            experiment_dir=args.experiment_dir,
            results_dir=args.output_dir,
        )
    elif args.command == "validate":
        run_hoo_validation(
            manifest_dir=args.manifest_dir,
            activations_path=args.activations,
            template=args.template,
            seeds=args.seeds,
            experiment_dir=args.experiment_dir,
            results_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
