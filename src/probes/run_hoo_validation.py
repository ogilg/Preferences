"""Evaluate held-one-out probes on their training data (in-distribution validation R²)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.probes.config import ProbeEvaluationConfig
from src.probes.storage import load_manifest, load_probe
from src.probes.activations import load_activations
from src.measurement_storage.loading import load_run_utilities


def run_hoo_validation(
    manifest_dir: Path,
    template: str | None = None,
    seeds: list[int] | None = None,
    experiment_dir: Path | None = None,
    activations_path: Path | None = None,
    results_dir: Path | None = None,
) -> dict:
    """Evaluate HOO probes on their training data (in-distribution).

    For each probe, evaluates it on the combined data it was trained on.

    Args:
        manifest_dir: unified manifest directory with all probes
        template: template name to evaluate on (read from manifest if None)
        seeds: seeds for evaluation (read from manifest if None)
        experiment_dir: experiment directory with measurements (read from manifest if None)
        activations_path: path to activations.npz (auto-detected if None)
        results_dir: directory to save results (default: manifest_dir parent)

    Returns:
        dict with validation results by fold
    """
    manifest = load_manifest(manifest_dir)

    # Read from manifest if not provided
    if template is None:
        if manifest.get("probes"):
            templates = manifest["probes"][0].get("templates") or manifest["probes"][0].get("template")
            if isinstance(templates, list):
                template = templates[0]
            else:
                template = templates
        if not template:
            raise ValueError("Template not found in manifest and not provided")

    if seeds is None:
        if manifest.get("probes"):
            seeds = manifest["probes"][0].get("seeds", [0])
        else:
            seeds = [0]

    if experiment_dir is None:
        experiment_dir = manifest.get("experiment_dir")
        if not experiment_dir:
            raise ValueError("experiment_dir not found in manifest and not provided")
        experiment_dir = Path(experiment_dir)
    else:
        experiment_dir = Path(experiment_dir)

    # Defaults for other paths
    if activations_path is None:
        raise ValueError("activations_path is required (e.g., activations/llama_3_1_8b/)")
    if results_dir is None:
        results_dir = manifest_dir.parent
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HELD-ONE-OUT VALIDATION (In-Distribution)")
    print("=" * 80)
    print(f"Manifest: {manifest_dir}")
    print(f"Template: {template}")
    print(f"Seeds: {seeds}\n")

    # Load activations once
    # Ensure activations_path points to directory, not the npz file
    if activations_path.name == "activations.npz":
        activations_path = activations_path.parent
    print(f"Loading activations from {activations_path}")
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
            f"Could not find measurement run for template={template}, "
            f"seeds={seeds} in {experiment_dir}"
        )

    print(f"Using measurement run: {run_dir}\n")

    # Load scores for all datasets
    try:
        all_scores, all_task_ids = load_run_utilities(run_dir)
    except FileNotFoundError as e:
        print(f"Error: Could not load utilities from {run_dir}: {e}")
        return {"folds": [], "created_at": datetime.now().isoformat()}

    all_results = {"created_at": datetime.now().isoformat(), "folds": []}

    # Evaluate each probe on its training data
    for probe in manifest["probes"]:
        probe_id = probe["id"]
        layer = probe["layer"]
        train_datasets = probe.get("hoo_fold_train_datasets", [])

        # Handle both old (singular) and new (plural) eval dataset formats
        eval_datasets = probe.get("hoo_fold_eval_datasets") or probe.get("hoo_fold_eval_dataset")
        if isinstance(eval_datasets, str):
            eval_datasets = [eval_datasets]
        elif not eval_datasets:
            eval_datasets = ["unknown"]
        eval_dataset_str = ", ".join(eval_datasets)

        print("=" * 80)
        print(f"PROBE {probe_id} (layer {layer})")
        print(f"Trained on: {', '.join(train_datasets)}")
        print(f"Held-out: {eval_dataset_str}")
        print("=" * 80)

        # Load probe weights
        try:
            probe_weights = load_probe(manifest_dir, probe_id)
        except FileNotFoundError as e:
            print(f"Error loading probe: {e}\n")
            continue

        coef = probe_weights[:-1]
        intercept = probe_weights[-1]

        # Get activations for this layer
        X = activations_dict[layer]

        # Match all scores to activations
        id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
        valid_indices = []
        valid_scores = []

        for task_id, score in zip(all_task_ids, all_scores):
            if task_id in id_to_idx:
                valid_indices.append(id_to_idx[task_id])
                valid_scores.append(score)

        if len(valid_indices) < 10:
            print(f"Warning: Not enough samples ({len(valid_indices)}) for evaluation\n")
            continue

        indices = np.array(valid_indices)
        y = np.array(valid_scores)
        X_eval = X[indices]

        # Predict
        y_pred = X_eval @ coef + intercept

        # Compute metrics
        from sklearn.metrics import r2_score, mean_squared_error
        from scipy.stats import pearsonr

        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        pearson_r, _ = pearsonr(y, y_pred)

        # Mean-adjusted R²
        y_mean = np.mean(y)
        y_pred_adjusted = y_pred - np.mean(y_pred) + y_mean
        r2_adjusted = r2_score(y, y_pred_adjusted)

        print(f"  R² = {r2:.4f}")
        print(f"  R² (adjusted) = {r2_adjusted:.4f}")
        print(f"  Pearson r = {pearson_r:.4f}")
        print(f"  MSE = {mse:.4f}")
        print(f"  n_samples = {len(y)}")

        fold_result = {
            "probe_id": probe_id,
            "layer": layer,
            "train_datasets": train_datasets,
            "eval_datasets": eval_datasets,
            "metrics": {
                "r2": float(r2),
                "r2_adjusted": float(r2_adjusted),
                "mse": float(mse),
                "pearson_r": float(pearson_r),
                "n_samples": len(y),
            },
        }
        all_results["folds"].append(fold_result)
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY - In-Distribution Validation R²")
    print("=" * 80)
    print(f"\n{'Probe':<8} {'Layer':<8} {'Datasets':<30} {'R²':<10} {'Adj. R²':<10} {'Pearson r':<12}")
    print("-" * 80)

    for fold in all_results["folds"]:
        probe_id = fold["probe_id"]
        layer = fold["layer"]
        datasets = ", ".join(fold["train_datasets"][:2]) + ("..." if len(fold["train_datasets"]) > 2 else "")
        r2 = fold["metrics"]["r2"]
        r2_adj = fold["metrics"]["r2_adjusted"]
        pearson = fold["metrics"]["pearson_r"]

        print(f"{probe_id:<8} {layer:<8} {datasets:<30} {r2:>9.4f} {r2_adj:>9.4f} {pearson:>11.4f}")

    # Save results
    results_path = results_dir / "hoo_validation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


def main():
    """CLI entry point for HOO validation."""
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate held-one-out probes on their training data (in-distribution validation)"
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Unified manifest directory (e.g., probe_data/manifests/probe_hoo)",
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Template name to evaluate on (read from manifest if omitted)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Seeds for evaluation (read from manifest if omitted)",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        help="Experiment directory with measurements (read from manifest if omitted)",
    )
    parser.add_argument(
        "--activations",
        type=Path,
        required=True,
        help="Path to activations directory (e.g., activations/llama_3_1_8b/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save results (default: manifest-dir parent)",
    )

    args = parser.parse_args()

    results = run_hoo_validation(
        manifest_dir=args.manifest_dir,
        template=args.template,
        seeds=args.seeds,
        experiment_dir=args.experiment_dir,
        activations_path=args.activations,
        results_dir=args.output_dir,
    )

    print("\n" + "=" * 80)
    print("In-distribution validation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
