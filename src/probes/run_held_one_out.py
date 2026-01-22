"""Run held-one-out (leave-one-out) validation for probes."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.probes.config import ProbeTrainingConfig, ProbeEvaluationConfig
from src.probes.storage import load_manifest
from src.probes.run_probe_evaluation import run_evaluation


def get_available_datasets(activations_path: Path) -> set[str]:
    """Load available datasets from activations metadata."""
    # Ensure we have a directory
    if activations_path.name == "activations.npz":
        activations_path = activations_path.parent

    completions_path = activations_path / "completions_with_activations.json"
    if not completions_path.exists():
        raise FileNotFoundError(f"Cannot find {completions_path}")

    with open(completions_path) as f:
        completions = json.load(f)

    origins = set(c.get("origin") for c in completions if c.get("origin"))
    return {o.lower() for o in origins}  # Normalize to lowercase


def create_training_config(
    base_config: dict,
    exclude_dataset: str | None,
    datasets: set[str],
) -> ProbeTrainingConfig:
    """Create training config excluding one dataset."""
    train_datasets = sorted(datasets - {exclude_dataset} if exclude_dataset else datasets)

    config_dict = dict(base_config)
    config_dict["dataset_combinations"] = [train_datasets]

    if exclude_dataset:
        config_dict["experiment_name"] = f"{base_config['experiment_name']}_exclude_{exclude_dataset}"
        config_dict["experiment_dir"] = Path(str(base_config["experiment_dir"]).replace(
            "DATASET", exclude_dataset
        ))
        config_dict["manifest_dir"] = Path(str(base_config["manifest_dir"]).replace(
            "DATASET", exclude_dataset
        ))

    config_dict["experiment_dir"] = Path(config_dict["experiment_dir"])
    config_dict["activations_path"] = Path(config_dict["activations_path"])
    config_dict["manifest_dir"] = Path(config_dict["manifest_dir"])

    return ProbeTrainingConfig(**config_dict)


def create_eval_config(
    training_config: ProbeTrainingConfig,
    eval_dataset: str,
    template: str,
    seeds: list[int],
    results_dir: Path,
) -> ProbeEvaluationConfig:
    """Create evaluation config for held-out dataset."""
    return ProbeEvaluationConfig(
        manifest_dir=training_config.manifest_dir,
        probe_ids=[],  # Empty = all probes
        experiment_dir=training_config.experiment_dir,
        template=template,
        seeds=seeds,
        dataset_filter=eval_dataset,
        activations_path=training_config.activations_path.parent / "activations.npz",
        results_file=results_dir / f"probe_hoo_eval_{eval_dataset}.json",
    )


def run_training(training_config: ProbeTrainingConfig) -> dict:
    """Run probe training via subprocess."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_dict = {
            "experiment_name": training_config.experiment_name,
            "experiment_dir": str(training_config.experiment_dir),
            "activations_path": str(training_config.activations_path),
            "template_combinations": training_config.template_combinations,
            "dataset_combinations": training_config.dataset_combinations,
            "seed_combinations": training_config.seed_combinations,
            "layers": training_config.layers,
            "cv_folds": training_config.cv_folds,
            "alpha_sweep_size": training_config.alpha_sweep_size,
            "manifest_dir": str(training_config.manifest_dir),
        }
        yaml.dump(config_dict, f)
        temp_config_path = f.name

    try:
        subprocess.run(
            [sys.executable, "-m", "src.probes.train_probe_experiment", "--config", temp_config_path],
            check=True,
            cwd=Path.cwd(),
        )
    finally:
        Path(temp_config_path).unlink()

    # Load and return manifest
    return load_manifest(training_config.manifest_dir)


def run_held_one_out(
    base_training_config: dict,
    template: str,
    seeds: list[int],
    datasets: set[str] | None = None,
    output_dir: Path | None = None,
    skip_training: bool = False,
) -> dict:
    """Run held-one-out validation across all datasets.

    Args:
        base_training_config: dict with base config (experiment_name, experiment_dir, etc.)
            Use "DATASET" placeholder for paths to be substituted
        template: template name to evaluate on
        seeds: seeds for evaluation
        datasets: set of datasets to validate on, or None to auto-detect
        output_dir: directory to save results
        skip_training: if True, skip training and only run evaluation

    Returns:
        dict with all results
    """
    # Determine datasets
    if datasets is None:
        activations_path = Path(base_training_config["activations_path"])
        # If path points to activations.npz file, use parent directory
        if activations_path.name == "activations.npz":
            activations_path = activations_path.parent
        datasets = get_available_datasets(activations_path)

    print("=" * 80)
    print("HELD-ONE-OUT VALIDATION")
    print("=" * 80)
    print(f"Datasets: {sorted(datasets)}")
    print(f"Template: {template}")
    print(f"Seeds: {seeds}\n")

    results = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "template": template,
            "seeds": seeds,
            "datasets": sorted(datasets),
        },
        "folds": [],
    }

    for eval_dataset in sorted(datasets):
        print("=" * 80)
        print(f"FOLD: Hold out {eval_dataset.upper()}")
        print("=" * 80)

        # Create training and eval configs
        training_config = create_training_config(base_training_config, eval_dataset, datasets)
        eval_config = create_eval_config(
            training_config, eval_dataset, template, seeds, output_dir or Path("results")
        )

        print(f"\nTraining on: {sorted(datasets - {eval_dataset})}")
        print(f"Evaluating on: {eval_dataset}\n")

        # Train
        if not skip_training:
            print("[1/2] Training...")
            try:
                training_manifest = run_training(training_config)
                print(f"Trained {len(training_manifest['probes'])} probes\n")
            except Exception as e:
                print(f"Error during training: {e}")
                continue
        else:
            print("[1/2] Training - SKIPPED")
            try:
                training_manifest = load_manifest(training_config.manifest_dir)
                print(f"Loaded {len(training_manifest['probes'])} probes from manifest\n")
            except FileNotFoundError:
                print(f"No manifest found at {training_config.manifest_dir}")
                continue

        # Evaluate
        print("[2/2] Evaluating...")
        try:
            eval_result = run_evaluation(eval_config)
            fold_result = {
                "eval_dataset": eval_dataset,
                "training_datasets": sorted(datasets - {eval_dataset}),
                "manifest_dir": str(training_config.manifest_dir),
                "results_file": str(eval_config.results_file),
                "n_probes": len(eval_result["probes"]),
                "probes": eval_result["probes"],
            }
            results["folds"].append(fold_result)
            print(f"Evaluated {len(eval_result['probes'])} probes\n")
        except Exception as e:
            import traceback
            print(f"Error during evaluation: {e}")
            print(traceback.format_exc())
            continue

    # Aggregate results
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if not results["folds"]:
        print("No results to summarize")
        return results

    # Print table
    print(f"\n{'Eval Dataset':<15} {'# Probes':<12} {'Median R²':<15} {'Mean Pearson r':<15}")
    print("-" * 60)

    for fold in results["folds"]:
        eval_dataset = fold["eval_dataset"]
        probes = fold["probes"]

        r2_scores = [p["eval_metrics"]["r2"] for p in probes if p["eval_metrics"]["r2"] is not None]
        pearson_scores = [
            p["eval_metrics"]["pearson_r"] for p in probes if p["eval_metrics"]["pearson_r"] is not None
        ]

        if r2_scores:
            median_r2 = sorted(r2_scores)[len(r2_scores) // 2]
            mean_pearson = sum(pearson_scores) / len(pearson_scores)
            print(f"{eval_dataset:<15} {len(probes):<12} {median_r2:<15.4f} {mean_pearson:<15.4f}")
        else:
            print(f"{eval_dataset:<15} {len(probes):<12} {'N/A':<15} {'N/A':<15}")

    # Save summary
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "hoo_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to {summary_path}")

    return results


def main():
    """CLI entry point."""
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run held-one-out validation for probes across datasets"
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="Base config YAML with DATASET placeholder (e.g., configs/probe_training/hoo_base.yaml)",
    )
    parser.add_argument("--template", type=str, required=True, help="Template name to evaluate on")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seeds for evaluation")
    parser.add_argument("--datasets", type=str, nargs="+", help="Datasets to validate on (auto-detect if omitted)")
    parser.add_argument("--output-dir", type=Path, help="Directory for saving results")
    parser.add_argument("--skip-training", action="store_true", help="Skip training phase")

    args = parser.parse_args()

    # Load base config
    with open(args.base_config) as f:
        base_config = yaml.safe_load(f)

    # Normalize datasets
    datasets = {d.lower() for d in args.datasets} if args.datasets else None

    # Run held-one-out
    results = run_held_one_out(
        base_config,
        template=args.template,
        seeds=args.seeds,
        datasets=datasets,
        output_dir=args.output_dir,
        skip_training=args.skip_training,
    )

    print("\n" + "=" * 80)
    print("Held-one-out validation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
