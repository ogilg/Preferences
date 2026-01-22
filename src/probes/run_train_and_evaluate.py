"""Orchestrate probe training and evaluation across multiple datasets."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.probes.config import ProbeTrainingConfig, ProbeEvaluationConfig
from src.probes.storage import load_manifest
from src.probes.run_probe_evaluation import run_evaluation


def run_train_and_evaluate(
    training_config: ProbeTrainingConfig,
    eval_configs: list[ProbeEvaluationConfig],
    output_dir: Path | None = None,
    skip_training: bool = False,
) -> dict:
    """Train probes and evaluate on multiple evaluation datasets.

    Args:
        training_config: ProbeTrainingConfig for training
        eval_configs: list of ProbeEvaluationConfig for evaluation runs
        output_dir: optional directory for saving results
        skip_training: if True, skip training and only run evaluation

    Returns:
        dict with training results and all evaluation results
    """
    print("=" * 80)
    print("PROBE TRAINING AND EVALUATION PIPELINE")
    print("=" * 80)

    # Step 1: Train probes (unless skipped)
    if not skip_training:
        print("\n[1/2] TRAINING PROBES")
        print("-" * 80)
        print(f"Training config: {training_config.experiment_name}")
        print(f"Manifest dir: {training_config.manifest_dir}")
        print()

        # Create a temporary config file for training
        import tempfile
        import yaml

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
            # Call train_probe_experiment.py
            result = subprocess.run(
                [sys.executable, "-m", "src.probes.train_probe_experiment", "--config", temp_config_path],
                check=True,
                cwd=Path.cwd(),
            )
        finally:
            Path(temp_config_path).unlink()

        print("\nTraining complete")
    else:
        print("\n[1/2] TRAINING PROBES - SKIPPED")
        print("-" * 80)

    # Load training results from manifest
    try:
        manifest = load_manifest(training_config.manifest_dir)
        training_results = manifest
        print(f"Loaded {len(manifest['probes'])} trained probes from manifest")
        for probe in manifest["probes"]:
            print(f"  - Probe {probe['id']}: R²={probe['cv_r2_mean']:.4f} ± {probe['cv_r2_std']:.4f}")
    except FileNotFoundError:
        print("Warning: No manifest found (training may have skipped)")
        training_results = {"probes": []}

    # Step 2: Evaluate probes
    print("\n[2/2] EVALUATING PROBES")
    print("-" * 80)
    print()

    evaluation_results = []
    for eval_idx, eval_config in enumerate(eval_configs, 1):
        print(f"\nEvaluation {eval_idx}/{len(eval_configs)}: {eval_config.template}")
        print(f"  Dataset filter: {eval_config.dataset_filter or 'none'}")
        print(f"  Seeds: {eval_config.seeds}")

        eval_result = run_evaluation(eval_config)
        evaluation_results.append(eval_result)

        print(f"  Evaluated {len(eval_result['probes'])} probes")

    # Step 3: Aggregate results
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    summary = {
        "created_at": datetime.now().isoformat(),
        "training": training_results,
        "evaluation": evaluation_results,
    }

    # Print summary table
    print("\nCross-dataset evaluation matrix:")
    print(f"{'Probe ID':<10} {'Layer':<8}", end="")
    for eval_idx in range(len(eval_configs)):
        print(f" {'Eval' + str(eval_idx + 1):<15}", end="")
    print()
    print("-" * (80))

    # Find all trained probes
    trained_probe_ids = {p["id"]: p for p in training_results["probes"]}

    for probe_id in sorted(trained_probe_ids.keys()):
        probe_meta = trained_probe_ids[probe_id]
        print(f"{probe_id:<10} {probe_meta['layer']:<8}", end="")

        for eval_result in evaluation_results:
            # Find this probe in this evaluation
            probe_eval = next((p for p in eval_result["probes"] if p["id"] == probe_id), None)
            if probe_eval and probe_eval["eval_metrics"]["r2"] is not None:
                r2 = probe_eval["eval_metrics"]["r2"]
                print(f" {r2:>6.4f}        ", end="")
            else:
                print(f" {'N/A':>15}", end="")
        print()

    # Save summary if output dir specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "train_eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull results saved to {summary_path}")

    return summary


def main():
    """CLI entry point."""
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Train probes and evaluate on multiple datasets"
    )
    parser.add_argument("--train-config", type=Path, required=True, help="Path to training config YAML")
    parser.add_argument(
        "--eval-configs",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to evaluation config YAMLs",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for saving results")
    parser.add_argument("--skip-training", action="store_true", help="Skip training phase")

    args = parser.parse_args()

    # Load configs
    training_config = ProbeTrainingConfig.from_yaml(args.train_config)
    eval_configs = [ProbeEvaluationConfig.from_yaml(p) for p in args.eval_configs]

    # Run pipeline
    summary = run_train_and_evaluate(
        training_config, eval_configs, args.output_dir, skip_training=args.skip_training
    )

    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
