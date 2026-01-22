"""Evaluate trained probes on evaluation datasets."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.probes.config import ProbeEvaluationConfig
from src.probes.storage import load_probe, load_manifest
from src.probes.activations import load_activations
from src.probes.evaluate import evaluate_probe_on_data
from src.measurement_storage.loading import load_run_utilities
from src.measurement_storage.base import find_project_root


def run_evaluation(config: ProbeEvaluationConfig) -> dict:
    """Run evaluation of probes on dataset specified in config.

    Args:
        config: ProbeEvaluationConfig specifying probes and eval data

    Returns:
        dict with results for each probe
    """
    # Load manifest and probe metadata
    manifest = load_manifest(config.manifest_dir)

    # Determine activations path (directory containing activations.npz)
    if config.activations_path is None:
        activations_path = find_project_root() / "probe_data" / "activations"
    else:
        # If path points to activations.npz file, use parent directory
        activations_path = config.activations_path
        if activations_path.name == "activations.npz":
            activations_path = activations_path.parent

    # Load activations
    print(f"Loading activations from {activations_path}")
    task_ids, activations_dict = load_activations(activations_path)

    # Find measurement run directory for template and seeds
    # Convention: template_model_response_format_seed{N}
    runs_by_layer: dict[int, Path] = {}
    for probe_meta in manifest["probes"]:
        layer = probe_meta["layer"]
        if layer in runs_by_layer:
            continue

        # Search for matching run in experiment_dir
        # Expected structure: {experiment_dir}/post_task_stated/{template}_{model}_{format}_seed{N}
        # For now, assume single run per template+seed combination
        template_prefix = config.template
        run_dir = None

        # Determine which subdirectory to search in based on template
        if config.template.startswith("pre_task"):
            search_dir = config.experiment_dir / "pre_task_stated"
        else:
            search_dir = config.experiment_dir / "post_task_stated"

        # Look for directory matching template and one of the seeds
        if search_dir.exists():
            for seed in config.seeds:
                # Try to find run directory matching template_*_seed{seed}
                for child in search_dir.iterdir():
                    if not child.is_dir():
                        continue
                    if template_prefix in child.name and f"seed{seed}" in child.name:
                        run_dir = child
                        break
                if run_dir:
                    break

        if run_dir is None:
            raise FileNotFoundError(
                f"Could not find measurement run for template={config.template}, "
                f"seeds={config.seeds} in {config.experiment_dir}"
            )

        runs_by_layer[layer] = run_dir

    # Evaluate each probe
    results = {
        "config": {
            "manifest_dir": str(config.manifest_dir),
            "probe_ids": config.probe_ids,
            "experiment_dir": str(config.experiment_dir),
            "template": config.template,
            "seeds": config.seeds,
            "dataset_filter": config.dataset_filter,
        },
        "created_at": datetime.now().isoformat(),
        "probes": [],
    }

    for probe_meta in manifest["probes"]:
        # Use all probes if probe_ids is empty, otherwise filter
        if config.probe_ids and probe_meta["id"] not in config.probe_ids:
            continue

        probe_id = probe_meta["id"]
        layer = probe_meta["layer"]
        print(f"\nEvaluating probe {probe_id} (layer {layer})")

        # Load probe weights
        probe_weights = load_probe(config.manifest_dir, probe_id)

        # Get activations for this layer
        X = activations_dict[layer]

        # Load scores from run directory
        run_dir = runs_by_layer[layer]
        try:
            scores, task_ids_scores = load_run_utilities(run_dir)
        except FileNotFoundError as e:
            print(f"  Warning: Could not load utilities from {run_dir}: {e}")
            continue

        # Filter by dataset if requested
        if config.dataset_filter is not None:
            from src.probes.activations import load_task_origins

            origins = load_task_origins(activations_path)
            origin_filter = config.dataset_filter.upper()
            if origin_filter not in origins:
                print(
                    f"  Warning: dataset filter '{config.dataset_filter}' not found. "
                    f"Available: {list(origins.keys())}"
                )
                valid_task_ids = set()
            else:
                valid_task_ids = origins[origin_filter]

            # Filter scores and activations
            mask = np.array([tid in valid_task_ids for tid in task_ids_scores])
            scores = scores[mask]
            task_ids_scores = [tid for tid, m in zip(task_ids_scores, mask) if m]

            if len(scores) == 0:
                print(f"  Warning: No samples after dataset filter '{config.dataset_filter}'")
                continue

        # Evaluate
        eval_result = evaluate_probe_on_data(
            probe_weights=probe_weights,
            activations=X,
            scores=scores,
            task_ids_data=task_ids,
            task_ids_scores=task_ids_scores,
        )

        probe_result = {
            "id": probe_id,
            "layer": layer,
            "trained_on_templates": probe_meta.get("templates", []),
            "trained_on_datasets": probe_meta.get("datasets"),
            "eval_metrics": eval_result,
        }
        results["probes"].append(probe_result)

        # Print results (handle None values)
        r2_str = f"{eval_result['r2']:.4f}" if eval_result['r2'] is not None else "N/A"
        pearson_str = f"{eval_result['pearson_r']:.4f}" if eval_result['pearson_r'] is not None else "N/A"
        print(
            f"  R²={r2_str}, n={eval_result['n_samples']}, "
            f"Pearson r={pearson_str}"
        )

    return results


def main():
    """CLI entry point."""
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate trained probes on evaluation dataset")
    parser.add_argument("--config", type=Path, required=True, help="Path to evaluation config YAML")
    parser.add_argument("--output", type=Path, help="Output file for results (optional)")

    args = parser.parse_args()

    # Load config
    config = ProbeEvaluationConfig.from_yaml(args.config)

    # Run evaluation
    results = run_evaluation(config)

    # Determine output path
    if args.output:
        output_path = args.output
    elif config.results_file:
        output_path = config.results_file
    else:
        output_path = Path("probe_eval_results.json")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
