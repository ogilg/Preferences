"""Evaluate held-one-out probes on their held-out datasets.

This script evaluates each probe on the dataset it was held out from.
Run this after run_held_one_out.py completes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from src.probes.config import ProbeEvaluationConfig
from src.probes.storage import load_manifest
from src.probes.run_probe_evaluation import run_evaluation


def run_hoo_evaluation(
    manifest_dir: Path,
    template: str | None = None,
    seeds: list[int] | None = None,
    experiment_dir: Path | None = None,
    activations_path: Path | None = None,
    results_dir: Path | None = None,
) -> dict:
    """Evaluate HOO probes on their held-out datasets.

    For each probe, evaluates it on the dataset it was held out from.
    Each probe is evaluated exactly once on its corresponding held-out dataset.

    If template, seeds, or experiment_dir are not provided, they are read from the manifest.

    Args:
        manifest_dir: unified manifest directory with all probes
        template: template name to evaluate on (read from manifest if None)
        seeds: seeds for evaluation (read from manifest if None)
        experiment_dir: experiment directory with measurements (read from manifest if None)
        activations_path: path to activations.npz (auto-detected if None)
        results_dir: directory to save results (default: manifest_dir parent)

    Returns:
        dict with evaluation results by fold
    """
    manifest = load_manifest(manifest_dir)

    # Read from manifest if not provided
    if template is None:
        # Get template from first probe
        if manifest.get("probes"):
            templates = manifest["probes"][0].get("templates") or manifest["probes"][0].get("template")
            if isinstance(templates, list):
                template = templates[0]
            else:
                template = templates
        if not template:
            raise ValueError("Template not found in manifest and not provided")

    if seeds is None:
        # Get seeds from first probe
        if manifest.get("probes"):
            seeds = manifest["probes"][0].get("seeds", [0])
        else:
            seeds = [0]

    if experiment_dir is None:
        # Get from manifest top-level
        experiment_dir = manifest.get("experiment_dir")
        if not experiment_dir:
            raise ValueError("experiment_dir not found in manifest and not provided")
        experiment_dir = Path(experiment_dir)
    else:
        experiment_dir = Path(experiment_dir)

    # Defaults for other paths
    if activations_path is None:
        activations_path = Path("probe_data/activations/activations.npz")
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

    # Group probes by their held-out datasets
    # Each probe maps to a tuple of eval datasets (could be 1 or more)
    probes_by_holdout: dict[tuple[str, ...], list[str]] = {}
    for probe in manifest["probes"]:
        # Support both old singular and new plural format
        holdout_datasets = probe.get("hoo_fold_eval_datasets") or probe.get("hoo_fold_eval_dataset")
        if holdout_datasets:
            # Normalize to tuple
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

    all_results = {"created_at": datetime.now().isoformat(), "folds": []}

    # Evaluate each probe on its held-out dataset(s)
    for eval_datasets_tuple in sorted(probes_by_holdout.keys()):
        probe_ids = probes_by_holdout[eval_datasets_tuple]
        eval_datasets_list = list(eval_datasets_tuple)
        eval_str = ", ".join(d.upper() for d in eval_datasets_list)

        print("=" * 80)
        print(f"FOLD: Evaluate on {eval_str}")
        print("=" * 80)
        print(f"Evaluating {len(probe_ids)} probes on {len(eval_datasets_list)} dataset(s)\n")

        # Evaluate on each held-out dataset separately, then aggregate
        fold_probe_results: dict[str, dict] = {pid: {"eval_metrics_by_dataset": {}} for pid in probe_ids}

        for eval_dataset in eval_datasets_list:
            print(f"  Evaluating on {eval_dataset}...")
            eval_config = ProbeEvaluationConfig(
                manifest_dir=manifest_dir,
                probe_ids=probe_ids,
                experiment_dir=experiment_dir,
                template=template,
                seeds=seeds,
                dataset_filter=eval_dataset,
                activations_path=activations_path,
                results_file=results_dir / f"probe_hoo_eval_{eval_dataset}.json",
            )

            try:
                eval_result = run_evaluation(eval_config)
                for probe in eval_result["probes"]:
                    pid = probe["id"]
                    metrics = probe.get("eval_metrics", {})
                    # Remove predictions
                    metrics = {k: v for k, v in metrics.items() if k != "predictions"}
                    fold_probe_results[pid]["eval_metrics_by_dataset"][eval_dataset] = metrics
            except Exception as e:
                import traceback
                print(f"    Error evaluating on {eval_dataset}: {e}")
                print(traceback.format_exc())
                continue

        # Aggregate metrics across datasets for each probe
        clean_probes = []
        for pid in probe_ids:
            by_dataset = fold_probe_results[pid]["eval_metrics_by_dataset"]
            if not by_dataset:
                continue

            # Average metrics across held-out datasets
            all_r2 = [m["r2"] for m in by_dataset.values() if m.get("r2") is not None]
            all_r2_adj = [m["r2_adjusted"] for m in by_dataset.values() if m.get("r2_adjusted") is not None]
            all_pearson = [m["pearson_r"] for m in by_dataset.values() if m.get("pearson_r") is not None]

            aggregated_metrics = {
                "r2": sum(all_r2) / len(all_r2) if all_r2 else None,
                "r2_adjusted": sum(all_r2_adj) / len(all_r2_adj) if all_r2_adj else None,
                "pearson_r": sum(all_pearson) / len(all_pearson) if all_pearson else None,
                "by_dataset": by_dataset,
            }

            clean_probes.append({
                "id": pid,
                "eval_metrics": aggregated_metrics,
            })

        fold_result = {
            "eval_datasets": eval_datasets_list,
            "probe_ids": probe_ids,
            "n_probes": len(probe_ids),
            "probes": clean_probes,
        }
        all_results["folds"].append(fold_result)
        print(f"Evaluated {len(probe_ids)} probes\n")

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if not all_results["folds"]:
        print("No evaluation results")
        return all_results

    print(f"\n{'Hold-out Dataset(s)':<30} {'# Probes':<12} {'Median R²':<15} {'Adj. R²':<15} {'Pearson r':<15}")
    print("-" * 90)

    for fold in all_results["folds"]:
        eval_datasets = fold["eval_datasets"]
        eval_str = ", ".join(eval_datasets)
        probes = fold["probes"]

        r2_scores = [p["eval_metrics"]["r2"] for p in probes if p["eval_metrics"]["r2"] is not None]
        r2_adj_scores = [
            p["eval_metrics"]["r2_adjusted"] for p in probes if p["eval_metrics"].get("r2_adjusted") is not None
        ]
        pearson_scores = [
            p["eval_metrics"]["pearson_r"] for p in probes if p["eval_metrics"]["pearson_r"] is not None
        ]

        if r2_scores:
            median_r2 = sorted(r2_scores)[len(r2_scores) // 2]
            median_r2_adj = sorted(r2_adj_scores)[len(r2_adj_scores) // 2] if r2_adj_scores else None
            mean_pearson = sum(pearson_scores) / len(pearson_scores)

            r2_adj_str = f"{median_r2_adj:.4f}" if median_r2_adj is not None else "N/A"
            print(f"{eval_str:<30} {len(probes):<12} {median_r2:<15.4f} {r2_adj_str:<15} {mean_pearson:<15.4f}")
        else:
            print(f"{eval_str:<30} {len(probes):<12} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

    # Save results to manifest_dir (not results_dir)
    eval_dir = manifest_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    summary_path = eval_dir / "hoo_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {summary_path}")

    # Generate R² figure
    plot_hoo_r2(all_results, eval_dir)

    return all_results


def plot_hoo_r2(results: dict, output_dir: Path) -> None:
    """Plot R² and mean-adjusted R² by hold-out dataset."""
    folds = results.get("folds", [])
    if not folds:
        return

    eval_datasets = []
    r2_values = []
    r2_adj_values = []
    pearson_values = []

    for fold in folds:
        # Handle both old and new format
        if "eval_datasets" in fold:
            label = ", ".join(d.upper() for d in fold["eval_datasets"])
        else:
            label = fold["eval_dataset"].upper()
        eval_datasets.append(label)

        r2_scores = [
            p["eval_metrics"]["r2"]
            for p in fold["probes"]
            if p["eval_metrics"]["r2"] is not None
        ]
        r2_adj_scores = [
            p["eval_metrics"].get("r2_adjusted")
            for p in fold["probes"]
            if p["eval_metrics"].get("r2_adjusted") is not None
        ]
        pearson_scores = [
            p["eval_metrics"]["pearson_r"]
            for p in fold["probes"]
            if p["eval_metrics"]["pearson_r"] is not None
        ]

        r2_values.append(np.median(r2_scores) if r2_scores else None)
        r2_adj_values.append(np.median(r2_adj_scores) if r2_adj_scores else None)
        pearson_values.append(np.median(pearson_scores) if pearson_scores else None)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("HOO Probe Performance by Dataset", fontsize=14, fontweight="bold")

    # Plot 1: Standard R²
    colors = ["#2ecc71" if r2 and r2 > 0 else "#e74c3c" for r2 in r2_values]
    bars1 = ax1.bar(eval_datasets, r2_values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax1.set_ylabel("Median R²", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Held-Out Dataset", fontsize=11, fontweight="bold")
    ax1.set_title("Standard R²", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, val in zip(bars1, r2_values):
        if val is not None:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

    # Plot 2: Mean-adjusted R²
    colors_adj = ["#3498db" if r2 and r2 > 0 else "#e67e22" for r2 in r2_adj_values]
    bars2 = ax2.bar(eval_datasets, r2_adj_values, color=colors_adj, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_ylabel("Median R² (Mean-Adjusted)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Held-Out Dataset", fontsize=11, fontweight="bold")
    ax2.set_title("Mean-Adjusted R² (accounts for dataset mean differences)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, val in zip(bars2, r2_adj_values):
        if val is not None:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    date_str = datetime.now().strftime("%m%d%y")
    plot_path = output_dir / f"plot_{date_str}_hoo_r2.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")


def main():
    """CLI entry point for HOO evaluation."""
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate held-one-out probes on their held-out datasets"
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
        help="Path to activations.npz (default: probe_data/activations/activations.npz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save results (default: manifest-dir parent)",
    )

    args = parser.parse_args()

    results = run_hoo_evaluation(
        manifest_dir=args.manifest_dir,
        template=args.template,
        seeds=args.seeds,
        experiment_dir=args.experiment_dir,
        activations_path=args.activations,
        results_dir=args.output_dir,
    )

    print("\n" + "=" * 80)
    print("Held-one-out evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
