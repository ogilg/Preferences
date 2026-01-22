"""Train probes on noise baselines to benchmark real probe performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.measurement_storage.loading import load_raw_scores
from src.probes.activations import load_activations
from src.probes.linear_probe import train_and_evaluate

ALPHA_SWEEP_SIZE = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run noise baselines for probe benchmarking")
    parser.add_argument("experiment_dir", type=Path, help="Experiment directory (e.g. results/experiments/probe_4_all_datasets)")
    parser.add_argument("--activations-dir", type=Path, default=Path("probe_data/activations"), help="Directory containing activations.npz")
    parser.add_argument("--template", type=str, default="post_task_qualitative_001", help="Template name to load scores from")
    parser.add_argument("--seed", type=int, default=0, help="Seed filter")
    parser.add_argument("--layer", type=int, default=16, help="Layer to use for baselines")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of random seeds for baselines")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    return parser.parse_args()


def load_scores_from_experiment(
    experiment_dir: Path,
    template: str,
    seed: int,
) -> dict[str, float]:
    """Load scores from experiment directory, averaging per task_id."""
    task_type = "pre_task" if template.startswith("pre_task") else "post_task"
    measurement_dir = experiment_dir / f"{task_type}_stated"

    raw_measurements = load_raw_scores(
        measurement_dir,
        [template],
        [seed],
    )

    scores_by_task: dict[str, list[float]] = {}
    for task_id, score in raw_measurements:
        if task_id not in scores_by_task:
            scores_by_task[task_id] = []
        scores_by_task[task_id].append(score)

    return {tid: float(np.mean(scores)) for tid, scores in scores_by_task.items()}


def run_shuffled_labels_baseline(
    X: np.ndarray,
    y: np.ndarray,
    n_seeds: int,
    cv_folds: int,
) -> list[dict]:
    results = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        y_shuffled = rng.permutation(y)
        _, result, _ = train_and_evaluate(X, y_shuffled, cv_folds, ALPHA_SWEEP_SIZE)
        results.append({
            "seed": seed,
            "cv_r2_mean": result["cv_r2_mean"],
            "cv_r2_std": result["cv_r2_std"],
            "cv_mse_mean": result["cv_mse_mean"],
            "cv_mse_std": result["cv_mse_std"],
            "best_alpha": result["best_alpha"],
        })
    return results


def run_random_activations_baseline(
    X: np.ndarray,
    y: np.ndarray,
    n_seeds: int,
    cv_folds: int,
) -> list[dict]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    results = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        X_noise = rng.normal(loc=mean, scale=std, size=X.shape)
        _, result, _ = train_and_evaluate(X_noise, y, cv_folds, ALPHA_SWEEP_SIZE)
        results.append({
            "seed": seed,
            "cv_r2_mean": result["cv_r2_mean"],
            "cv_r2_std": result["cv_r2_std"],
            "cv_mse_mean": result["cv_mse_mean"],
            "cv_mse_std": result["cv_mse_std"],
            "best_alpha": result["best_alpha"],
        })
    return results


def plot_comparison(
    real_result: dict,
    shuffled_results: list[dict],
    random_results: list[dict],
    label_variance: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    labels = ["Real Probe", "Shuffled Labels", "Random Activations"]
    colors = ["steelblue", "darkorange", "forestgreen"]

    # Aggregate baseline results
    shuffled_r2 = np.mean([r["cv_r2_mean"] for r in shuffled_results])
    shuffled_r2_std = np.std([r["cv_r2_mean"] for r in shuffled_results])
    random_r2 = np.mean([r["cv_r2_mean"] for r in random_results])
    random_r2_std = np.std([r["cv_r2_mean"] for r in random_results])

    shuffled_mse = np.mean([r["cv_mse_mean"] for r in shuffled_results])
    shuffled_mse_std = np.std([r["cv_mse_mean"] for r in shuffled_results])
    random_mse = np.mean([r["cv_mse_mean"] for r in random_results])
    random_mse_std = np.std([r["cv_mse_mean"] for r in random_results])

    # R² plot
    ax = axes[0]
    r2_vals = [real_result["cv_r2_mean"], shuffled_r2, random_r2]
    r2_stds = [real_result["cv_r2_std"], shuffled_r2_std, random_r2_std]
    x = np.arange(len(labels))
    ax.bar(x, r2_vals, yerr=r2_stds, color=colors, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("CV R²")
    ax.set_title("R² (higher = better)")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # MSE plot
    ax = axes[1]
    mse_vals = [real_result["cv_mse_mean"], shuffled_mse, random_mse]
    mse_stds = [real_result["cv_mse_std"], shuffled_mse_std, random_mse_std]
    ax.bar(x, mse_vals, yerr=mse_stds, color=colors, capsize=5)
    ax.axhline(y=label_variance, color="red", linestyle="--", linewidth=1.5, label=f"Var(y) = {label_variance:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("CV MSE")
    ax.set_title("MSE (lower = better)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved comparison plot to {output_path}")


def main() -> None:
    args = parse_args()

    print(f"Loading scores from {args.experiment_dir}...")
    print(f"  Template: {args.template}, seed: {args.seed}")
    scores_map = load_scores_from_experiment(
        args.experiment_dir,
        args.template,
        args.seed,
    )
    print(f"  Found {len(scores_map)} task scores")

    if not scores_map:
        print("No scores found. Check experiment directory and filters.")
        return

    print(f"\nLoading activations from {args.activations_dir}...")
    task_ids, activations = load_activations(
        args.activations_dir,
        task_id_filter=set(scores_map.keys()),
        layers=[args.layer],
    )
    print(f"  Loaded {len(task_ids)} activations for layer {args.layer}")

    # Build aligned arrays
    y = np.array([scores_map[tid] for tid in task_ids])
    X = activations[args.layer]
    print(f"  Final dataset: {len(y)} samples, {X.shape[1]} dims")

    label_variance = float(np.var(y))
    print(f"  Label variance: {label_variance:.4f}")

    # Train real probe
    print(f"\nTraining real probe...")
    _, real_result, _ = train_and_evaluate(X, y, args.cv_folds, ALPHA_SWEEP_SIZE)
    print(f"  CV R² = {real_result['cv_r2_mean']:.4f} ± {real_result['cv_r2_std']:.4f}")
    print(f"  CV MSE = {real_result['cv_mse_mean']:.4f} ± {real_result['cv_mse_std']:.4f}")

    # Run baselines
    print(f"\nRunning shuffled labels baseline ({args.n_seeds} seeds)...")
    shuffled = run_shuffled_labels_baseline(X, y, args.n_seeds, args.cv_folds)
    shuffled_r2s = [r["cv_r2_mean"] for r in shuffled]
    print(f"  Mean CV R²: {np.mean(shuffled_r2s):.4f} ± {np.std(shuffled_r2s):.4f}")

    print(f"\nRunning random activations baseline ({args.n_seeds} seeds)...")
    random_acts = run_random_activations_baseline(X, y, args.n_seeds, args.cv_folds)
    random_r2s = [r["cv_r2_mean"] for r in random_acts]
    print(f"  Mean CV R²: {np.mean(random_r2s):.4f} ± {np.std(random_r2s):.4f}")

    # Save results
    results = {
        "experiment_dir": str(args.experiment_dir),
        "template": args.template,
        "seed": args.seed,
        "layer": args.layer,
        "n_samples": len(y),
        "label_variance": label_variance,
        "real_probe": real_result,
        "shuffled_labels": {
            "cv_r2_mean": float(np.mean(shuffled_r2s)),
            "cv_r2_std": float(np.std(shuffled_r2s)),
            "per_seed": shuffled,
        },
        "random_activations": {
            "cv_r2_mean": float(np.mean(random_r2s)),
            "cv_r2_std": float(np.std(random_r2s)),
            "per_seed": random_acts,
        },
    }

    output_dir = args.experiment_dir / "noise_baseline"
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / f"results_layer{args.layer}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    plot_path = output_dir / f"comparison_layer{args.layer}.png"
    plot_comparison(real_result, shuffled, random_acts, label_variance, plot_path)


if __name__ == "__main__":
    main()
