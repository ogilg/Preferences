"""Train probes on noise baselines to benchmark real probe performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.probes.linear_probe import DEFAULT_ALPHAS, train_and_evaluate

BASELINE_ALPHAS = np.array([10.0, 100.0, 1000.0, 10000.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run noise baselines for probe benchmarking")
    parser.add_argument("data_dir", type=Path, help="Directory containing probe data")
    parser.add_argument("scores_file", type=Path, help="JSON file mapping task_id to score")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    return parser.parse_args()


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
        _, result, _ = train_and_evaluate(X, y_shuffled, cv_folds=cv_folds, alphas=BASELINE_ALPHAS)
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
        _, result, _ = train_and_evaluate(X_noise, y, cv_folds=cv_folds, alphas=BASELINE_ALPHAS)
        results.append({
            "seed": seed,
            "cv_r2_mean": result["cv_r2_mean"],
            "cv_r2_std": result["cv_r2_std"],
            "cv_mse_mean": result["cv_mse_mean"],
            "cv_mse_std": result["cv_mse_std"],
            "best_alpha": result["best_alpha"],
        })
    return results


def load_real_probe_results(data_dir: Path, layers: list[int], y: np.ndarray, data: dict, cv_folds: int) -> dict:
    """Load precomputed real probe results or train if not available."""
    existing_results_path = data_dir / "noise_baseline_results.json"
    if existing_results_path.exists():
        with open(existing_results_path) as f:
            existing = json.load(f)
        # Check if we have MSE results already
        first_layer = str(layers[0])
        if first_layer in existing.get("layers", {}) and "cv_mse_mean" in existing["layers"][first_layer].get("real_probe", {}):
            print("  Loading precomputed results from existing JSON")
            return {layer: existing["layers"][str(layer)]["real_probe"] for layer in layers}

    print("  Training real probes (no precomputed results found)...")
    results = {}
    for layer in layers:
        X = data[f"layer_{layer}"]
        _, result, _ = train_and_evaluate(X, y, cv_folds=cv_folds)
        results[layer] = {
            "cv_r2_mean": result["cv_r2_mean"],
            "cv_r2_std": result["cv_r2_std"],
            "cv_mse_mean": result["cv_mse_mean"],
            "cv_mse_std": result["cv_mse_std"],
        }
    return results


def plot_comparison(
    layers: list[int],
    real_results: dict,
    shuffled_results: dict,
    random_results: dict,
    label_variance: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(layers))
    width = 0.25

    # R² plot
    ax = axes[0]
    real_r2 = [real_results[layer]["cv_r2_mean"] for layer in layers]
    real_r2_std = [real_results[layer]["cv_r2_std"] for layer in layers]
    shuffled_r2 = [shuffled_results[layer]["cv_r2_mean"] for layer in layers]
    shuffled_r2_std = [shuffled_results[layer]["cv_r2_std"] for layer in layers]
    random_r2 = [random_results[layer]["cv_r2_mean"] for layer in layers]
    random_r2_std = [random_results[layer]["cv_r2_std"] for layer in layers]

    ax.bar(x - width, real_r2, width, yerr=real_r2_std, label="Real probe", color="steelblue", capsize=3)
    ax.bar(x, shuffled_r2, width, yerr=shuffled_r2_std, label="Shuffled labels", color="darkorange", capsize=3)
    ax.bar(x + width, random_r2, width, yerr=random_r2_std, label="Random activations", color="forestgreen", capsize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV R²")
    ax.set_title("R² (higher = better)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    # MSE plot
    ax = axes[1]
    real_mse = [real_results[layer]["cv_mse_mean"] for layer in layers]
    real_mse_std = [real_results[layer]["cv_mse_std"] for layer in layers]
    shuffled_mse = [shuffled_results[layer]["cv_mse_mean"] for layer in layers]
    shuffled_mse_std = [shuffled_results[layer]["cv_mse_std"] for layer in layers]
    random_mse = [random_results[layer]["cv_mse_mean"] for layer in layers]
    random_mse_std = [random_results[layer]["cv_mse_std"] for layer in layers]

    ax.bar(x - width, real_mse, width, yerr=real_mse_std, label="Real probe", color="steelblue", capsize=3)
    ax.bar(x, shuffled_mse, width, yerr=shuffled_mse_std, label="Shuffled labels", color="darkorange", capsize=3)
    ax.bar(x + width, random_mse, width, yerr=random_mse_std, label="Random activations", color="forestgreen", capsize=3)
    ax.axhline(y=label_variance, color="red", linestyle="--", linewidth=1.5, label=f"Var(y) = {label_variance:.3f}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV MSE")
    ax.set_title("MSE (lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved comparison plot to {output_path}")


def main() -> None:
    args = parse_args()

    print(f"Loading data from {args.data_dir}...")
    npz_data = np.load(args.data_dir / "activations.npz")
    task_ids = npz_data["task_ids"].tolist()

    print(f"Loading scores from {args.scores_file}...")
    with open(args.scores_file) as f:
        scores_map = json.load(f)

    # Filter to task_ids with scores
    indices = [i for i, tid in enumerate(task_ids) if tid in scores_map]
    y = np.array([scores_map[task_ids[i]] for i in indices])
    print(f"Matched {len(y)} samples with scores")

    layer_keys = [k for k in npz_data.keys() if k.startswith("layer_")]
    layers = sorted([int(k.split("_")[1]) for k in layer_keys])

    # Filter activations to matching indices
    filtered_data = {key: npz_data[key][indices] for key in layer_keys}
    print(f"Loaded {len(y)} samples, layers: {layers}")

    label_variance = float(np.var(y))
    print(f"Label variance: {label_variance:.4f} (MSE baseline for predicting mean)")

    print(f"\nRunning real probe training for comparison...")
    real_results = load_real_probe_results(
        args.data_dir, layers, y, filtered_data, args.cv_folds
    )
    for layer in layers:
        r = real_results[layer]
        print(f"  Layer {layer}: CV R² = {r['cv_r2_mean']:.3f} ± {r['cv_r2_std']:.3f}, MSE = {r['cv_mse_mean']:.4f} ± {r['cv_mse_std']:.4f}")

    all_results = {"seeds": list(range(args.n_seeds)), "layers": {}}

    for layer in layers:
        X = filtered_data[f"layer_{layer}"]
        print(f"\nLayer {layer} ({X.shape[1]} dims):")

        print(f"  Running shuffled labels baseline ({args.n_seeds} seeds)...")
        shuffled = run_shuffled_labels_baseline(X, y, args.n_seeds, args.cv_folds)
        shuffled_r2s = [r["cv_r2_mean"] for r in shuffled]
        shuffled_mses = [r["cv_mse_mean"] for r in shuffled]
        print(f"    Mean CV R²: {np.mean(shuffled_r2s):.4f} ± {np.std(shuffled_r2s):.4f}, MSE: {np.mean(shuffled_mses):.4f} ± {np.std(shuffled_mses):.4f}")

        print(f"  Running random activations baseline ({args.n_seeds} seeds)...")
        random_acts = run_random_activations_baseline(X, y, args.n_seeds, args.cv_folds)
        random_r2s = [r["cv_r2_mean"] for r in random_acts]
        random_mses = [r["cv_mse_mean"] for r in random_acts]
        print(f"    Mean CV R²: {np.mean(random_r2s):.4f} ± {np.std(random_r2s):.4f}, MSE: {np.mean(random_mses):.4f} ± {np.std(random_mses):.4f}")

        all_results["layers"][str(layer)] = {
            "shuffled_labels": {
                "cv_r2_mean": float(np.mean(shuffled_r2s)),
                "cv_r2_std": float(np.std(shuffled_r2s)),
                "cv_mse_mean": float(np.mean(shuffled_mses)),
                "cv_mse_std": float(np.std(shuffled_mses)),
                "per_seed": shuffled,
            },
            "random_activations": {
                "cv_r2_mean": float(np.mean(random_r2s)),
                "cv_r2_std": float(np.std(random_r2s)),
                "cv_mse_mean": float(np.mean(random_mses)),
                "cv_mse_std": float(np.std(random_mses)),
                "per_seed": random_acts,
            },
            "real_probe": real_results[layer],
        }

    all_results["label_variance"] = label_variance

    results_path = args.data_dir / "noise_baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    shuffled_summary = {
        layer: {
            "cv_r2_mean": all_results["layers"][str(layer)]["shuffled_labels"]["cv_r2_mean"],
            "cv_r2_std": all_results["layers"][str(layer)]["shuffled_labels"]["cv_r2_std"],
            "cv_mse_mean": all_results["layers"][str(layer)]["shuffled_labels"]["cv_mse_mean"],
            "cv_mse_std": all_results["layers"][str(layer)]["shuffled_labels"]["cv_mse_std"],
        }
        for layer in layers
    }
    random_summary = {
        layer: {
            "cv_r2_mean": all_results["layers"][str(layer)]["random_activations"]["cv_r2_mean"],
            "cv_r2_std": all_results["layers"][str(layer)]["random_activations"]["cv_r2_std"],
            "cv_mse_mean": all_results["layers"][str(layer)]["random_activations"]["cv_mse_mean"],
            "cv_mse_std": all_results["layers"][str(layer)]["random_activations"]["cv_mse_std"],
        }
        for layer in layers
    }

    plot_path = args.data_dir / "noise_baseline_comparison.png"
    plot_comparison(layers, real_results, shuffled_summary, random_summary, label_variance, plot_path)


if __name__ == "__main__":
    main()
