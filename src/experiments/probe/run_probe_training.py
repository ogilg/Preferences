"""Train linear probes on collected activation data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.probes.data import load_probe_dataset
from src.probes.linear_probe import AlphaResult, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train probes on collected activations")
    parser.add_argument("data_dir", type=Path, help="Directory containing probe data")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    return parser.parse_args()


def plot_alpha_sweep(
    layers: list[int],
    all_alpha_results: list[list[AlphaResult]],
    best_alphas: list[float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 5), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, layer, alpha_results, best_alpha in zip(axes, layers, all_alpha_results, best_alphas):
        alphas = [r.alpha for r in alpha_results]
        train_r2s = [r.train_r2 for r in alpha_results]
        cv_r2s = [r.cv_r2_mean for r in alpha_results]
        cv_stds = [r.cv_r2_std for r in alpha_results]

        ax.semilogx(alphas, train_r2s, "o-", label="Train R²", color="steelblue")
        ax.semilogx(alphas, cv_r2s, "o-", label="Val R²", color="darkorange")
        ax.fill_between(
            alphas,
            np.array(cv_r2s) - np.array(cv_stds),
            np.array(cv_r2s) + np.array(cv_stds),
            alpha=0.2,
            color="darkorange",
        )
        ax.axvline(best_alpha, color="gray", linestyle="--", alpha=0.7, label=f"Best α={best_alpha:.1e}")
        ax.set_xlabel("Alpha (regularization)")
        ax.set_title(f"Layer {layer}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("R²")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def print_alpha_table(layer: int, alpha_results: list[AlphaResult]) -> None:
    print(f"\n  Layer {layer} alpha sweep:")
    print(f"  {'Alpha':>10}  {'Train R²':>10}  {'Val R²':>10}  {'Val Std':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in alpha_results:
        print(f"  {r.alpha:>10.2e}  {r.train_r2:>10.3f}  {r.cv_r2_mean:>10.3f}  {r.cv_r2_std:>10.3f}")


def main() -> None:
    args = parse_args()

    print(f"Loading data from {args.data_dir}...")
    data_points = load_probe_dataset(args.data_dir)
    print(f"Loaded {len(data_points)} data points")

    layers = list(data_points[0].activations.keys())
    print(f"Available layers: {layers}")

    y = np.array([dp.score for dp in data_points])
    print(f"Score range: {y.min():.2f} - {y.max():.2f}, mean: {y.mean():.2f}")

    print(f"\nTraining probes (cv_folds={args.cv_folds})...")
    all_results = []
    all_alpha_results = []
    for layer in layers:
        X = np.stack([dp.activations[layer] for dp in data_points])
        probe, results, alpha_results = train_and_evaluate(X, y, cv_folds=args.cv_folds)
        all_results.append(results)
        all_alpha_results.append(alpha_results)
        print(
            f"  Layer {layer}: CV R² = {results['cv_r2_mean']:.3f} ± {results['cv_r2_std']:.3f} "
            f"(alpha={results['best_alpha']:.2e})"
        )
        print_alpha_table(layer, alpha_results)

    plot_path = args.data_dir / "probe_results.png"
    best_alphas = [r["best_alpha"] for r in all_results]
    plot_alpha_sweep(layers, all_alpha_results, best_alphas, plot_path)


if __name__ == "__main__":
    main()
