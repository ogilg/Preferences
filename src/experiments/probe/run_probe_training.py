"""Train linear probes on collected activation data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.probes.data import load_probe_dataset
from src.probes.linear_probe import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train probes on collected activations")
    parser.add_argument("data_dir", type=Path, help="Directory containing probe data")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading data from {args.data_dir}...")
    data_points = load_probe_dataset(args.data_dir)
    print(f"Loaded {len(data_points)} data points")

    layers = list(data_points[0].activations.keys())
    print(f"Available layers: {layers}")

    y = np.array([dp.score for dp in data_points])
    print(f"Score range: {y.min():.2f} - {y.max():.2f}, mean: {y.mean():.2f}")

    print(f"\nTraining probes (alpha={args.alpha}, cv_folds={args.cv_folds})...")
    for layer in layers:
        X = np.stack([dp.activations[layer] for dp in data_points])
        probe, results = train_and_evaluate(X, y, cv_folds=args.cv_folds, alpha=args.alpha)
        print(f"  Layer {layer}: CV R² = {results['cv_r2_mean']:.3f} ± {results['cv_r2_std']:.3f}")


if __name__ == "__main__":
    main()
