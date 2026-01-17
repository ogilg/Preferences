"""Train linear probes on activations using rating scores as labels."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Ill-conditioned matrix")

from src.probes.linear_probe import train_and_evaluate


def load_activations(data_dir: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Load merged activations.npz.

    Returns:
        task_ids: array of task IDs
        activations: dict mapping layer -> (n_samples, hidden_dim) array
    """
    npz_path = data_dir / "activations.npz"
    data = np.load(npz_path, allow_pickle=True)

    task_ids = data["task_ids"]
    layer_keys = [k for k in data.keys() if k.startswith("layer_")]
    layers = sorted(int(k.split("_")[1]) for k in layer_keys)
    activations = {layer: data[f"layer_{layer}"] for layer in layers}

    return task_ids, activations


def load_scores(scores_path: Path) -> dict[str, float]:
    with open(scores_path) as f:
        return json.load(f)


def train_for_scores(
    task_ids: np.ndarray,
    activations: dict[int, np.ndarray],
    scores: dict[str, float],
    cv_folds: int,
) -> list[dict]:
    """Train probes for all layers using given scores."""
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    valid_indices = []
    valid_scores = []
    for task_id, score in scores.items():
        if task_id in id_to_idx:
            valid_indices.append(id_to_idx[task_id])
            valid_scores.append(score)

    if len(valid_indices) < cv_folds * 2:
        return []

    indices = np.array(valid_indices)
    y = np.array(valid_scores)

    results = []
    for layer in sorted(activations.keys()):
        X = activations[layer][indices]
        _, eval_results, _ = train_and_evaluate(X, y, cv_folds=cv_folds)

        results.append({
            "layer": layer,
            "cv_r2_mean": eval_results["cv_r2_mean"],
            "cv_r2_std": eval_results["cv_r2_std"],
            "cv_mse_mean": eval_results["cv_mse_mean"],
            "cv_mse_std": eval_results["cv_mse_std"],
            "best_alpha": eval_results["best_alpha"],
            "n_samples": len(y),
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train probes on activations")
    parser.add_argument("data_dir", type=Path, help="Directory with activations.npz")
    parser.add_argument("--ratings-dir", type=Path, help="Directory with scores_*.json files")
    parser.add_argument("--scores", type=Path, nargs="+", help="Specific score files")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Output JSON path")
    args = parser.parse_args()

    print(f"Loading activations from {args.data_dir}...")
    task_ids, activations = load_activations(args.data_dir)
    layers = sorted(activations.keys())
    print(f"Loaded {len(task_ids)} samples, layers: {layers}")

    # Collect score files
    score_files: list[Path] = []
    if args.scores:
        score_files = args.scores
    elif args.ratings_dir:
        score_files = sorted(args.ratings_dir.glob("scores_*.json"))
    else:
        ratings_dir = args.data_dir / "ratings"
        if ratings_dir.exists():
            score_files = sorted(ratings_dir.glob("scores_*.json"))

    if not score_files:
        print("No score files found.")
        return

    print(f"Found {len(score_files)} score files")

    all_results = {}

    for scores_path in score_files:
        template_name = scores_path.stem.replace("scores_", "")
        scores = load_scores(scores_path)
        print(f"\n{template_name}: {len(scores)} scores")

        results = train_for_scores(task_ids, activations, scores, args.cv_folds)

        if not results:
            print(f"  Skipped (need ≥{args.cv_folds * 2} samples for {args.cv_folds}-fold CV)")
            continue

        all_results[template_name] = results

        for r in results:
            print(f"  Layer {r['layer']}: R² = {r['cv_r2_mean']:.3f} ± {r['cv_r2_std']:.3f}")

    # Save
    output_path = args.output or args.data_dir / "probe_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
