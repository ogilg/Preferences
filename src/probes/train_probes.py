"""Train linear probes on activations using rating scores as labels."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Ill-conditioned matrix")

from src.measurement_storage.loading import (
    discover_post_stated_caches,
    get_activation_task_ids,
    load_pooled_scores,
    load_scores_from_cache,
)
from src.probes.linear_probe import train_and_evaluate


def load_activations(data_dir: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Load activations.npz, returning (task_ids, {layer: activations})."""
    npz_path = data_dir / "activations.npz"
    data = np.load(npz_path, allow_pickle=True)

    task_ids = data["task_ids"]
    layer_keys = [k for k in data.keys() if k.startswith("layer_")]
    layers = sorted(int(k.split("_")[1]) for k in layer_keys)
    activations = {layer: data[f"layer_{layer}"] for layer in layers}

    return task_ids, activations


def load_scores_from_json(scores_path: Path) -> dict[str, float]:
    """Load scores from old JSON format."""
    with open(scores_path) as f:
        return json.load(f)


def filter_activations_by_origin(
    task_ids: np.ndarray,
    origin: str,
    activations_dir: Path,
) -> np.ndarray:
    """Return boolean mask for tasks matching origin dataset."""
    matching_ids = get_activation_task_ids(activations_dir, origin_filter=origin)
    return np.array([tid in matching_ids for tid in task_ids])


def train_for_scores(
    task_ids: np.ndarray,
    activations: dict[int, np.ndarray],
    scores: dict[str, float],
    cv_folds: int,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    """Train probes for all layers, returning (results, {layer: weights})."""
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    valid_indices = []
    valid_scores = []
    for task_id, score in scores.items():
        if task_id in id_to_idx:
            valid_indices.append(id_to_idx[task_id])
            valid_scores.append(score)

    if len(valid_indices) < cv_folds * 2:
        return [], {}

    indices = np.array(valid_indices)
    y = np.array(valid_scores)

    results = []
    probes = {}
    for layer in sorted(activations.keys()):
        X = activations[layer][indices]
        probe, eval_results, _ = train_and_evaluate(X, y, cv_folds=cv_folds)

        results.append({
            "layer": layer,
            "cv_r2_mean": eval_results["cv_r2_mean"],
            "cv_r2_std": eval_results["cv_r2_std"],
            "cv_mse_mean": eval_results["cv_mse_mean"],
            "cv_mse_std": eval_results["cv_mse_std"],
            "best_alpha": eval_results["best_alpha"],
            "n_samples": len(y),
        })

        # Store weights: [coef..., intercept]
        probes[layer] = np.append(probe.coef_, probe.intercept_)

    return results, probes


def main() -> None:
    parser = argparse.ArgumentParser(description="Train probes on activations")
    parser.add_argument("data_dir", type=Path, help="Directory with activations.npz")
    parser.add_argument("--ratings-dir", type=Path, help="Directory with scores_*.json files (legacy)")
    parser.add_argument("--scores", type=Path, nargs="+", help="Specific score files (legacy)")
    parser.add_argument("--use-cache", action="store_true", help="Load from PostStatedCache")
    parser.add_argument("--model-filter", type=str, help="Filter caches by model name")
    parser.add_argument("--template-filter", type=str, help="Filter caches by template name")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Output JSON path")
    parser.add_argument("--save-probes", type=Path, help="Directory to save probe weights")
    args = parser.parse_args()

    print(f"Loading activations from {args.data_dir}...")
    task_ids, activations = load_activations(args.data_dir)
    layers = sorted(activations.keys())
    print(f"Loaded {len(task_ids)} samples, layers: {layers}")

    # Collect score sources
    score_sources: list[tuple[str, Path, bool]] = []  # (name, path, is_cache)

    if args.use_cache:
        caches = discover_post_stated_caches(args.model_filter, args.template_filter)
        for name, cache_dir in caches:
            score_sources.append((name, cache_dir, True))
        print(f"Found {len(caches)} PostStatedCache directories")
    elif args.scores:
        for path in args.scores:
            score_sources.append((path.stem.replace("scores_", ""), path, False))
    elif args.ratings_dir:
        for path in sorted(args.ratings_dir.glob("scores_*.json")):
            score_sources.append((path.stem.replace("scores_", ""), path, False))
    else:
        ratings_dir = args.data_dir / "ratings"
        if ratings_dir.exists():
            for path in sorted(ratings_dir.glob("scores_*.json")):
                score_sources.append((path.stem.replace("scores_", ""), path, False))

    if not score_sources:
        print("No score sources found. Use --use-cache or provide score files.")
        return

    print(f"Found {len(score_sources)} score sources")

    # Setup probe output directory
    probes_dir = args.save_probes
    if probes_dir:
        probes_dir.mkdir(parents=True, exist_ok=True)
        print(f"Will save probe weights to {probes_dir}")

    all_results = {}
    all_probes = {}

    for name, path, is_cache in score_sources:
        if is_cache:
            scores = load_scores_from_cache(path)
        else:
            scores = load_scores_from_json(path)

        print(f"\n{name}: {len(scores)} scores")

        results, probes = train_for_scores(task_ids, activations, scores, args.cv_folds)

        if not results:
            print(f"  Skipped (need ≥{args.cv_folds * 2} samples for {args.cv_folds}-fold CV)")
            continue

        all_results[name] = results
        all_probes[name] = probes

        for r in results:
            print(f"  Layer {r['layer']}: R² = {r['cv_r2_mean']:.3f} ± {r['cv_r2_std']:.3f}")

    # Save results
    output_path = args.output or args.data_dir / "probe_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Save probe weights
    if probes_dir and all_probes:
        for name, layer_probes in all_probes.items():
            for layer, weights in layer_probes.items():
                probe_path = probes_dir / f"{name}_layer{layer}.npy"
                np.save(probe_path, weights)
        print(f"Saved {sum(len(p) for p in all_probes.values())} probe weight files to {probes_dir}")


if __name__ == "__main__":
    main()
