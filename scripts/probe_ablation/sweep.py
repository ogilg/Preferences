"""Ridge probe ablation sweep: layers × demeaning × standardize.

48 combinations (6 layers × 4 demean options × 2 standardize).
Each is a Ridge CV fit — whole sweep takes a few minutes.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

from src.probes.core.activations import load_activations
from src.probes.core.linear_probe import train_and_evaluate
from src.probes.data_loading import load_thurstonian_scores
from src.probes.experiments.hoo_ridge import build_ridge_xy
from src.probes.residualization import demean_scores

load_dotenv()

ACTIVATIONS_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
TOPICS_JSON = Path("src/analysis/topic_classification/output/topics_v2.json")
OUTPUT_DIR = Path("results/probes/ablation_sweep")

LAYERS = [15, 31, 37, 43, 49, 55]
DEMEAN_OPTIONS: list[list[str] | None] = [
    None,
    ["topic"],
    ["dataset"],
    ["topic", "dataset"],
]
STANDARDIZE_OPTIONS = [True, False]

CV_FOLDS = 5
ALPHA_SWEEP_SIZE = 10


def demean_label(confounds: list[str] | None) -> str:
    if confounds is None:
        return "none"
    return "+".join(confounds)


def run_sweep() -> list[dict]:
    print("Loading scores...")
    raw_scores = load_thurstonian_scores(RUN_DIR)
    print(f"  {len(raw_scores)} tasks")

    # Pre-compute demeaned score variants
    score_variants: dict[str, dict[str, float]] = {}
    for confounds in DEMEAN_OPTIONS:
        label = demean_label(confounds)
        if confounds is None:
            score_variants[label] = raw_scores
        else:
            demeaned, stats = demean_scores(raw_scores, TOPICS_JSON, confounds=confounds)
            print(f"  Demean [{label}]: R²={stats['metadata_r2']:.4f}, n={stats['n_tasks_demeaned']}")
            score_variants[label] = demeaned

    # Load all activations once (all layers)
    print("\nLoading activations (all layers)...")
    all_task_ids = set()
    for scores in score_variants.values():
        all_task_ids.update(scores.keys())
    task_ids, activations = load_activations(
        ACTIVATIONS_PATH,
        task_id_filter=all_task_ids,
        layers=LAYERS,
    )
    print(f"  {len(task_ids)} tasks, {len(LAYERS)} layers")

    results = []
    grid = list(product(LAYERS, DEMEAN_OPTIONS, STANDARDIZE_OPTIONS))
    print(f"\nRunning {len(grid)} combinations...\n")

    for i, (layer, confounds, standardize) in enumerate(grid):
        label = demean_label(confounds)
        scores = score_variants[label]

        indices, y = build_ridge_xy(task_ids, scores)
        X = activations[layer][indices]

        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        probe, eval_results, alpha_sweep = train_and_evaluate(
            X_scaled, y, cv_folds=CV_FOLDS,
            alpha_sweep_size=ALPHA_SWEEP_SIZE,
        )

        row = {
            "layer": layer,
            "demean": label,
            "standardize": standardize,
            "n_samples": len(y),
            "cv_r2_mean": eval_results["cv_r2_mean"],
            "cv_r2_std": eval_results["cv_r2_std"],
            "train_r2": eval_results["train_r2"],
            "best_alpha": eval_results["best_alpha"],
            "cv_pearson_r_mean": eval_results["cv_pearson_r_mean"],
            "train_test_gap": eval_results["train_test_gap"],
            "cv_stability": eval_results["cv_stability"],
        }
        results.append(row)

        scale_str = "scaled" if standardize else "raw"
        print(f"  [{i+1:2d}/{len(grid)}] L{layer:02d} {label:15s} {scale_str:6s}  "
              f"cv_R²={row['cv_r2_mean']:.4f} ± {row['cv_r2_std']:.4f}  "
              f"train_R²={row['train_r2']:.4f}  "
              f"alpha={row['best_alpha']:.2e}")

    return results


def print_summary_table(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("ABLATION SWEEP SUMMARY")
    print("=" * 100)
    header = f"{'Layer':>5}  {'Demean':>15}  {'Scale':>6}  {'cv_R²':>10}  {'± std':>8}  {'train_R²':>10}  {'gap':>8}  {'alpha':>10}  {'pearson_r':>10}"
    print(header)
    print("-" * 100)

    for r in sorted(results, key=lambda x: (x["layer"], x["demean"], not x["standardize"])):
        scale_str = "yes" if r["standardize"] else "no"
        print(f"{r['layer']:>5}  {r['demean']:>15}  {scale_str:>6}  "
              f"{r['cv_r2_mean']:>10.4f}  {r['cv_r2_std']:>8.4f}  "
              f"{r['train_r2']:>10.4f}  {r['train_test_gap']:>8.4f}  "
              f"{r['best_alpha']:>10.2e}  {r['cv_pearson_r_mean']:>10.4f}")

    # Best per layer
    print("\n" + "-" * 60)
    print("BEST CONDITION PER LAYER:")
    for layer in LAYERS:
        layer_results = [r for r in results if r["layer"] == layer]
        best = max(layer_results, key=lambda x: x["cv_r2_mean"])
        scale_str = "scaled" if best["standardize"] else "raw"
        print(f"  L{layer:02d}: cv_R²={best['cv_r2_mean']:.4f}  ({best['demean']}, {scale_str})")


def main() -> None:
    results = run_sweep()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "created_at": datetime.now().isoformat(),
            "grid": {
                "layers": LAYERS,
                "demean_options": [demean_label(d) for d in DEMEAN_OPTIONS],
                "standardize_options": STANDARDIZE_OPTIONS,
            },
            "cv_folds": CV_FOLDS,
            "alpha_sweep_size": ALPHA_SWEEP_SIZE,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {summary_path}")

    print_summary_table(results)


if __name__ == "__main__":
    main()
