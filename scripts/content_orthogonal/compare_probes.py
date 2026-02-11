"""Compare standard probes vs content-orthogonal probes.

Trains Ridge probes on:
  1. Standard activations (topic-residualized scores)
  2. Content-orthogonalized activations (residualized against content embeddings)

Reports per-layer comparison table.
"""

import gc
import json
from pathlib import Path

import numpy as np

from src.probes.content_embedding import load_content_embeddings
from src.probes.content_orthogonal import residualize_activations
from src.probes.core.activations import load_activations
from src.probes.core.linear_probe import train_and_evaluate
from src.probes.data_loading import load_thurstonian_scores
from src.probes.experiments.hoo_ridge import build_ridge_xy
from src.probes.residualization import residualize_scores

RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
ACTIVATIONS_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
CONTENT_EMBEDDING_PATH = Path("activations/content_embeddings/embeddings_all-MiniLM-L6-v2.npz")
TOPICS_JSON = Path("src/analysis/topic_classification/output/topics_v2.json")
OUTPUT_DIR = Path("results/probes/content_orthogonal_comparison")

LAYERS = [31, 43, 55]
CV_FOLDS = 5
ALPHA_SWEEP_SIZE = 10


def train_ridge(task_ids: np.ndarray, activations: np.ndarray, scores: dict[str, float]) -> dict:
    indices, y = build_ridge_xy(task_ids, scores)
    X = activations[indices]
    probe, eval_results, alpha_sweep = train_and_evaluate(
        X, y, cv_folds=CV_FOLDS, alpha_sweep_size=ALPHA_SWEEP_SIZE, standardize=True,
    )
    return eval_results


def main() -> None:
    # Load data
    print("Loading scores...")
    raw_scores = load_thurstonian_scores(RUN_DIR)
    scores, resid_stats = residualize_scores(raw_scores, TOPICS_JSON, confounds=["topic"])
    print(f"  {len(scores)} tasks, topic R²={resid_stats['metadata_r2']:.4f}")

    print("Loading content embeddings...")
    content_task_ids, content_emb = load_content_embeddings(CONTENT_EMBEDDING_PATH)
    print(f"  {len(content_task_ids)} embeddings ({content_emb.shape[1]}d)")

    task_id_filter = set(scores.keys())

    results = []

    for layer in LAYERS:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        task_ids, activations = load_activations(
            ACTIVATIONS_PATH, task_id_filter=task_id_filter, layers=[layer],
        )
        act = activations[layer]

        # Standard probe
        print("  Standard probe...")
        std_results = train_ridge(task_ids, act, scores)
        print(f"    cv_R²={std_results['cv_r2_mean']:.4f} ± {std_results['cv_r2_std']:.4f}")

        # Content-orthogonal probe
        print("  Content-orthogonal probe...")
        aligned_ids, residual_act, content_stats = residualize_activations(
            act, task_ids, content_emb, content_task_ids, alpha=1.0,
        )
        print(f"    Content R²={content_stats['content_r2']:.4f}")
        co_results = train_ridge(aligned_ids, residual_act, scores)
        print(f"    cv_R²={co_results['cv_r2_mean']:.4f} ± {co_results['cv_r2_std']:.4f}")

        results.append({
            "layer": layer,
            "standard_r2": std_results["cv_r2_mean"],
            "standard_r2_std": std_results["cv_r2_std"],
            "standard_best_alpha": std_results["best_alpha"],
            "content_orth_r2": co_results["cv_r2_mean"],
            "content_orth_r2_std": co_results["cv_r2_std"],
            "content_orth_best_alpha": co_results["best_alpha"],
            "content_r2": content_stats["content_r2"],
            "n_tasks": content_stats["n_tasks"],
        })

        del activations, act, residual_act
        gc.collect()

    # Summary table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Layer':>6}  {'Standard R²':>12}  {'Content-Orth R²':>16}  {'Delta':>8}  {'Content R²':>11}  {'% Retained':>11}")
    print("-" * 75)
    for r in results:
        delta = r["content_orth_r2"] - r["standard_r2"]
        pct_retained = r["content_orth_r2"] / r["standard_r2"] * 100 if r["standard_r2"] > 0 else 0
        print(f"{r['layer']:>6}  {r['standard_r2']:>12.4f}  {r['content_orth_r2']:>16.4f}  {delta:>+8.4f}  {r['content_r2']:>11.4f}  {pct_retained:>10.1f}%")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
