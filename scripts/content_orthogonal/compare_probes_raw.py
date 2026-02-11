"""Compare standard vs content-orthogonal probes on RAW (non-residualized) scores.

Same as compare_probes.py but without topic residualization,
matching the manifest_standardized.json baseline (R²≈0.86).
"""

import gc
import json
from pathlib import Path

import numpy as np

from src.probes.content_embedding import load_content_embeddings
from src.probes.content_orthogonal import project_out_content
from src.probes.core.activations import load_activations
from src.probes.core.linear_probe import train_and_evaluate
from src.probes.data_loading import load_thurstonian_scores
from src.probes.experiments.hoo_ridge import build_ridge_xy

RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
ACTIVATIONS_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
CONTENT_EMBEDDING_PATH = Path("activations/content_embeddings/embeddings_all-MiniLM-L6-v2.npz")
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
    # Load RAW scores (no residualization)
    print("Loading RAW Thurstonian scores (no residualization)...")
    scores = load_thurstonian_scores(RUN_DIR)
    print(f"  {len(scores)} tasks")

    print("Loading content embeddings...")
    content_task_ids, content_emb = load_content_embeddings(CONTENT_EMBEDDING_PATH)
    print(f"  {len(content_task_ids)} embeddings ({content_emb.shape[1]}d)")

    task_id_filter = set(scores.keys())

    # Content-only baseline on raw scores
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    emb_lookup = {tid: i for i, tid in enumerate(content_task_ids)}
    tids, y_vals, emb_indices = [], [], []
    for tid, score in scores.items():
        if tid in emb_lookup:
            tids.append(tid)
            y_vals.append(score)
            emb_indices.append(emb_lookup[tid])
    X_content = content_emb[emb_indices]
    y_content = np.array(y_vals)

    best_content_r2 = -np.inf
    for alpha in np.logspace(0, 6, 10):
        cv_scores = cross_val_score(Ridge(alpha=alpha), X_content, y_content, cv=5, scoring="r2")
        if np.mean(cv_scores) > best_content_r2:
            best_content_r2 = np.mean(cv_scores)

    print(f"\nContent-only baseline on RAW scores: cv_R²={best_content_r2:.4f}")

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
        aligned_ids, residual_act, content_stats = project_out_content(
            act, task_ids, content_emb, content_task_ids, alpha=1.0,
        )
        print(f"    Content R² of activations={content_stats['content_r2']:.4f}")
        co_results = train_ridge(aligned_ids, residual_act, scores)
        print(f"    cv_R²={co_results['cv_r2_mean']:.4f} ± {co_results['cv_r2_std']:.4f}")

        results.append({
            "layer": layer,
            "standard_r2": std_results["cv_r2_mean"],
            "standard_r2_std": std_results["cv_r2_std"],
            "content_orth_r2": co_results["cv_r2_mean"],
            "content_orth_r2_std": co_results["cv_r2_std"],
            "content_r2_activations": content_stats["content_r2"],
            "content_only_r2": best_content_r2,
            "n_tasks": content_stats["n_tasks"],
        })

        del activations, act, residual_act
        gc.collect()

    # Summary
    print(f"\n{'='*80}")
    print("COMPARISON TABLE (RAW SCORES, NO TOPIC RESIDUALIZATION)")
    print(f"{'='*80}")
    print(f"{'Layer':>6}  {'Standard R²':>12}  {'Content-Orth R²':>16}  {'Delta':>8}  {'Content R² Act':>15}  {'% Retained':>11}")
    print("-" * 80)
    for r in results:
        delta = r["content_orth_r2"] - r["standard_r2"]
        pct = r["content_orth_r2"] / r["standard_r2"] * 100
        print(f"{r['layer']:>6}  {r['standard_r2']:>12.4f}  {r['content_orth_r2']:>16.4f}  {delta:>+8.4f}  {r['content_r2_activations']:>15.4f}  {pct:>10.1f}%")

    print(f"\nContent-only baseline (embeddings → raw scores): cv_R²={best_content_r2:.4f}")

    # Save
    output_path = OUTPUT_DIR / "comparison_results_raw.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
