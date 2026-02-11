"""Content-orthogonal analysis: content-only baseline + content R² of activations.

Complements run.py which does the standard vs content-orthogonal probe comparison.
This script answers:
  1. How well do content embeddings alone predict preferences? (content-only baseline)
  2. How much activation variance is content-predictable per layer? (content R²)
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from src.probes.content_embedding import load_content_embeddings
from src.probes.content_orthogonal import _align_by_task_id
from src.probes.core.activations import load_activations
from src.probes.data_loading import load_thurstonian_scores
from src.probes.residualization import demean_scores

RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
ACTIVATIONS_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
CONTENT_EMBEDDING_PATH = Path("activations/content_embeddings/embeddings_all-MiniLM-L6-v2.npz")
TOPICS_JSON = Path("src/analysis/topic_classification/output/topics_v2.json")

LAYERS = [31, 43, 55]
CV_FOLDS = 5
ALPHA_RANGE = np.logspace(0, 6, 10)


def content_only_baseline(scores: dict[str, float], content_task_ids: np.ndarray, content_emb: np.ndarray) -> dict:
    """Train Ridge on content embeddings to predict preference scores."""
    emb_lookup = {tid: i for i, tid in enumerate(content_task_ids)}

    tids = []
    y_vals = []
    emb_indices = []
    for tid, score in scores.items():
        if tid in emb_lookup:
            tids.append(tid)
            y_vals.append(score)
            emb_indices.append(emb_lookup[tid])

    X = content_emb[emb_indices]
    y = np.array(y_vals)
    print(f"\nContent-only baseline: {len(y)} tasks, {X.shape[1]}d embeddings")

    best_alpha = None
    best_r2 = -np.inf
    sweep = []
    for alpha in ALPHA_RANGE:
        cv_scores = cross_val_score(
            Ridge(alpha=alpha, fit_intercept=True),
            X, y, cv=CV_FOLDS, scoring="r2",
        )
        mean_r2 = float(np.mean(cv_scores))
        std_r2 = float(np.std(cv_scores))
        sweep.append({"alpha": float(alpha), "cv_r2_mean": mean_r2, "cv_r2_std": std_r2})
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_alpha = float(alpha)

    print(f"  Best alpha={best_alpha:.1f}, cv_R²={best_r2:.4f}")

    return {
        "n_tasks": len(y),
        "d_embed": X.shape[1],
        "best_alpha": best_alpha,
        "best_cv_r2": best_r2,
        "alpha_sweep": sweep,
    }


def content_r2_of_activations(
    content_task_ids: np.ndarray,
    content_emb: np.ndarray,
    task_id_filter: set[str],
) -> dict:
    """Fit Ridge: content embeddings → activations. How much activation variance is content-predictable?"""
    results = {}

    for layer in LAYERS:
        task_ids, activations = load_activations(
            ACTIVATIONS_PATH,
            task_id_filter=task_id_filter,
            layers=[layer],
        )
        act = activations[layer]

        act_idx, emb_idx = _align_by_task_id(task_ids, content_task_ids)
        X = content_emb[emb_idx]
        Y = act[act_idx]

        # Fit Ridge and compute R²
        best_alpha = None
        best_r2 = -np.inf
        for alpha in ALPHA_RANGE:
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X, Y)
            r2 = ridge.score(X, Y)
            if r2 > best_r2:
                best_r2 = r2
                best_alpha = alpha

        # Cross-validated R² at best alpha
        ridge = Ridge(alpha=best_alpha, fit_intercept=True)
        ridge.fit(X, Y)
        predicted = ridge.predict(X)
        ss_res = np.sum((Y - predicted) ** 2)
        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
        train_r2 = 1.0 - ss_res / ss_tot

        # Per-dimension analysis
        per_dim_r2 = []
        for d in range(Y.shape[1]):
            ss_res_d = np.sum((Y[:, d] - predicted[:, d]) ** 2)
            ss_tot_d = np.sum((Y[:, d] - Y[:, d].mean()) ** 2)
            per_dim_r2.append(1.0 - ss_res_d / ss_tot_d if ss_tot_d > 0 else 0.0)
        per_dim_r2 = np.array(per_dim_r2)

        results[layer] = {
            "train_r2": round(float(train_r2), 4),
            "best_alpha": float(best_alpha),
            "n_tasks": len(act_idx),
            "d_model": Y.shape[1],
            "per_dim_r2_mean": round(float(np.mean(per_dim_r2)), 4),
            "per_dim_r2_median": round(float(np.median(per_dim_r2)), 4),
            "per_dim_r2_std": round(float(np.std(per_dim_r2)), 4),
        }

        print(f"  Layer {layer}: content R²={train_r2:.4f} (alpha={best_alpha:.1f})")
        print(f"    Per-dim R²: mean={np.mean(per_dim_r2):.4f}, median={np.median(per_dim_r2):.4f}")

        del activations

    return results


def main() -> None:
    # Load scores (topic-residualized, matching the standard probe setup)
    print("Loading Thurstonian scores...")
    raw_scores = load_thurstonian_scores(RUN_DIR)
    scores, resid_stats = demean_scores(raw_scores, TOPICS_JSON, confounds=["topic"])
    print(f"  {len(scores)} tasks, topic R²={resid_stats['metadata_r2']:.4f}")

    # Load content embeddings
    print("Loading content embeddings...")
    content_task_ids, content_emb = load_content_embeddings(CONTENT_EMBEDDING_PATH)
    print(f"  {len(content_task_ids)} embeddings ({content_emb.shape[1]}d)")

    task_id_filter = set(scores.keys())

    # 1. Content-only baseline
    print("\n" + "=" * 60)
    print("1. CONTENT-ONLY BASELINE")
    print("=" * 60)
    content_baseline = content_only_baseline(scores, content_task_ids, content_emb)

    # 2. Content R² of activations
    print("\n" + "=" * 60)
    print("2. CONTENT R² OF ACTIVATIONS")
    print("=" * 60)
    content_r2 = content_r2_of_activations(content_task_ids, content_emb, task_id_filter)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nContent-only baseline (embeddings → scores): cv_R²={content_baseline['best_cv_r2']:.4f}")
    print(f"\nContent R² of activations (embeddings → activations):")
    for layer in LAYERS:
        r = content_r2[layer]
        print(f"  Layer {layer}: R²={r['train_r2']:.4f} (per-dim mean={r['per_dim_r2_mean']:.4f})")


if __name__ == "__main__":
    main()
