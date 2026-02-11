"""Find the best alpha for content→activation Ridge via cross-validation.

For each layer, sweep alpha and pick the one that best predicts activations
on held-out data. Then residualize with that alpha and report the
content-orthogonal probe R².
"""

import gc
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src.probes.content_embedding import load_content_embeddings
from src.probes.content_orthogonal import _align_by_task_id, project_out_content
from src.probes.core.activations import load_activations
from src.probes.core.linear_probe import train_and_evaluate
from src.probes.data_loading import load_thurstonian_scores
from src.probes.experiments.hoo_ridge import build_ridge_xy

RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
ACTIVATIONS_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
CONTENT_EMBEDDING_PATH = Path("activations/content_embeddings/embeddings_all-MiniLM-L6-v2.npz")
OUTPUT_DIR = Path("results/probes/content_orthogonal_comparison")

LAYERS = [31, 43, 55]
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
CV_FOLDS = 5
PROBE_ALPHA_SWEEP_SIZE = 10


def cv_content_r2(X: np.ndarray, Y: np.ndarray, alpha: float, n_folds: int = 5) -> float:
    """Cross-validated R² for content embeddings predicting activations."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_r2s = []
    for train_idx, val_idx in kf.split(X):
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X[train_idx], Y[train_idx])
        fold_r2s.append(ridge.score(X[val_idx], Y[val_idx]))
    return float(np.mean(fold_r2s))


def train_probe(task_ids: np.ndarray, activations: np.ndarray, scores: dict[str, float]) -> dict:
    indices, y = build_ridge_xy(task_ids, scores)
    X = activations[indices]
    _, eval_results, _ = train_and_evaluate(
        X, y, cv_folds=CV_FOLDS, alpha_sweep_size=PROBE_ALPHA_SWEEP_SIZE, standardize=True,
    )
    return eval_results


def main() -> None:
    print("Loading scores...")
    scores = load_thurstonian_scores(RUN_DIR)
    print(f"  {len(scores)} tasks")

    print("Loading content embeddings...")
    content_task_ids, content_emb = load_content_embeddings(CONTENT_EMBEDDING_PATH)
    print(f"  {len(content_task_ids)} embeddings ({content_emb.shape[1]}d)")

    task_id_filter = set(scores.keys())

    results = {}

    for layer in LAYERS:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        task_ids, activations = load_activations(
            ACTIVATIONS_PATH, task_id_filter=task_id_filter, layers=[layer],
        )
        act = activations[layer]

        # Align activations with content embeddings
        act_idx, emb_idx = _align_by_task_id(task_ids, content_task_ids)
        X = content_emb[emb_idx]
        Y = act[act_idx]
        print(f"  {len(act_idx)} aligned tasks")

        # Sweep alpha for content→activation Ridge with CV
        print(f"\n  Alpha sweep (CV R² for content → activations):")
        best_alpha = None
        best_cv_r2 = -np.inf
        sweep = []
        for alpha in ALPHAS:
            cv_r2 = cv_content_r2(X, Y, alpha)
            sweep.append({"alpha": alpha, "cv_r2": round(cv_r2, 4)})
            marker = ""
            if cv_r2 > best_cv_r2:
                best_cv_r2 = cv_r2
                best_alpha = alpha
                marker = " <-- best"
            print(f"    α={alpha:<10g}  cv_R²={cv_r2:.4f}{marker}")

        print(f"\n  Best alpha: {best_alpha} (cv_R²={best_cv_r2:.4f})")

        # Train content→activation Ridge at best alpha (on full data)
        ridge = Ridge(alpha=best_alpha, fit_intercept=True)
        ridge.fit(X, Y)
        train_r2 = ridge.score(X, Y)
        print(f"  Train R² at best alpha: {train_r2:.4f}")

        # Standard probe
        std_results = train_probe(task_ids, act, scores)
        print(f"\n  Standard probe: cv_R²={std_results['cv_r2_mean']:.4f}")

        # Content-orthogonal probe at best alpha
        aligned_ids, residual_act, content_stats = project_out_content(
            act, task_ids, content_emb, content_task_ids, alpha=best_alpha,
        )
        co_results = train_probe(aligned_ids, residual_act, scores)
        pct_retained = co_results["cv_r2_mean"] / std_results["cv_r2_mean"] * 100

        print(f"  Content-orthogonal probe (α={best_alpha}): cv_R²={co_results['cv_r2_mean']:.4f}")
        print(f"  Content R² (train): {content_stats['content_r2']:.4f}")
        print(f"  Content R² (CV): {best_cv_r2:.4f}")
        print(f"  % retained: {pct_retained:.1f}%")

        results[str(layer)] = {
            "best_alpha": best_alpha,
            "content_cv_r2": round(best_cv_r2, 4),
            "content_train_r2": round(train_r2, 4),
            "standard_probe_r2": std_results["cv_r2_mean"],
            "content_orth_probe_r2": co_results["cv_r2_mean"],
            "content_orth_probe_r2_std": co_results["cv_r2_std"],
            "pct_retained": round(pct_retained, 1),
            "alpha_sweep": sweep,
        }

        del activations, act, residual_act
        gc.collect()

    # Summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS (CV-selected alpha)")
    print(f"{'='*80}")
    print(f"{'Layer':>6}  {'Best α':>8}  {'Content cv_R²':>14}  {'Standard':>9}  {'Content-Orth':>13}  {'% Retained':>11}")
    print("-" * 70)
    for layer in LAYERS:
        r = results[str(layer)]
        print(f"{layer:>6}  {r['best_alpha']:>8g}  {r['content_cv_r2']:>14.4f}  {r['standard_probe_r2']:>9.4f}  {r['content_orth_probe_r2']:>13.4f}  {r['pct_retained']:>10.1f}%")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "cv_alpha_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
