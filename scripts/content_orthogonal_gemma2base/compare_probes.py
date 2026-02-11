"""Compare content-orthogonal probes: sentence-transformer vs Gemma-2 27B base embeddings.

Runs the full comparison pipeline for both content encoders:
  1. Standard probe on raw activations
  2. Content-orthogonal probe with sentence-transformer (384d)
  3. Content-orthogonal probe with Gemma-2 9B base (3584d)
  4. Content-only baselines for both encoders

Usage:
  python scripts/content_orthogonal_gemma2base/compare_probes.py
"""

import gc
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from src.probes.content_embedding import load_content_embeddings
from src.probes.content_orthogonal import project_out_content
from src.probes.core.activations import load_activations
from src.probes.core.linear_probe import train_and_evaluate
from src.probes.data_loading import load_thurstonian_scores
from src.probes.experiments.hoo_ridge import build_ridge_xy

RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
ACTIVATIONS_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
ST_EMBEDDING_PATH = Path("activations/content_embeddings/embeddings_all-MiniLM-L6-v2.npz")
GEMMA2_EMBEDDING_PATH = Path("activations/content_embeddings/embeddings_gemma-2-9b-base.npz")
OUTPUT_DIR = Path("results/probes/content_orthogonal_gemma2base")

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


def content_only_baseline(scores: dict[str, float], content_task_ids: np.ndarray, content_emb: np.ndarray, label: str) -> float:
    emb_lookup = {tid: i for i, tid in enumerate(content_task_ids)}
    tids, y_vals, emb_indices = [], [], []
    for tid, score in scores.items():
        if tid in emb_lookup:
            tids.append(tid)
            y_vals.append(score)
            emb_indices.append(emb_lookup[tid])
    X = content_emb[emb_indices]
    y = np.array(y_vals)

    best_r2 = -np.inf
    for alpha in np.logspace(0, 6, 10):
        cv_scores = cross_val_score(Ridge(alpha=alpha), X, y, cv=CV_FOLDS, scoring="r2")
        if np.mean(cv_scores) > best_r2:
            best_r2 = float(np.mean(cv_scores))

    print(f"  Content-only baseline ({label}): cv_R²={best_r2:.4f}")
    return best_r2


def run_content_orthogonal(
    layer: int,
    task_ids: np.ndarray,
    act: np.ndarray,
    scores: dict[str, float],
    content_task_ids: np.ndarray,
    content_emb: np.ndarray,
    label: str,
) -> dict:
    # CV alpha selection for content -> activation Ridge
    from src.probes.content_orthogonal import _align_by_task_id
    from sklearn.model_selection import KFold

    act_idx, emb_idx = _align_by_task_id(task_ids, content_task_ids)
    X_cv = content_emb[emb_idx]
    Y_cv = act[act_idx]

    best_alpha = 1.0
    best_cv_r2 = -np.inf
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_r2s = []
        for train_idx, val_idx in kf.split(X_cv):
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X_cv[train_idx], Y_cv[train_idx])
            fold_r2s.append(ridge.score(X_cv[val_idx], Y_cv[val_idx]))
        cv_r2 = float(np.mean(fold_r2s))
        if cv_r2 > best_cv_r2:
            best_cv_r2 = cv_r2
            best_alpha = alpha

    print(f"    Best alpha for {label}: {best_alpha} (content→act cv_R²={best_cv_r2:.4f})")

    aligned_ids, residual_act, content_stats = project_out_content(
        act, task_ids, content_emb, content_task_ids, alpha=best_alpha,
    )
    co_results = train_ridge(aligned_ids, residual_act, scores)
    return {
        "cv_r2_mean": co_results["cv_r2_mean"],
        "cv_r2_std": co_results["cv_r2_std"],
        "best_alpha": co_results["best_alpha"],
        "content_r2_train": content_stats["content_r2"],
        "content_r2_cv": best_cv_r2,
        "residual_alpha": best_alpha,
    }


def main() -> None:
    # Load scores
    print("Loading RAW Thurstonian scores...")
    scores = load_thurstonian_scores(RUN_DIR)
    print(f"  {len(scores)} tasks")

    # Load embeddings
    encoders = {}

    if ST_EMBEDDING_PATH.exists():
        print("Loading sentence-transformer embeddings...")
        st_task_ids, st_emb = load_content_embeddings(ST_EMBEDDING_PATH)
        print(f"  {len(st_task_ids)} embeddings ({st_emb.shape[1]}d)")
        encoders["sentence-transformer (384d)"] = (st_task_ids, st_emb)

    if GEMMA2_EMBEDDING_PATH.exists():
        print("Loading Gemma-2 27B base embeddings...")
        g2_task_ids, g2_emb = load_content_embeddings(GEMMA2_EMBEDDING_PATH)
        print(f"  {len(g2_task_ids)} embeddings ({g2_emb.shape[1]}d)")
        encoders["Gemma-2 9B base (3584d)"] = (g2_task_ids, g2_emb)

    if not encoders:
        print("ERROR: No embeddings found!")
        return

    task_id_filter = set(scores.keys())

    # Content-only baselines
    print("\nContent-only baselines (embeddings → raw scores):")
    content_baselines = {}
    for label, (ctids, cemb) in encoders.items():
        content_baselines[label] = content_only_baseline(scores, ctids, cemb, label)

    # Per-layer comparison
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

        layer_result = {
            "layer": layer,
            "standard_r2": std_results["cv_r2_mean"],
            "standard_r2_std": std_results["cv_r2_std"],
        }

        for label, (ctids, cemb) in encoders.items():
            short = "st" if "sentence" in label else "gemma2"
            print(f"  Content-orthogonal probe ({label})...")
            co = run_content_orthogonal(layer, task_ids, act, scores, ctids, cemb, label)
            print(f"    cv_R²={co['cv_r2_mean']:.4f} ± {co['cv_r2_std']:.4f}")
            layer_result[f"{short}_co_r2"] = co["cv_r2_mean"]
            layer_result[f"{short}_co_r2_std"] = co["cv_r2_std"]
            layer_result[f"{short}_content_r2_train"] = co["content_r2_train"]
            layer_result[f"{short}_content_r2_cv"] = co["content_r2_cv"]
            layer_result[f"{short}_residual_alpha"] = co["residual_alpha"]

        results.append(layer_result)
        del activations, act
        gc.collect()

    # Summary
    print(f"\n{'='*90}")
    print("COMPARISON TABLE")
    print(f"{'='*90}")

    has_st = "sentence-transformer (384d)" in encoders
    has_g2 = "Gemma-2 9B base (3584d)" in encoders

    header = f"{'Layer':>6}  {'Standard':>9}"
    if has_st:
        header += f"  {'ST-Orth R²':>11}  {'ST % Ret':>9}"
    if has_g2:
        header += f"  {'G2-Orth R²':>11}  {'G2 % Ret':>9}"
    if has_st and has_g2:
        header += f"  {'G2 vs ST':>9}"
    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r['layer']:>6}  {r['standard_r2']:>9.4f}"
        if has_st:
            pct_st = r["st_co_r2"] / r["standard_r2"] * 100
            line += f"  {r['st_co_r2']:>11.4f}  {pct_st:>8.1f}%"
        if has_g2:
            pct_g2 = r["gemma2_co_r2"] / r["standard_r2"] * 100
            line += f"  {r['gemma2_co_r2']:>11.4f}  {pct_g2:>8.1f}%"
        if has_st and has_g2:
            delta = r["gemma2_co_r2"] - r["st_co_r2"]
            line += f"  {delta:>+9.4f}"
        print(line)

    if has_st:
        print(f"\nContent-only baseline (ST): {content_baselines['sentence-transformer (384d)']:.4f}")
    if has_g2:
        print(f"Content-only baseline (G2): {content_baselines['Gemma-2 9B base (3584d)']:.4f}")

    # Content R² comparison
    if has_st and has_g2:
        print(f"\nContent→Activation R² comparison:")
        print(f"{'Layer':>6}  {'ST train':>9}  {'ST cv':>7}  {'G2 train':>9}  {'G2 cv':>7}")
        for r in results:
            print(f"{r['layer']:>6}  {r['st_content_r2_train']:>9.4f}  {r['st_content_r2_cv']:>7.4f}  {r['gemma2_content_r2_train']:>9.4f}  {r['gemma2_content_r2_cv']:>7.4f}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "results": results,
        "content_baselines": content_baselines,
        "layers": LAYERS,
    }
    output_path = OUTPUT_DIR / "comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
