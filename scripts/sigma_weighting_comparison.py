"""Compare Ridge probes with different sigma weighting strategies.

Train on 3K, sweep alpha on 4K val, report results.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from src.probes.core.activations import load_activations
from src.probes.core.evaluate import evaluate_probe_on_data
from src.probes.core.linear_probe import get_default_alphas
from src.probes.data_loading import load_thurstonian_scores_with_sigma, load_pairwise_measurements
from src.probes.bradley_terry.data import PairwiseActivationData
from src.probes.experiments.hoo_ridge import build_ridge_xy
from src.measurement.storage.loading import load_run_utilities

load_dotenv()

TRAIN_RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/"
                      "completion_preference_gemma-3-27b_completion_canonical_seed0")
VAL_RUN_DIR = Path("results/experiments/gemma3_4k_pre_task/pre_task_active_learning/"
                    "completion_preference_gemma-3-27b_completion_canonical_seed0")
ACTIVATIONS_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")

LAYERS = [15, 31, 37, 43, 49, 55]
ALPHA_SWEEP_SIZE = 50
SIGMA_MODES = ["none", "inverse_variance", "inverse_sigma"]


def compute_sigma_weights(
    task_ids: np.ndarray,
    scores: dict[str, float],
    sigmas: dict[str, float],
    mode: str,
) -> np.ndarray:
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    sigma_vals = []
    for task_id in scores:
        if task_id in id_to_idx:
            sigma_vals.append(sigmas[task_id])
    sigma_arr = np.array(sigma_vals)
    if mode == "inverse_variance":
        return 1.0 / (sigma_arr ** 2)
    elif mode == "inverse_sigma":
        return 1.0 / sigma_arr
    raise ValueError(f"Unknown mode: {mode}")


def sweep_alpha_on_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> list[dict]:
    """Sweep alphas: fit on train, evaluate on val."""
    sweep = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        if sample_weight is not None:
            ridge.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            ridge.fit(X_train, y_train)

        y_pred_train = ridge.predict(X_train)
        y_pred_val = ridge.predict(X_val)

        train_r2 = float(np.corrcoef(y_train, y_pred_train)[0, 1] ** 2)
        val_r2 = float(r2_score(y_val, y_pred_val))
        val_r, _ = pearsonr(y_val, y_pred_val)

        sweep.append({
            "alpha": float(alpha),
            "train_r2": train_r2,
            "val_r2": val_r2,
            "val_pearson_r": float(val_r),
        })
    return sweep


def main():
    # Load train data (3K)
    print("Loading 3K training data...")
    train_scores, train_sigmas = load_thurstonian_scores_with_sigma(TRAIN_RUN_DIR)
    print(f"  {len(train_scores)} tasks")
    if train_sigmas is not None:
        sigma_arr = np.array([train_sigmas[tid] for tid in train_scores])
        print(f"  Sigma stats: mean={sigma_arr.mean():.4f}, "
              f"median={np.median(sigma_arr):.4f}, "
              f"min={sigma_arr.min():.4f}, max={sigma_arr.max():.4f}")

    # Load val data (4K)
    print("\nLoading 4K validation data...")
    val_scores_arr, val_task_ids = load_run_utilities(VAL_RUN_DIR)
    val_scores = dict(zip(val_task_ids, val_scores_arr))
    try:
        val_measurements = load_pairwise_measurements(VAL_RUN_DIR)
        print(f"  {len(val_scores)} tasks, {len(val_measurements)} pairwise comparisons")
    except FileNotFoundError:
        val_measurements = []
        print(f"  {len(val_scores)} tasks (no pairwise measurements)")

    # Check overlap
    overlap = set(train_scores.keys()) & set(val_scores.keys())
    print(f"\n  Task overlap between 3K and 4K: {len(overlap)}")
    assert len(overlap) == 0, "Expected no overlap between 3K train and 4K val"

    all_task_ids_needed = set(train_scores.keys()) | set(val_scores.keys())
    alphas = get_default_alphas(ALPHA_SWEEP_SIZE)

    results_by_mode: dict[str, list[dict]] = {mode: [] for mode in SIGMA_MODES}

    for layer in LAYERS:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        task_ids, activations = load_activations(
            ACTIVATIONS_PATH,
            task_id_filter=all_task_ids_needed,
            layers=[layer],
        )
        acts = activations[layer]

        # Build pairwise data for val set
        val_bt_data = None
        if val_measurements:
            val_bt_data = PairwiseActivationData.from_measurements(
                val_measurements, task_ids, activations,
            )

        # Build train X, y
        train_indices, y_train = build_ridge_xy(task_ids, train_scores)
        X_train = acts[train_indices]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Build val X, y (in same scaler space)
        val_indices, y_val = build_ridge_xy(task_ids, val_scores)
        X_val_scaled = scaler.transform(acts[val_indices])

        for mode in SIGMA_MODES:
            print(f"\n  --- {mode} ---")

            sw = None
            if mode != "none" and train_sigmas is not None:
                sw = compute_sigma_weights(task_ids, train_scores, train_sigmas, mode)

            # Sweep alpha on val
            sweep = sweep_alpha_on_val(
                X_train_scaled, y_train, X_val_scaled, y_val,
                alphas, sample_weight=sw,
            )
            best = max(sweep, key=lambda s: s["val_r2"])

            # Fit final probe at best alpha
            ridge = Ridge(alpha=best["alpha"])
            if sw is not None:
                ridge.fit(X_train_scaled, y_train, sample_weight=sw)
            else:
                ridge.fit(X_train_scaled, y_train)

            # Convert to raw space
            coef_raw = ridge.coef_ / scaler.scale_
            intercept_raw = ridge.intercept_ - coef_raw @ scaler.mean_
            weights = np.append(coef_raw, intercept_raw)

            # Full eval on 4K (includes pairwise acc)
            val_result = evaluate_probe_on_data(
                probe_weights=weights,
                activations=acts,
                scores=val_scores_arr,
                task_ids_data=task_ids,
                task_ids_scores=val_task_ids,
                pairwise_data=val_bt_data,
            )

            print(f"  best_alpha={best['alpha']:.2g}, "
                  f"train_R²={best['train_r2']:.4f}")
            print(f"  Val 4K: R²={val_result['r2']:.4f}, "
                  f"r={val_result['pearson_r']:.4f}, "
                  f"n={val_result['n_samples']}")
            if val_result.get("pairwise_acc") is not None:
                print(f"  Val 4K pairwise acc: {val_result['pairwise_acc']:.4f}")

            results_by_mode[mode].append({
                "layer": layer,
                "best_alpha": best["alpha"],
                "train_r2": best["train_r2"],
                "val_r2": val_result["r2"],
                "val_r2_adjusted": val_result["r2_adjusted"],
                "val_pearson_r": val_result["pearson_r"],
                "val_pairwise_acc": val_result.get("pairwise_acc"),
                "val_n": val_result["n_samples"],
                "sweep": sweep,
            })

        del activations, acts
        gc.collect()

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY: Train on 3K, Alpha selected on 4K val")
    print(f"{'='*80}")
    header = f"{'Layer':>6} {'Mode':>18} {'best_α':>8} {'train_R²':>9} {'val_R²':>8} {'val_r':>8} {'val_acc':>8}"
    print(header)
    print("-" * len(header))

    for layer in LAYERS:
        for mode in SIGMA_MODES:
            entry = next(r for r in results_by_mode[mode] if r["layer"] == layer)
            acc_str = f"{entry['val_pairwise_acc']:.4f}" if entry["val_pairwise_acc"] is not None else "N/A"
            print(f"{layer:>6} {mode:>18} {entry['best_alpha']:>8.2g} "
                  f"{entry['train_r2']:>9.4f} {entry['val_r2']:>8.4f} "
                  f"{entry['val_pearson_r']:>8.4f} {acc_str:>8}")

    # Per-mode averages
    print(f"\n{'='*80}")
    print("AVERAGES ACROSS LAYERS")
    print(f"{'='*80}")
    for mode in SIGMA_MODES:
        entries = results_by_mode[mode]
        avg_train_r2 = np.mean([e["train_r2"] for e in entries])
        avg_val_r2 = np.mean([e["val_r2"] for e in entries])
        avg_val_r = np.mean([e["val_pearson_r"] for e in entries])
        accs = [e["val_pairwise_acc"] for e in entries if e["val_pairwise_acc"] is not None]
        avg_acc = np.mean(accs) if accs else float("nan")
        print(f"  {mode:>18}: train_R²={avg_train_r2:.4f}, val_R²={avg_val_r2:.4f}, "
              f"val_r={avg_val_r:.4f}, val_acc={avg_acc:.4f}")


if __name__ == "__main__":
    main()
