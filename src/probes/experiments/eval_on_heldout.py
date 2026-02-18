"""Train probes on one run, evaluate on a held-out run.

Trains Ridge on 3K Thurstonian scores, uses half of the 4K eval set for
alpha sweep (Pearson r), and evaluates on the other half (Pearson r + pairwise acc).

Usage:
    python -m src.probes.experiments.eval_on_heldout --config configs/probes/heldout_eval.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.probes.bradley_terry.data import PairwiseActivationData
from src.probes.bradley_terry.training import pairwise_accuracy_from_scores
from src.probes.core.activations import load_activations
from src.probes.data_loading import load_thurstonian_scores, load_pairwise_measurements
from src.probes.experiments.hoo_ridge import build_ridge_xy
from src.probes.residualization import demean_scores


def heldout_eval(
    train_scores: dict[str, float],
    eval_scores: dict[str, float],
    task_ids_arr: np.ndarray,
    activations: dict[int, np.ndarray],
    eval_measurements: list,
    layers: list[int],
    standardize: bool = True,
    alpha_sweep_size: int = 10,
    eval_split_seed: int = 42,
) -> dict[int, dict]:
    """Core held-out evaluation logic.

    Args:
        train_scores: task_id -> score for training set
        eval_scores: task_id -> score for evaluation set
        task_ids_arr: task IDs aligned with activations
        activations: layer -> (n_tasks, d_model) array
        eval_measurements: pairwise measurements for eval run
        layers: which layers to evaluate
        standardize: whether to StandardScaler activations
        alpha_sweep_size: number of alphas to sweep
        eval_split_seed: seed for splitting eval into sweep/final

    Returns:
        dict mapping layer -> results dict with keys: best_alpha, sweep_r,
        final_r, final_acc, n_train, n_sweep, n_final, n_final_pairs, alpha_sweep
    """
    # Split eval into sweep and final-eval halves
    rng = np.random.default_rng(eval_split_seed)
    eval_task_ids = sorted(eval_scores.keys())
    perm = rng.permutation(len(eval_task_ids))
    half = len(eval_task_ids) // 2
    sweep_ids = {eval_task_ids[i] for i in perm[:half]}
    final_ids = {eval_task_ids[i] for i in perm[half:]}
    sweep_scores = {tid: eval_scores[tid] for tid in sweep_ids}
    final_scores = {tid: eval_scores[tid] for tid in final_ids}
    print(f"Eval split: {len(sweep_scores)} sweep, {len(final_scores)} final")

    # Build pairwise data and filter to final set
    id_to_idx = {tid: i for i, tid in enumerate(task_ids_arr)}
    final_idx_set = {id_to_idx[tid] for tid in final_ids if tid in id_to_idx}

    all_bt_data = PairwiseActivationData.from_measurements(eval_measurements, task_ids_arr, activations)
    final_bt_data = all_bt_data.filter_by_indices(final_idx_set)
    print(f"Final eval pairs: {len(final_bt_data.pairs)}")

    # Build train X, y
    train_indices, y_train = build_ridge_xy(task_ids_arr, train_scores)
    sweep_indices, y_sweep = build_ridge_xy(task_ids_arr, sweep_scores)
    final_indices, y_final = build_ridge_xy(task_ids_arr, final_scores)

    alphas = np.logspace(-1, 5, alpha_sweep_size)
    results_per_layer = {}

    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        acts = activations[layer]

        X_train = acts[train_indices]
        X_sweep = acts[sweep_indices]
        X_final = acts[final_indices]

        if standardize:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_sweep_scaled = scaler.transform(X_sweep)
            X_final_scaled = scaler.transform(X_final)
        else:
            X_train_scaled = X_train
            X_sweep_scaled = X_sweep
            X_final_scaled = X_final

        # Alpha sweep: train on train set, eval Pearson r on sweep half
        best_alpha = None
        best_sweep_r = -1.0
        sweep_results = []
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train_scaled, y_train)
            y_pred = ridge.predict(X_sweep_scaled)
            if len(y_pred) >= 10:
                r, _ = pearsonr(y_sweep, y_pred)
                r = float(r)
            else:
                r = float("nan")
            sweep_results.append({"alpha": float(alpha), "sweep_r": r})
            if not np.isnan(r) and r > best_sweep_r:
                best_sweep_r = r
                best_alpha = float(alpha)

        print(f"  Best alpha: {best_alpha:.4g} (sweep r={best_sweep_r:.4f})")

        # Final eval: train at best alpha, evaluate on final half
        ridge = Ridge(alpha=best_alpha)
        ridge.fit(X_train_scaled, y_train)
        y_pred_final = ridge.predict(X_final_scaled)

        if len(y_pred_final) >= 10:
            final_r, _ = pearsonr(y_final, y_pred_final)
            final_r = float(final_r)
        else:
            final_r = None

        # Pairwise accuracy on final half (predict in raw space)
        if standardize:
            coef_raw = ridge.coef_ / scaler.scale_
            intercept_raw = ridge.intercept_ - coef_raw @ scaler.mean_
        else:
            coef_raw = ridge.coef_
            intercept_raw = ridge.intercept_
        all_predicted = acts @ coef_raw + intercept_raw

        final_acc = None
        if len(final_bt_data.pairs) > 0:
            final_acc = pairwise_accuracy_from_scores(all_predicted, final_bt_data)

        final_r_str = f"{final_r:.4f}" if final_r is not None else "N/A"
        final_acc_str = f"{final_acc:.4f}" if final_acc is not None else "N/A"
        print(f"  Final: r={final_r_str}, acc={final_acc_str}")

        results_per_layer[layer] = {
            "best_alpha": best_alpha,
            "sweep_r": best_sweep_r,
            "final_r": final_r,
            "final_acc": final_acc,
            "n_train": len(y_train),
            "n_sweep": len(y_sweep),
            "n_final": len(y_final),
            "n_final_pairs": len(final_bt_data.pairs),
            "alpha_sweep": sweep_results,
        }

    return results_per_layer


def run_heldout_eval(config_path: Path) -> dict:
    """CLI wrapper: reads YAML config, calls heldout_eval(), saves JSON."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    train_run_dir = Path(cfg["train_run_dir"])
    eval_run_dir = Path(cfg["eval_run_dir"])
    activations_path = Path(cfg["activations_path"])
    output_dir = Path(cfg["output_dir"])
    layers = cfg["layers"]
    standardize = cfg.get("standardize", True)
    alpha_sweep_size = cfg.get("alpha_sweep_size", 10)
    eval_split_seed = cfg.get("eval_split_seed", 42)
    demean_confounds = cfg.get("demean_confounds")
    topics_json = Path(cfg["topics_json"]) if "topics_json" in cfg else None

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Held-out eval: {cfg['experiment_name']}")
    print(f"Train: {train_run_dir}")
    print(f"Eval:  {eval_run_dir}")

    # Load scores
    train_scores = load_thurstonian_scores(train_run_dir)
    eval_scores = load_thurstonian_scores(eval_run_dir)
    print(f"Train scores: {len(train_scores)}, Eval scores: {len(eval_scores)}")

    # Optionally demean
    if demean_confounds:
        assert topics_json is not None
        print(f"Demeaning against: {demean_confounds}")
        train_scores, train_stats = demean_scores(train_scores, topics_json, confounds=demean_confounds)
        eval_scores, eval_stats = demean_scores(eval_scores, topics_json, confounds=demean_confounds)
        print(f"  Train R²={train_stats['metadata_r2']:.4f}, Eval R²={eval_stats['metadata_r2']:.4f}")

    # Load pairwise measurements for eval run
    eval_measurements = load_pairwise_measurements(eval_run_dir)
    print(f"Eval measurements: {len(eval_measurements)}")

    # Load activations filtered to union of all task IDs
    all_needed_ids = set(train_scores.keys()) | set(eval_scores.keys())
    task_ids_arr, activations = load_activations(
        activations_path,
        task_id_filter=all_needed_ids,
        layers=layers,
    )
    print(f"Loaded {len(task_ids_arr)} tasks with activations across {len(layers)} layers")

    results_per_layer = heldout_eval(
        train_scores=train_scores,
        eval_scores=eval_scores,
        task_ids_arr=task_ids_arr,
        activations=activations,
        eval_measurements=eval_measurements,
        layers=layers,
        standardize=standardize,
        alpha_sweep_size=alpha_sweep_size,
        eval_split_seed=eval_split_seed,
    )

    # Summary
    summary = {
        "experiment_name": cfg["experiment_name"],
        "created_at": datetime.now().isoformat(),
        "train_run_dir": str(train_run_dir),
        "eval_run_dir": str(eval_run_dir),
        "activations_path": str(activations_path),
        "standardize": standardize,
        "demean_confounds": demean_confounds,
        "eval_split_seed": eval_split_seed,
        "layers": {str(k): v for k, v in results_per_layer.items()},
    }

    output_path = output_dir / "heldout_eval.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on one run, evaluate on held-out run")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_heldout_eval(args.config)


if __name__ == "__main__":
    main()
