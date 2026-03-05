"""Evaluate the noprompt probe on evil persona activations + utilities.

Replicates section 5.1 methodology: train probe on noprompt data (split_a+c),
sweep alpha on half of split_b, evaluate on the other half — but now also
evaluate that same probe on each evil persona's activations + utilities.

Usage: python -m scripts.multi_role_ablation.eval_noprompt_on_evil
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.probes.core.activations import load_probe_data
from src.probes.core.linear_probe import get_default_alphas
from src.probes.data_loading import load_thurstonian_scores

# All personas: original + evil
ALL_PERSONAS = [
    "noprompt", "villain", "aesthete", "midwest",
    "provocateur", "trickster", "autocrat", "sadist",
]

ACTIVATION_PATHS = {
    "noprompt": Path("activations/gemma_3_27b/activations_prompt_last.npz"),
    "villain": Path("activations/gemma_3_27b_villain/activations_prompt_last.npz"),
    "midwest": Path("activations/gemma_3_27b_midwest/activations_prompt_last.npz"),
    "aesthete": Path("activations/gemma_3_27b_aesthete/activations_prompt_last.npz"),
    "provocateur": Path("activations/gemma_3_27b_provocateur/activations_prompt_last.npz"),
    "trickster": Path("activations/gemma_3_27b_trickster/activations_prompt_last.npz"),
    "autocrat": Path("activations/gemma_3_27b_autocrat/activations_prompt_last.npz"),
    "sadist": Path("activations/gemma_3_27b_sadist/activations_prompt_last.npz"),
}

# Map persona -> (experiment_dir, sys_hash)
PERSONA_RUNS = {
    "noprompt": (Path("results/experiments/mra_exp2/pre_task_active_learning"), ""),
    "villain": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "syse8f24ac6"),
    "aesthete": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "sys021d8ca1"),
    "midwest": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "sys5d504504"),
    "provocateur": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sysf4d93514"),
    "trickster": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys09a42edc"),
    "autocrat": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys1c18219a"),
    "sadist": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys39e01d59"),
}

SPLIT_TASK_ID_FILES = {
    "a": Path("configs/measurement/active_learning/mra_exp2_split_a_1000_task_ids.txt"),
    "b": Path("configs/measurement/active_learning/mra_exp2_split_b_500_task_ids.txt"),
    "c": Path("configs/measurement/active_learning/mra_exp2_split_c_1000_task_ids.txt"),
}

LAYERS = [31, 43, 55]
ALPHAS = get_default_alphas(10)
OUTPUT_DIR = Path("results/experiments/mra_exp3/probes")


def load_split_task_ids(split: str) -> set[str]:
    with open(SPLIT_TASK_ID_FILES[split]) as f:
        return {line.strip() for line in f if line.strip()}


def get_run_dir(persona: str, split: str) -> Path:
    results_dir, sys_hash = PERSONA_RUNS[persona]
    n = {"a": 1000, "b": 500, "c": 1000}[split]
    exp_id = "mra_exp2"  # both exp2 and exp3 use the same split naming
    prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"
    suffix = f"{exp_id}_split_{split}_{n}_task_ids"
    dirname = f"{prefix}_{sys_hash}_{suffix}" if sys_hash else f"{prefix}_{suffix}"
    return results_dir / dirname


def load_persona_split_data(persona: str, split: str, layer: int):
    run_dir = get_run_dir(persona, split)
    scores = load_thurstonian_scores(run_dir)
    task_ids = sorted(load_split_task_ids(split) & set(scores.keys()))
    X, y, matched_ids = load_probe_data(
        ACTIVATION_PATHS[persona], scores, task_ids, layer
    )
    return X, y, matched_ids


def load_persona_train_data(persona: str, layer: int):
    X_a, y_a, ids_a = load_persona_split_data(persona, "a", layer)
    X_c, y_c, ids_c = load_persona_split_data(persona, "c", layer)
    X = np.concatenate([X_a, X_c])
    y = np.concatenate([y_a, y_c])
    ids = list(ids_a) + list(ids_c)
    return X, y, ids


def train_probe_with_alpha_selection(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    best_alpha = None
    best_r2 = -np.inf
    for alpha in ALPHAS:
        probe = Ridge(alpha=alpha)
        probe.fit(X_train_s, y_train)
        y_pred = probe.predict(X_val_s)
        r2 = 1 - np.sum((y_val - y_pred)**2) / np.sum((y_val - np.mean(y_val))**2)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    probe = Ridge(alpha=best_alpha)
    probe.fit(X_train_s, y_train)
    return probe, scaler, best_alpha, best_r2


def evaluate_probe(probe, scaler, X_eval, y_eval):
    X_s = scaler.transform(X_eval)
    y_pred = probe.predict(X_s)

    r, _ = pearsonr(y_eval, y_pred)
    r2 = 1 - np.sum((y_eval - y_pred)**2) / np.sum((y_eval - np.mean(y_eval))**2)

    y_pred_adj = y_pred - np.mean(y_pred) + np.mean(y_eval)
    r2_adj = 1 - np.sum((y_eval - y_pred_adj)**2) / np.sum((y_eval - np.mean(y_eval))**2)

    return {
        "r2": float(r2),
        "r2_adjusted": float(r2_adj),
        "pearson_r": float(r),
        "n_samples": len(y_eval),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    rng = np.random.RandomState(42)

    for layer in LAYERS:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")
        layer_key = f"L{layer}"

        # Train noprompt probe on split_a+c
        X_train, y_train, train_ids = load_persona_train_data("noprompt", layer)
        print(f"  noprompt training: {len(train_ids)} tasks")

        # Split noprompt split_b into sweep/eval halves
        X_b, y_b, ids_b = load_persona_split_data("noprompt", "b", layer)
        n = len(y_b)
        idx = rng.permutation(n)
        half = n // 2
        X_val, y_val = X_b[idx[:half]], y_b[idx[:half]]
        X_np_eval, y_np_eval = X_b[idx[half:]], y_b[idx[half:]]

        probe, scaler, best_alpha, val_r2 = train_probe_with_alpha_selection(
            X_train, y_train, X_val, y_val
        )
        print(f"  alpha={best_alpha:.1f}, sweep R²={val_r2:.4f}")

        # Evaluate on noprompt eval half
        np_metrics = evaluate_probe(probe, scaler, X_np_eval, y_np_eval)
        print(f"  noprompt -> noprompt: r={np_metrics['pearson_r']:.4f}, "
              f"R²_adj={np_metrics['r2_adjusted']:.4f}")

        layer_results = {
            "probe": {"best_alpha": float(best_alpha), "val_r2": float(val_r2), "n_train": len(train_ids)},
            "eval": {"noprompt": np_metrics},
        }

        # Evaluate on each other persona
        for persona in ALL_PERSONAS:
            if persona == "noprompt":
                continue
            try:
                X_b_p, y_b_p, ids_b_p = load_persona_split_data(persona, "b", layer)
                n_p = len(y_b_p)
                idx_p = rng.permutation(n_p)
                half_p = n_p // 2
                X_p_eval, y_p_eval = X_b_p[idx_p[half_p:]], y_b_p[idx_p[half_p:]]

                metrics = evaluate_probe(probe, scaler, X_p_eval, y_p_eval)
                layer_results["eval"][persona] = metrics
                print(f"  noprompt -> {persona}: r={metrics['pearson_r']:.4f}, "
                      f"R²_adj={metrics['r2_adjusted']:.4f}")
            except Exception as e:
                print(f"  noprompt -> {persona}: FAILED ({e})")

        results[layer_key] = layer_results

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: noprompt probe → each persona (Pearson r)")
    print(f"{'='*60}")
    header = f"{'Persona':<15}"
    for layer in LAYERS:
        header += f"  L{layer:>2}"
    print(header)
    print("-" * len(header))

    for persona in ALL_PERSONAS:
        row = f"{persona:<15}"
        for layer in LAYERS:
            layer_key = f"L{layer}"
            if persona in results[layer_key]["eval"]:
                r = results[layer_key]["eval"][persona]["pearson_r"]
                row += f"  {r:.3f}"
            else:
                row += "    N/A"
        print(row)

    output_path = OUTPUT_DIR / "noprompt_probe_on_evil.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
