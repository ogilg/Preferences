"""MRA v2: train on 2000 (split_a + split_c), eval on 500 (split_b).

Phase 1 only: per-persona probes + cross-evaluation.
Uses half of split_b for alpha sweep, half for eval.
"""

import json
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.probes.core.activations import load_probe_data
from src.probes.core.linear_probe import get_default_alphas
from src.probes.data_loading import load_thurstonian_scores

PERSONAS = ["noprompt", "villain", "aesthete", "midwest"]

ACTIVATION_PATHS = {
    "noprompt": Path("activations/gemma_3_27b/activations_prompt_last.npz"),
    "villain": Path("activations/gemma_3_27b_villain/activations_prompt_last.npz"),
    "midwest": Path("activations/gemma_3_27b_midwest/activations_prompt_last.npz"),
    "aesthete": Path("activations/gemma_3_27b_aesthete/activations_prompt_last.npz"),
}

SYS_HASHES = {
    "noprompt": "",
    "villain": "syse8f24ac6",
    "aesthete": "sys021d8ca1",
    "midwest": "sys5d504504",
}

SPLIT_TASK_ID_FILES = {
    "a": Path("configs/measurement/active_learning/mra_exp2_split_a_1000_task_ids.txt"),
    "b": Path("configs/measurement/active_learning/mra_exp2_split_b_500_task_ids.txt"),
    "c": Path("configs/measurement/active_learning/mra_exp2_split_c_1000_task_ids.txt"),
}

LAYERS = [31, 43, 55]
ALPHAS = get_default_alphas(10)
OUTPUT_DIR = Path("results/experiments/mra_exp2/probes_v2")


def load_split_task_ids(split: str) -> set[str]:
    with open(SPLIT_TASK_ID_FILES[split]) as f:
        return {line.strip() for line in f if line.strip()}


def get_run_dir(persona: str, split: str) -> Path:
    n = {"a": 1000, "b": 500, "c": 1000}[split]
    sys = SYS_HASHES[persona]
    prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"
    suffix = f"mra_exp2_split_{split}_{n}_task_ids"
    dirname = f"{prefix}_{sys}_{suffix}" if sys else f"{prefix}_{suffix}"
    return Path("results/experiments/mra_exp2/pre_task_active_learning") / dirname


def load_persona_split_data(persona: str, split: str, layer: int):
    run_dir = get_run_dir(persona, split)
    scores = load_thurstonian_scores(run_dir)
    task_ids = sorted(load_split_task_ids(split) & set(scores.keys()))
    X, y, matched_ids = load_probe_data(
        ACTIVATION_PATHS[persona], scores, task_ids, layer
    )
    return X, y, matched_ids


def load_persona_train_data(persona: str, layer: int):
    """Load split_a + split_c as training data."""
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


def run_phase1():
    """Train on split_a+c (2000), sweep alpha on half of split_b (250), eval on other half (250)."""
    print("=" * 60)
    print("PHASE 1: train=2000 (a+c), eval=500 (b)")
    print("  Alpha sweep on first 250 of split_b, eval on last 250")
    print("=" * 60)

    results = {}
    rng = np.random.RandomState(42)

    for layer in LAYERS:
        print(f"\n--- Layer {layer} ---")
        layer_key = f"L{layer}"
        results[layer_key] = {"cross_eval": {}, "probes": {}}

        # Load split_b for all personas, split into sweep/eval halves
        eval_data = {}
        sweep_data = {}
        for persona in PERSONAS:
            X_b, y_b, ids_b = load_persona_split_data(persona, "b", layer)
            n = len(y_b)
            idx = rng.permutation(n)
            half = n // 2
            sweep_idx, eval_idx = idx[:half], idx[half:]
            sweep_data[persona] = (X_b[sweep_idx], y_b[sweep_idx])
            eval_data[persona] = (X_b[eval_idx], y_b[eval_idx])
            print(f"  {persona}: {half} sweep, {n - half} eval")

        probes = {}
        for train_persona in PERSONAS:
            X_train, y_train, train_ids = load_persona_train_data(train_persona, layer)
            X_val, y_val = sweep_data[train_persona]
            print(f"  Training {train_persona}: {len(train_ids)} train, {len(y_val)} sweep")

            probe, scaler, best_alpha, val_r2 = train_probe_with_alpha_selection(
                X_train, y_train, X_val, y_val
            )
            probes[train_persona] = (probe, scaler)
            results[layer_key]["probes"][train_persona] = {
                "best_alpha": float(best_alpha),
                "val_r2": float(val_r2),
                "n_train": len(train_ids),
            }
            print(f"    alpha={best_alpha:.1f}, sweep R²={val_r2:.4f}")

        # Cross-evaluate
        cross_eval = {}
        for train_persona in PERSONAS:
            cross_eval[train_persona] = {}
            probe, scaler = probes[train_persona]
            for eval_persona in PERSONAS:
                X_eval, y_eval = eval_data[eval_persona]
                metrics = evaluate_probe(probe, scaler, X_eval, y_eval)
                cross_eval[train_persona][eval_persona] = metrics
                marker = " <-- within" if train_persona == eval_persona else ""
                print(f"    {train_persona} -> {eval_persona}: "
                      f"r={metrics['pearson_r']:.4f}, R²_adj={metrics['r2_adjusted']:.4f}{marker}")

        results[layer_key]["cross_eval"] = cross_eval

    return results


def run_phase2():
    """Diversity ablation: vary N training personas at fixed total data.

    Always eval on half of split_b (250 tasks), sweep on other half.
    Training draws from split_a + split_c (2000 tasks per persona available).

    Conditions:
      A: 1 persona x 2000 tasks (full split_a + split_c)
      B: 2 personas x 1000 tasks each (subsample split_a + split_c)
      C: 3 personas x 667 tasks each
      D: 4 personas x 500 tasks each (matched 2000 total)
      E: 4 personas x 2000 tasks each (ceiling, 8000 total)
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Diversity ablation (eval on split_b)")
    print("=" * 60)

    results = {}
    rng = np.random.RandomState(42)

    for layer in LAYERS:
        print(f"\n--- Layer {layer} ---")
        layer_key = f"L{layer}"
        results[layer_key] = {"conditions": []}

        # Preload all data
        train_pool = {}
        for persona in PERSONAS:
            X_train, y_train, ids_train = load_persona_train_data(persona, layer)
            train_pool[persona] = (X_train, y_train, ids_train)
            print(f"  {persona} train pool: {len(ids_train)} tasks")

        # Split_b into sweep/eval halves (same split as phase 1)
        rng_split = np.random.RandomState(42)
        sweep_data = {}
        eval_data = {}
        for persona in PERSONAS:
            X_b, y_b, ids_b = load_persona_split_data(persona, "b", layer)
            n = len(y_b)
            idx = rng_split.permutation(n)
            half = n // 2
            sweep_data[persona] = (X_b[idx[:half]], y_b[idx[:half]])
            eval_data[persona] = (X_b[idx[half:]], y_b[idx[half:]])

        def subsample(X, y, n, rng):
            idx = rng.choice(len(X), size=n, replace=False)
            return X[idx], y[idx]

        def run_condition(cond_name, train_personas, tasks_per_persona, eval_persona):
            X_parts, y_parts = [], []
            for p in train_personas:
                X_full, y_full, _ = train_pool[p]
                if tasks_per_persona < len(y_full):
                    X_sub, y_sub = subsample(X_full, y_full, tasks_per_persona, rng)
                else:
                    X_sub, y_sub = X_full, y_full
                X_parts.append(X_sub)
                y_parts.append(y_sub)

            X_train = np.concatenate(X_parts)
            y_train = np.concatenate(y_parts)

            # Sweep alpha on training personas' sweep data
            X_val_parts = [sweep_data[p][0] for p in train_personas]
            y_val_parts = [sweep_data[p][1] for p in train_personas]
            X_val = np.concatenate(X_val_parts)
            y_val = np.concatenate(y_val_parts)

            probe, scaler, best_alpha, val_r2 = train_probe_with_alpha_selection(
                X_train, y_train, X_val, y_val
            )

            X_eval, y_eval = eval_data[eval_persona]
            metrics = evaluate_probe(probe, scaler, X_eval, y_eval)
            return {
                "condition": cond_name,
                "n_train_personas": len(train_personas),
                "train_personas": sorted(train_personas),
                "eval_persona": eval_persona,
                "tasks_per_persona": tasks_per_persona,
                "n_total_train": len(y_train),
                "best_alpha": float(best_alpha),
                **metrics,
            }

        # Condition A: 1 persona x 2000
        print("\n  Condition A: 1 persona x 2000")
        for train_p in PERSONAS:
            for eval_p in PERSONAS:
                if eval_p == train_p:
                    continue
                entry = run_condition("A_1x2000", [train_p], 2000, eval_p)
                results[layer_key]["conditions"].append(entry)
                print(f"    {train_p} -> {eval_p}: r={entry['pearson_r']:.4f}")

        # Condition B: 2 personas x 1000
        print("\n  Condition B: 2 personas x 1000")
        for p1, p2 in itertools.combinations(PERSONAS, 2):
            held_out = [p for p in PERSONAS if p not in (p1, p2)]
            for eval_p in held_out:
                entry = run_condition("B_2x1000", [p1, p2], 1000, eval_p)
                results[layer_key]["conditions"].append(entry)
                print(f"    {p1}+{p2} -> {eval_p}: r={entry['pearson_r']:.4f}")

        # Condition C: 3 personas x 667
        print("\n  Condition C: 3 personas x 667")
        for held_out in PERSONAS:
            train_ps = [p for p in PERSONAS if p != held_out]
            entry = run_condition("C_3x667", train_ps, 667, held_out)
            results[layer_key]["conditions"].append(entry)
            print(f"    {'+'.join(train_ps)} -> {held_out}: r={entry['pearson_r']:.4f}")

        # Condition D: 4 personas x 500 (matched total = 2000)
        print("\n  Condition D: 4 personas x 500 (matched 2000)")
        for eval_p in PERSONAS:
            entry = run_condition("D_4x500", PERSONAS, 500, eval_p)
            results[layer_key]["conditions"].append(entry)
            print(f"    all4 -> {eval_p}: r={entry['pearson_r']:.4f}")

        # Condition E: 4 personas x 2000 (ceiling = 8000)
        print("\n  Condition E: 4 personas x 2000 (ceiling)")
        for eval_p in PERSONAS:
            entry = run_condition("E_4x2000", PERSONAS, 2000, eval_p)
            results[layer_key]["conditions"].append(entry)
            print(f"    all4_full -> {eval_p}: r={entry['pearson_r']:.4f}")

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    phase1 = run_phase1()
    phase2 = run_phase2()

    results = {"phase1": phase1, "phase2": phase2}

    output_path = OUTPUT_DIR / "mra_results_v2.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
