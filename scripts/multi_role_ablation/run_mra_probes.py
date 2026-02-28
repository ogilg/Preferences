"""Multi-Role Ablation: probe training and cross-persona evaluation.

Phase 1: Train per-persona probes, cross-evaluate on all personas.
Phase 2: Training ablation — data quantity vs persona diversity.
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

# === Configuration ===

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
OUTPUT_DIR = Path("results/experiments/mra_exp2/probes")


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
    """Load activations and Thurstonian scores for a persona/split/layer combo."""
    run_dir = get_run_dir(persona, split)
    scores = load_thurstonian_scores(run_dir)
    task_ids = sorted(load_split_task_ids(split) & set(scores.keys()))
    X, y, matched_ids = load_probe_data(
        ACTIVATION_PATHS[persona], scores, task_ids, layer
    )
    return X, y, matched_ids


def train_probe_with_alpha_selection(X_train, y_train, X_val, y_val):
    """Sweep alpha on validation set, return trained probe at best alpha."""
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

    # Store scaler params in probe for evaluation
    return probe, scaler, best_alpha, best_r2


def evaluate_probe(probe, scaler, X_eval, y_eval):
    """Evaluate a trained probe on new data."""
    X_s = scaler.transform(X_eval)
    y_pred = probe.predict(X_s)

    r, _ = pearsonr(y_eval, y_pred)
    r2 = 1 - np.sum((y_eval - y_pred)**2) / np.sum((y_eval - np.mean(y_eval))**2)

    # Mean-adjusted R² (accounts for different score distributions)
    y_pred_adj = y_pred - np.mean(y_pred) + np.mean(y_eval)
    r2_adj = 1 - np.sum((y_eval - y_pred_adj)**2) / np.sum((y_eval - np.mean(y_eval))**2)

    return {
        "r2": float(r2),
        "r2_adjusted": float(r2_adj),
        "pearson_r": float(r),
        "n_samples": len(y_eval),
    }


# === Phase 1: Per-persona probes + cross-evaluation ===

def run_phase1():
    print("=" * 60)
    print("PHASE 1: Per-persona probes + cross-evaluation")
    print("=" * 60)

    results = {}

    for layer in LAYERS:
        print(f"\n--- Layer {layer} ---")
        layer_key = f"L{layer}"
        results[layer_key] = {"cross_eval": {}, "probes": {}}

        # Load eval data (split_c) for all personas
        eval_data = {}
        for persona in PERSONAS:
            X, y, ids = load_persona_split_data(persona, "c", layer)
            eval_data[persona] = (X, y, ids)
            print(f"  Eval {persona}: {len(ids)} tasks")

        # Train per-persona probes on split_a, sweep alpha on split_b
        probes = {}
        for train_persona in PERSONAS:
            X_train, y_train, train_ids = load_persona_split_data(train_persona, "a", layer)
            X_val, y_val, val_ids = load_persona_split_data(train_persona, "b", layer)
            print(f"  Training {train_persona}: {len(train_ids)} train, {len(val_ids)} val")

            probe, scaler, best_alpha, val_r2 = train_probe_with_alpha_selection(
                X_train, y_train, X_val, y_val
            )
            probes[train_persona] = (probe, scaler)
            results[layer_key]["probes"][train_persona] = {
                "best_alpha": float(best_alpha),
                "val_r2": float(val_r2),
                "n_train": len(train_ids),
            }
            print(f"    alpha={best_alpha:.1f}, val R²={val_r2:.4f}")

            # Save probe weights
            weights = np.concatenate([probe.coef_, [probe.intercept_]])
            probe_path = OUTPUT_DIR / "probes" / f"{train_persona}_L{layer}.npy"
            probe_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(probe_path, weights)

        # Cross-evaluate
        cross_eval = {}
        for train_persona in PERSONAS:
            cross_eval[train_persona] = {}
            probe, scaler = probes[train_persona]
            for eval_persona in PERSONAS:
                X_eval, y_eval, _ = eval_data[eval_persona]
                metrics = evaluate_probe(probe, scaler, X_eval, y_eval)
                cross_eval[train_persona][eval_persona] = metrics
                marker = " <-- within" if train_persona == eval_persona else ""
                print(f"    {train_persona} -> {eval_persona}: R²={metrics['r2']:.4f}, "
                      f"R²_adj={metrics['r2_adjusted']:.4f}, r={metrics['pearson_r']:.4f}{marker}")

        results[layer_key]["cross_eval"] = cross_eval

    return results


# === Phase 2: Training ablation ===

def run_phase2():
    print("\n" + "=" * 60)
    print("PHASE 2: Training ablation (data quantity vs persona diversity)")
    print("=" * 60)

    results = {}

    for layer in LAYERS:
        print(f"\n--- Layer {layer} ---")
        layer_key = f"L{layer}"
        results[layer_key] = {"conditions": []}

        # Preload all data
        split_a_data = {}
        split_b_data = {}
        split_c_data = {}
        for persona in PERSONAS:
            split_a_data[persona] = load_persona_split_data(persona, "a", layer)
            split_b_data[persona] = load_persona_split_data(persona, "b", layer)
            split_c_data[persona] = load_persona_split_data(persona, "c", layer)

        # --- Condition A: 1 persona, 2000 tasks (split_a + split_c) ---
        print("\n  Condition A: 1 persona x 2000 tasks")
        for train_persona in PERSONAS:
            X_a, y_a, _ = split_a_data[train_persona]
            X_c, y_c, _ = split_c_data[train_persona]
            X_train = np.concatenate([X_a, X_c])
            y_train = np.concatenate([y_a, y_c])

            X_val, y_val, _ = split_b_data[train_persona]

            probe, scaler, best_alpha, val_r2 = train_probe_with_alpha_selection(
                X_train, y_train, X_val, y_val
            )

            # Evaluate on all OTHER personas' split_c
            for eval_persona in PERSONAS:
                if eval_persona == train_persona:
                    continue
                X_eval, y_eval, _ = split_c_data[eval_persona]
                metrics = evaluate_probe(probe, scaler, X_eval, y_eval)
                entry = {
                    "condition": "A_1persona_2000",
                    "n_train_personas": 1,
                    "train_personas": [train_persona],
                    "eval_persona": eval_persona,
                    "n_total_train": len(y_train),
                    "best_alpha": float(best_alpha),
                    **metrics,
                }
                results[layer_key]["conditions"].append(entry)
                print(f"    train={train_persona} -> eval={eval_persona}: "
                      f"R²={metrics['r2']:.4f}, R²_adj={metrics['r2_adjusted']:.4f}")

        # --- Condition B: 2 personas, 1000 each (split_a from each) ---
        print("\n  Condition B: 2 personas x 1000 tasks each")
        for p1, p2 in itertools.combinations(PERSONAS, 2):
            X_train = np.concatenate([split_a_data[p1][0], split_a_data[p2][0]])
            y_train = np.concatenate([split_a_data[p1][1], split_a_data[p2][1]])

            X_val = np.concatenate([split_b_data[p1][0], split_b_data[p2][0]])
            y_val = np.concatenate([split_b_data[p1][1], split_b_data[p2][1]])

            probe, scaler, best_alpha, val_r2 = train_probe_with_alpha_selection(
                X_train, y_train, X_val, y_val
            )

            # Evaluate on held-out personas
            held_out = [p for p in PERSONAS if p not in (p1, p2)]
            for eval_persona in held_out:
                X_eval, y_eval, _ = split_c_data[eval_persona]
                metrics = evaluate_probe(probe, scaler, X_eval, y_eval)
                entry = {
                    "condition": "B_2personas_1000each",
                    "n_train_personas": 2,
                    "train_personas": sorted([p1, p2]),
                    "eval_persona": eval_persona,
                    "n_total_train": len(y_train),
                    "best_alpha": float(best_alpha),
                    **metrics,
                }
                results[layer_key]["conditions"].append(entry)
                print(f"    train={p1}+{p2} -> eval={eval_persona}: "
                      f"R²={metrics['r2']:.4f}, R²_adj={metrics['r2_adjusted']:.4f}")

        # --- Condition C: 3 personas, ~667 each (subsample split_a) ---
        print("\n  Condition C: 3 personas x ~667 tasks each")
        rng = np.random.RandomState(42)
        for held_out in PERSONAS:
            train_personas = [p for p in PERSONAS if p != held_out]
            X_parts, y_parts = [], []
            val_X_parts, val_y_parts = [], []
            for p in train_personas:
                X_a, y_a, _ = split_a_data[p]
                # Subsample 667 from 1000
                indices = rng.choice(len(X_a), size=667, replace=False)
                X_parts.append(X_a[indices])
                y_parts.append(y_a[indices])
                val_X_parts.append(split_b_data[p][0])
                val_y_parts.append(split_b_data[p][1])

            X_train = np.concatenate(X_parts)
            y_train = np.concatenate(y_parts)
            X_val = np.concatenate(val_X_parts)
            y_val = np.concatenate(val_y_parts)

            probe, scaler, best_alpha, val_r2 = train_probe_with_alpha_selection(
                X_train, y_train, X_val, y_val
            )

            X_eval, y_eval, _ = split_c_data[held_out]
            metrics = evaluate_probe(probe, scaler, X_eval, y_eval)
            entry = {
                "condition": "C_3personas_667each",
                "n_train_personas": 3,
                "train_personas": sorted(train_personas),
                "eval_persona": held_out,
                "n_total_train": len(y_train),
                "best_alpha": float(best_alpha),
                **metrics,
            }
            results[layer_key]["conditions"].append(entry)
            print(f"    train={'+'.join(train_personas)} -> eval={held_out}: "
                  f"R²={metrics['r2']:.4f}, R²_adj={metrics['r2_adjusted']:.4f}")

        # --- Condition D: 4 personas combined (ceiling, split_a from all) ---
        print("\n  Condition D: 4 personas combined (ceiling)")
        X_train = np.concatenate([split_a_data[p][0] for p in PERSONAS])
        y_train = np.concatenate([split_a_data[p][1] for p in PERSONAS])
        X_val = np.concatenate([split_b_data[p][0] for p in PERSONAS])
        y_val = np.concatenate([split_b_data[p][1] for p in PERSONAS])

        probe, scaler, best_alpha, val_r2 = train_probe_with_alpha_selection(
            X_train, y_train, X_val, y_val
        )

        # Save combined probe
        weights = np.concatenate([probe.coef_, [probe.intercept_]])
        probe_path = OUTPUT_DIR / "probes" / f"all_4_L{layer}.npy"
        np.save(probe_path, weights)

        for eval_persona in PERSONAS:
            X_eval, y_eval, _ = split_c_data[eval_persona]
            metrics = evaluate_probe(probe, scaler, X_eval, y_eval)
            entry = {
                "condition": "D_4personas_combined",
                "n_train_personas": 4,
                "train_personas": sorted(PERSONAS),
                "eval_persona": eval_persona,
                "n_total_train": len(y_train),
                "best_alpha": float(best_alpha),
                **metrics,
            }
            results[layer_key]["conditions"].append(entry)
            print(f"    train=all4 -> eval={eval_persona}: "
                  f"R²={metrics['r2']:.4f}, R²_adj={metrics['r2_adjusted']:.4f}")

    return results


def main():
    print("Multi-Role Ablation: Probe Training")
    print(f"Personas: {PERSONAS}")
    print(f"Layers: {LAYERS}")
    print(f"Output: {OUTPUT_DIR}")

    phase1 = run_phase1()
    phase2 = run_phase2()

    all_results = {"phase1": phase1, "phase2": phase2}

    output_path = OUTPUT_DIR / "mra_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
