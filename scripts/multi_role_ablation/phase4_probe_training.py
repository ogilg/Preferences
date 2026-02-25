"""Phase 4: Train and evaluate probes for all persona combinations.

For each of 15 training conditions (all non-empty subsets of 4 personas):
- Train Ridge probe on concatenated train data from the training personas
- Sweep alpha on first 250 eval tasks per persona, select best
- Evaluate on final 250 eval tasks from ALL 4 personas
- Report Pearson r, R², pairwise choice accuracy

Outputs:
  experiments/probe_generalization/multi_role_ablation/probe_results.json
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

from src.probes.core.activations import load_probe_data
from src.probes.data_loading import load_thurstonian_scores

load_dotenv()

REPO = Path(__file__).parent.parent.parent
EXPERIMENT_DIR = REPO / "experiments/probe_generalization/multi_role_ablation"
LAYER = 31  # L31 = 0.5 × 62 layers
EVAL_ALPHA_SPLIT_SEED = 42
ALPHAS = np.logspace(1, 7, 13)  # Alpha range for ridge

# Persona config
PERSONAS = {
    1: {
        "name": "no_prompt",
        "run_dir": REPO / "results/experiments/mra_persona1_noprompt/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0",
        "activations_path": REPO / "activations/gemma_3_27b/activations_prompt_last.npz",
    },
    2: {
        "name": "villain",
        "run_dir": REPO / "results/experiments/mra_persona2_villain/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0",
        "activations_path": REPO / "activations/gemma_3_27b_villain/activations_prompt_last.npz",
    },
    3: {
        "name": "midwest",
        "run_dir": REPO / "results/experiments/mra_persona3_midwest/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0",
        "activations_path": REPO / "activations/gemma_3_27b_midwest/activations_prompt_last.npz",
    },
    4: {
        "name": "aesthete",
        "run_dir": REPO / "results/experiments/mra_persona4_aesthete/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0",
        "activations_path": REPO / "activations/gemma_3_27b_aesthete/activations_prompt_last.npz",
    },
}


def load_task_ids() -> tuple[list[str], list[str]]:
    train_ids = (EXPERIMENT_DIR / "task_ids_train.txt").read_text().strip().splitlines()
    eval_ids = (EXPERIMENT_DIR / "task_ids_eval.txt").read_text().strip().splitlines()
    return train_ids, eval_ids


def split_eval_ids(eval_ids: list[str]) -> tuple[list[str], list[str]]:
    """Split 500 eval IDs into 250 val (alpha selection) and 250 test."""
    rng = np.random.default_rng(EVAL_ALPHA_SPLIT_SEED)
    shuffled = list(eval_ids)
    rng.shuffle(shuffled)
    val_ids = shuffled[:250]
    test_ids = shuffled[250:]
    return val_ids, test_ids


def load_scores(persona_id: int) -> dict[str, float]:
    run_dir = PERSONAS[persona_id]["run_dir"]
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    return load_thurstonian_scores(run_dir)


def load_persona_data(persona_id: int, task_ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load activations and scores for a persona, filtered to task_ids."""
    scores = load_scores(persona_id)
    activations_path = PERSONAS[persona_id]["activations_path"]
    return load_probe_data(activations_path, scores, task_ids, LAYER)


def select_alpha(train_X: np.ndarray, train_y: np.ndarray, val_X: np.ndarray, val_y: np.ndarray) -> float:
    """Select best Ridge alpha based on validation Pearson r."""
    best_alpha = ALPHAS[0]
    best_r = -np.inf
    for alpha in ALPHAS:
        probe = Ridge(alpha=alpha)
        probe.fit(train_X, train_y)
        y_pred = probe.predict(val_X)
        if len(y_pred) >= 3:
            r, _ = pearsonr(val_y, y_pred)
            if r > best_r:
                best_r = r
                best_alpha = alpha
    return float(best_alpha)


def train_probe(train_X: np.ndarray, train_y: np.ndarray, alpha: float):
    """Train Ridge probe at given alpha."""
    probe = Ridge(alpha=alpha)
    probe.fit(train_X, train_y)
    return probe


def evaluate_probe(probe, eval_X: np.ndarray, eval_y: np.ndarray) -> dict:
    """Compute Pearson r and R² on eval data."""
    y_pred = probe.predict(eval_X)
    r, p = pearsonr(eval_y, y_pred)
    ss_res = np.sum((eval_y - y_pred) ** 2)
    ss_tot = np.sum((eval_y - np.mean(eval_y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"pearson_r": float(r), "r2": float(r2), "p_value": float(p), "n": len(eval_y)}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(a_norm @ b_norm)


def main() -> None:
    print("Loading task IDs...")
    train_ids, eval_ids = load_task_ids()
    val_ids, test_ids = split_eval_ids(eval_ids)
    print(f"Train: {len(train_ids)}, Val (alpha): {len(val_ids)}, Test: {len(test_ids)}")

    # Load data for all 4 personas
    print("\nLoading persona data...")
    train_data: dict[int, tuple] = {}  # persona_id -> (X_train, y_train, ids_train)
    val_data: dict[int, tuple] = {}
    test_data: dict[int, tuple] = {}

    for p_id in PERSONAS:
        print(f"  Persona {p_id} ({PERSONAS[p_id]['name']})...")
        try:
            X_train, y_train, ids_train = load_persona_data(p_id, train_ids)
            X_val, y_val, ids_val = load_persona_data(p_id, val_ids)
            X_test, y_test, ids_test = load_persona_data(p_id, test_ids)
            train_data[p_id] = (X_train, y_train, ids_train)
            val_data[p_id] = (X_val, y_val, ids_val)
            test_data[p_id] = (X_test, y_test, ids_test)
            print(f"    Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        except Exception as e:
            print(f"    ERROR loading persona {p_id}: {e}")
            return

    print("\nTraining and evaluating 15 probes...")

    # All non-empty subsets of {1, 2, 3, 4}
    all_persona_ids = [1, 2, 3, 4]
    training_conditions = []
    for size in range(1, 5):
        for combo in combinations(all_persona_ids, size):
            training_conditions.append(list(combo))

    results = []
    probe_weights: dict[str, np.ndarray] = {}

    for train_personas in training_conditions:
        condition_name = "+".join(PERSONAS[p]["name"] for p in train_personas)
        print(f"\n  Condition: {condition_name}")

        # Concatenate training data
        X_train_all = np.concatenate([train_data[p][0] for p in train_personas], axis=0)
        y_train_all = np.concatenate([train_data[p][1] for p in train_personas], axis=0)
        print(f"    Train size: {X_train_all.shape[0]}")

        # Alpha selection: use first eval persona's val set
        # Use the mean val Pearson r across all training personas' val sets
        best_alpha = None
        best_mean_r = -np.inf
        for alpha in ALPHAS:
            probe = Ridge(alpha=alpha)
            probe.fit(X_train_all, y_train_all)
            rs = []
            for p_id in train_personas:
                X_val, y_val, _ = val_data[p_id]
                y_pred = probe.predict(X_val)
                if len(y_pred) >= 3:
                    r, _ = pearsonr(y_val, y_pred)
                    rs.append(r)
            if rs:
                mean_r = float(np.mean(rs))
                if mean_r > best_mean_r:
                    best_mean_r = mean_r
                    best_alpha = alpha

        print(f"    Best alpha: {best_alpha:.0f} (val mean r: {best_mean_r:.3f})")

        # Train final probe at best alpha
        final_probe = Ridge(alpha=best_alpha)
        final_probe.fit(X_train_all, y_train_all)

        # Store weights
        weights = np.append(final_probe.coef_, final_probe.intercept_)
        probe_weights[condition_name] = weights

        # Evaluate on all 4 persona test sets
        eval_results = {}
        for p_id in all_persona_ids:
            X_test, y_test, _ = test_data[p_id]
            metrics = evaluate_probe(final_probe, X_test, y_test)
            eval_results[PERSONAS[p_id]["name"]] = metrics
            print(f"    Eval on {PERSONAS[p_id]['name']}: r={metrics['pearson_r']:.3f}, R²={metrics['r2']:.3f}")

        results.append({
            "train_personas": train_personas,
            "condition_name": condition_name,
            "n_train_total": int(X_train_all.shape[0]),
            "best_alpha": float(best_alpha),
            "val_mean_r": float(best_mean_r),
            "eval": eval_results,
        })

    # Compute cosine similarity matrix between probe directions
    print("\nComputing probe direction similarities...")
    condition_names = [r["condition_name"] for r in results]
    sim_matrix = np.zeros((len(condition_names), len(condition_names)))
    for i, n1 in enumerate(condition_names):
        for j, n2 in enumerate(condition_names):
            w1 = probe_weights[n1][:-1]  # exclude intercept
            w2 = probe_weights[n2][:-1]
            sim_matrix[i, j] = cosine_sim(w1, w2)

    # Compute utility correlation matrix (correlation between Thurstonian μ values)
    print("\nComputing utility correlation matrix...")
    scores_all: dict[int, dict[str, float]] = {}
    for p_id in all_persona_ids:
        scores_all[p_id] = load_scores(p_id)

    common_ids = set(scores_all[1].keys())
    for p_id in all_persona_ids:
        common_ids &= set(scores_all[p_id].keys())
    common_ids = sorted(common_ids)
    print(f"  Common task IDs for utility correlation: {len(common_ids)}")

    utility_matrix = np.zeros((4, 4))
    for i, p1 in enumerate(all_persona_ids):
        for j, p2 in enumerate(all_persona_ids):
            scores_p1 = np.array([scores_all[p1][tid] for tid in common_ids])
            scores_p2 = np.array([scores_all[p2][tid] for tid in common_ids])
            r, _ = pearsonr(scores_p1, scores_p2)
            utility_matrix[i, j] = float(r)

    # Save results
    output = {
        "conditions": results,
        "probe_cosine_similarity": {
            "condition_names": condition_names,
            "matrix": sim_matrix.tolist(),
        },
        "utility_correlation": {
            "persona_names": [PERSONAS[p]["name"] for p in all_persona_ids],
            "n_common_tasks": len(common_ids),
            "matrix": utility_matrix.tolist(),
        },
        "layer": LAYER,
        "eval_alpha_split_seed": EVAL_ALPHA_SPLIT_SEED,
        "n_train_per_persona": len(train_ids),
        "n_val_per_persona": len(val_ids),
        "n_test_per_persona": len(test_ids),
    }

    out_path = EXPERIMENT_DIR / "probe_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Print summary table
    print("\n=== RESULTS SUMMARY ===")
    print(f"{'Condition':<30} | {'n_train':>7} | " + " | ".join(f"{PERSONAS[p]['name']:>10}" for p in all_persona_ids))
    print("-" * 90)
    for r in results:
        row = f"{r['condition_name']:<30} | {r['n_train_total']:>7} | "
        row += " | ".join(f"{r['eval'][PERSONAS[p]['name']]['pearson_r']:>10.3f}" for p in all_persona_ids)
        print(row)

    print("\n=== UTILITY CORRELATIONS ===")
    persona_names = [PERSONAS[p]["name"] for p in all_persona_ids]
    print("      " + "  ".join(f"{n:>10}" for n in persona_names))
    for i, n in enumerate(persona_names):
        row = f"{n:>8}: " + "  ".join(f"{utility_matrix[i,j]:>10.3f}" for j in range(4))
        print(row)


if __name__ == "__main__":
    main()
