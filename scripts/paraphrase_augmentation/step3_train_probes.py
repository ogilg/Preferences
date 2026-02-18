"""Step 3: Train probes in three conditions and evaluate on held-out test set.

Conditions:
- Baseline: 80 original tasks
- Augmented: 80 originals + 80 paraphrases (paraphrases inherit parent utility)
- Paraphrase-only: 80 paraphrases, eval on original test tasks
"""

import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from src.probes.core.linear_probe import train_and_evaluate
from src.probes.data_loading import load_thurstonian_scores

load_dotenv()

PARAPHRASES_FILE = Path("experiments/probe_science/paraphrase_augmentation/paraphrases.json")
EXISTING_ACTIVATIONS = Path("activations/gemma_3_27b/activations_prompt_last.npz")
PARA_ACTIVATIONS = Path("experiments/probe_science/paraphrase_augmentation/paraphrase_activations.npz")
RUN_DIR = Path(
    "results/experiments/gemma3_3k_run2/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0"
)
OUTPUT_FILE = Path("experiments/probe_science/paraphrase_augmentation/probe_results.json")

LAYER = 31
CV_FOLDS = 5
TEST_FRACTION = 0.2
SEED = 42
N_ALPHA_SWEEP = 7


def compute_pairwise_accuracy(predicted: np.ndarray, true: np.ndarray) -> float:
    """Fraction of pairs where predicted ordering matches true ordering."""
    n = len(predicted)
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if true[i] == true[j]:
                continue
            total += 1
            if (predicted[i] - predicted[j]) * (true[i] - true[j]) > 0:
                correct += 1
    return correct / total if total > 0 else 0.0


def main():
    # Load paraphrases (for task IDs and utilities)
    with open(PARAPHRASES_FILE) as f:
        paraphrases = json.load(f)
    task_ids_100 = [p["task_id"] for p in paraphrases]
    utilities_100 = {p["task_id"]: p["utility"] for p in paraphrases}
    print(f"Loaded {len(paraphrases)} paraphrase entries")

    # Load existing activations (originals)
    existing = np.load(EXISTING_ACTIVATIONS, allow_pickle=True)
    existing_task_ids = list(existing["task_ids"])
    existing_acts = existing[f"layer_{LAYER}"]
    existing_id_to_idx = {tid: i for i, tid in enumerate(existing_task_ids)}

    # Load paraphrase activations
    para_data = np.load(PARA_ACTIVATIONS, allow_pickle=True)
    para_task_ids = list(para_data["task_ids"])
    para_acts = para_data[f"layer_{LAYER}"]
    para_id_to_idx = {tid: i for i, tid in enumerate(para_task_ids)}

    print(f"Existing activations: {existing_acts.shape}")
    print(f"Paraphrase activations: {para_acts.shape}")

    # Verify all 100 original tasks have activations
    available_ids = [tid for tid in task_ids_100 if tid in existing_id_to_idx]
    print(f"Tasks with both utility and activations: {len(available_ids)}")

    # Train/test split (stratified by utility quartile)
    rng = np.random.default_rng(SEED)
    n_test = int(len(available_ids) * TEST_FRACTION)

    # Sort by utility for stratified split
    sorted_ids = sorted(available_ids, key=lambda tid: utilities_100[tid])
    # Split each quartile proportionally
    quartile_size = len(sorted_ids) // 4
    test_ids = []
    train_ids = []
    for q in range(4):
        start = q * quartile_size
        end = (q + 1) * quartile_size if q < 3 else len(sorted_ids)
        q_ids = sorted_ids[start:end]
        n_test_q = max(1, int(len(q_ids) * TEST_FRACTION))
        q_test = list(rng.choice(len(q_ids), size=n_test_q, replace=False))
        for j, tid in enumerate(q_ids):
            if j in q_test:
                test_ids.append(tid)
            else:
                train_ids.append(tid)

    print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")

    # Prepare data
    def get_original_acts_and_labels(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        indices = [existing_id_to_idx[tid] for tid in ids]
        acts = existing_acts[indices]
        labels = np.array([utilities_100[tid] for tid in ids])
        return acts, labels

    def get_para_acts_and_labels(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        para_ids = [f"{tid}_para" for tid in ids]
        indices = [para_id_to_idx[pid] for pid in para_ids]
        acts = para_acts[indices]
        labels = np.array([utilities_100[tid] for tid in ids])
        return acts, labels

    # Test set (always original tasks)
    test_acts, test_labels = get_original_acts_and_labels(test_ids)

    # Train sets
    train_orig_acts, train_orig_labels = get_original_acts_and_labels(train_ids)
    train_para_acts, train_para_labels = get_para_acts_and_labels(train_ids)

    alphas = np.logspace(0, 6, N_ALPHA_SWEEP)

    results = {}

    # --- Condition 1: Baseline (originals only) ---
    print("\n=== Baseline (originals only) ===")
    probe_base, res_base, sweep_base = train_and_evaluate(
        train_orig_acts, train_orig_labels, CV_FOLDS, alphas=alphas,
    )
    pred_base = probe_base.predict(test_acts)
    test_r2_base = r2_score(test_labels, pred_base)
    test_pearson_base = pearsonr(test_labels, pred_base)[0]
    test_pairwise_base = compute_pairwise_accuracy(pred_base, test_labels)
    print(f"  CV R²: {res_base['cv_r2_mean']:.4f} ± {res_base['cv_r2_std']:.4f}")
    print(f"  Test R²: {test_r2_base:.4f}")
    print(f"  Test Pearson r: {test_pearson_base:.4f}")
    print(f"  Test Pairwise Acc: {test_pairwise_base:.4f}")
    results["baseline"] = {
        "train_n": len(train_ids),
        "test_n": len(test_ids),
        "best_alpha": res_base["best_alpha"],
        "cv_r2_mean": res_base["cv_r2_mean"],
        "cv_r2_std": res_base["cv_r2_std"],
        "test_r2": float(test_r2_base),
        "test_pearson_r": float(test_pearson_base),
        "test_pairwise_acc": float(test_pairwise_base),
        "sweep": sweep_base,
    }

    # --- Condition 2: Augmented (originals + paraphrases) ---
    print("\n=== Augmented (originals + paraphrases) ===")
    aug_acts = np.concatenate([train_orig_acts, train_para_acts], axis=0)
    aug_labels = np.concatenate([train_orig_labels, train_para_labels], axis=0)
    probe_aug, res_aug, sweep_aug = train_and_evaluate(
        aug_acts, aug_labels, CV_FOLDS, alphas=alphas,
    )
    pred_aug = probe_aug.predict(test_acts)
    test_r2_aug = r2_score(test_labels, pred_aug)
    test_pearson_aug = pearsonr(test_labels, pred_aug)[0]
    test_pairwise_aug = compute_pairwise_accuracy(pred_aug, test_labels)
    print(f"  CV R²: {res_aug['cv_r2_mean']:.4f} ± {res_aug['cv_r2_std']:.4f}")
    print(f"  Test R²: {test_r2_aug:.4f}")
    print(f"  Test Pearson r: {test_pearson_aug:.4f}")
    print(f"  Test Pairwise Acc: {test_pairwise_aug:.4f}")
    results["augmented"] = {
        "train_n": len(aug_labels),
        "test_n": len(test_ids),
        "best_alpha": res_aug["best_alpha"],
        "cv_r2_mean": res_aug["cv_r2_mean"],
        "cv_r2_std": res_aug["cv_r2_std"],
        "test_r2": float(test_r2_aug),
        "test_pearson_r": float(test_pearson_aug),
        "test_pairwise_acc": float(test_pairwise_aug),
        "sweep": sweep_aug,
    }

    # --- Condition 3: Paraphrase-only ---
    print("\n=== Paraphrase-only ===")
    probe_para, res_para, sweep_para = train_and_evaluate(
        train_para_acts, train_para_labels, CV_FOLDS, alphas=alphas,
    )
    pred_para = probe_para.predict(test_acts)
    test_r2_para = r2_score(test_labels, pred_para)
    test_pearson_para = pearsonr(test_labels, pred_para)[0]
    test_pairwise_para = compute_pairwise_accuracy(pred_para, test_labels)
    print(f"  CV R²: {res_para['cv_r2_mean']:.4f} ± {res_para['cv_r2_std']:.4f}")
    print(f"  Test R²: {test_r2_para:.4f}")
    print(f"  Test Pearson r: {test_pearson_para:.4f}")
    print(f"  Test Pairwise Acc: {test_pairwise_para:.4f}")
    results["paraphrase_only"] = {
        "train_n": len(train_ids),
        "test_n": len(test_ids),
        "best_alpha": res_para["best_alpha"],
        "cv_r2_mean": res_para["cv_r2_mean"],
        "cv_r2_std": res_para["cv_r2_std"],
        "test_r2": float(test_r2_para),
        "test_pearson_r": float(test_pearson_para),
        "test_pairwise_acc": float(test_pairwise_para),
        "sweep": sweep_para,
    }

    # --- Probe similarity ---
    weights_base = np.concatenate([probe_base.coef_.flatten(), [probe_base.intercept_]])
    weights_aug = np.concatenate([probe_aug.coef_.flatten(), [probe_aug.intercept_]])
    weights_para = np.concatenate([probe_para.coef_.flatten(), [probe_para.intercept_]])

    cos_base_aug = float(np.dot(weights_base, weights_aug) / (np.linalg.norm(weights_base) * np.linalg.norm(weights_aug)))
    cos_base_para = float(np.dot(weights_base, weights_para) / (np.linalg.norm(weights_base) * np.linalg.norm(weights_para)))
    cos_aug_para = float(np.dot(weights_aug, weights_para) / (np.linalg.norm(weights_aug) * np.linalg.norm(weights_para)))

    results["probe_similarity"] = {
        "baseline_vs_augmented": cos_base_aug,
        "baseline_vs_paraphrase_only": cos_base_para,
        "augmented_vs_paraphrase_only": cos_aug_para,
    }

    print(f"\n=== Probe Similarities ===")
    print(f"  Baseline vs Augmented: {cos_base_aug:.4f}")
    print(f"  Baseline vs Paraphrase-only: {cos_base_para:.4f}")
    print(f"  Augmented vs Paraphrase-only: {cos_aug_para:.4f}")

    # --- Summary ---
    print("\n=== SUMMARY ===")
    print(f"{'Condition':<20} {'Train N':<10} {'CV R²':<12} {'Test R²':<12} {'Test Pair Acc':<15} {'Alpha':<10}")
    for name in ["baseline", "augmented", "paraphrase_only"]:
        r = results[name]
        print(f"{name:<20} {r['train_n']:<10} {r['cv_r2_mean']:.4f}±{r['cv_r2_std']:.4f}  {r['test_r2']:<12.4f} {r['test_pairwise_acc']:<15.4f} {r['best_alpha']:<10.1f}")

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")

    # Metadata
    results["metadata"] = {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "layer": LAYER,
        "cv_folds": CV_FOLDS,
        "seed": SEED,
        "n_alpha_sweep": N_ALPHA_SWEEP,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
