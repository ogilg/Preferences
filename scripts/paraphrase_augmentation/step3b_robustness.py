"""Step 3b: Robustness checks — multiple random seeds for train/test split."""

import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from src.probes.core.linear_probe import train_and_evaluate

load_dotenv()

PARAPHRASES_FILE = Path("experiments/probe_science/paraphrase_augmentation/paraphrases.json")
EXISTING_ACTIVATIONS = Path("activations/gemma_3_27b/activations_prompt_last.npz")
PARA_ACTIVATIONS = Path("experiments/probe_science/paraphrase_augmentation/paraphrase_activations.npz")
OUTPUT_FILE = Path("experiments/probe_science/paraphrase_augmentation/probe_results_robustness.json")

LAYER = 31
CV_FOLDS = 5
TEST_FRACTION = 0.2
N_ALPHA_SWEEP = 9
SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 8888, 9999, 7777]


def compute_pairwise_accuracy(predicted: np.ndarray, true: np.ndarray) -> float:
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


def run_one_seed(
    seed: int,
    available_ids: list[str],
    utilities: dict[str, float],
    existing_acts: np.ndarray,
    existing_id_to_idx: dict[str, int],
    para_acts: np.ndarray,
    para_id_to_idx: dict[str, int],
) -> dict:
    rng = np.random.default_rng(seed)
    alphas = np.logspace(0, 7, N_ALPHA_SWEEP)

    # Stratified split
    sorted_ids = sorted(available_ids, key=lambda tid: utilities[tid])
    quartile_size = len(sorted_ids) // 4
    test_ids = []
    train_ids = []
    for q in range(4):
        start = q * quartile_size
        end = (q + 1) * quartile_size if q < 3 else len(sorted_ids)
        q_ids = sorted_ids[start:end]
        n_test_q = max(1, int(len(q_ids) * TEST_FRACTION))
        q_test = set(rng.choice(len(q_ids), size=n_test_q, replace=False))
        for j, tid in enumerate(q_ids):
            if j in q_test:
                test_ids.append(tid)
            else:
                train_ids.append(tid)

    def get_orig(ids):
        indices = [existing_id_to_idx[tid] for tid in ids]
        return existing_acts[indices], np.array([utilities[tid] for tid in ids])

    def get_para(ids):
        indices = [para_id_to_idx[f"{tid}_para"] for tid in ids]
        return para_acts[indices], np.array([utilities[tid] for tid in ids])

    test_acts, test_labels = get_orig(test_ids)
    train_orig_acts, train_orig_labels = get_orig(train_ids)
    train_para_acts, train_para_labels = get_para(train_ids)

    results = {}
    for name, train_X, train_y in [
        ("baseline", train_orig_acts, train_orig_labels),
        ("augmented", np.concatenate([train_orig_acts, train_para_acts]), np.concatenate([train_orig_labels, train_para_labels])),
        ("paraphrase_only", train_para_acts, train_para_labels),
    ]:
        probe, res, _ = train_and_evaluate(train_X, train_y, CV_FOLDS, alphas=alphas)
        pred = probe.predict(test_acts)
        results[name] = {
            "cv_r2": res["cv_r2_mean"],
            "test_r2": float(r2_score(test_labels, pred)),
            "test_pearson": float(pearsonr(test_labels, pred)[0]),
            "test_pairwise_acc": compute_pairwise_accuracy(pred, test_labels),
            "best_alpha": res["best_alpha"],
            "train_n": len(train_y),
            "test_n": len(test_ids),
        }

    return results


def main():
    with open(PARAPHRASES_FILE) as f:
        paraphrases = json.load(f)
    utilities = {p["task_id"]: p["utility"] for p in paraphrases}
    task_ids_100 = list(utilities.keys())

    existing = np.load(EXISTING_ACTIVATIONS, allow_pickle=True)
    existing_task_ids = list(existing["task_ids"])
    existing_acts = existing[f"layer_{LAYER}"]
    existing_id_to_idx = {tid: i for i, tid in enumerate(existing_task_ids)}

    para_data = np.load(PARA_ACTIVATIONS, allow_pickle=True)
    para_task_ids = list(para_data["task_ids"])
    para_acts = para_data[f"layer_{LAYER}"]
    para_id_to_idx = {tid: i for i, tid in enumerate(para_task_ids)}

    available_ids = [tid for tid in task_ids_100 if tid in existing_id_to_idx]

    all_results = {}
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        result = run_one_seed(seed, available_ids, utilities, existing_acts, existing_id_to_idx, para_acts, para_id_to_idx)
        all_results[seed] = result
        for name in ["baseline", "augmented", "paraphrase_only"]:
            r = result[name]
            print(f"  {name:<20} CV R²={r['cv_r2']:.4f}  Test R²={r['test_r2']:.4f}  PairAcc={r['test_pairwise_acc']:.4f}  alpha={r['best_alpha']:.0f}")

    # Aggregate
    print("\n=== AGGREGATE ACROSS SEEDS ===")
    print(f"{'Condition':<20} {'Test R² (mean±std)':<25} {'Pair Acc (mean±std)':<25} {'CV R² (mean±std)':<25}")
    for name in ["baseline", "augmented", "paraphrase_only"]:
        test_r2s = [all_results[s][name]["test_r2"] for s in SEEDS]
        pair_accs = [all_results[s][name]["test_pairwise_acc"] for s in SEEDS]
        cv_r2s = [all_results[s][name]["cv_r2"] for s in SEEDS]
        print(f"{name:<20} {np.mean(test_r2s):.4f} ± {np.std(test_r2s):.4f}      {np.mean(pair_accs):.4f} ± {np.std(pair_accs):.4f}      {np.mean(cv_r2s):.4f} ± {np.std(cv_r2s):.4f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"seeds": {str(s): r for s, r in all_results.items()}, "seed_list": SEEDS}, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
