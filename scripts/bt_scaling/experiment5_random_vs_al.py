"""Experiment 5: Random vs AL pair selection — is it only at the margin?

5a: Compare AL-ordered subsamples vs random subsamples at different data fractions.
5b: Test marginal value of AL-next vs random pairs at different base sizes.
"""
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*LinAlg.*")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.bt_scaling.data_loading import (
    aggregate_pairs,
    filter_pairs_by_tasks,
    get_task_folds,
    load_activations_layer,
    load_measurements,
    weighted_pairwise_accuracy,
)
from src.fitting.thurstonian_fitting.thurstonian import PairwiseData, fit_thurstonian
from src.task_data.task import OriginDataset, Task

MEASUREMENTS_JSON = Path("scripts/active_learning_calibration/measurements_fast.json")
RESULTS_DIR = Path("experiments/probe_science/bt_scaling")

# Hyperparameters from Experiment 1
RIDGE_ALPHA = 1374.0
BT_SCALED_LAMBDA = 0.193

# AL iteration structure: iteration 0 = d-regular (7500 pairs), iterations 1-8 = 2000 each
ITER_SIZES = [7500] + [2000] * 8  # 9 iterations, 23500 total


def bt_loss_and_grad(w, acts, idx_i, idx_j, wins_i, wins_j, total_weight, l2_lambda):
    task_scores = acts @ w
    logits = task_scores[idx_i] - task_scores[idx_j]
    loss_ij = wins_i * np.logaddexp(0, -logits) + wins_j * np.logaddexp(0, logits)
    loss = np.sum(loss_ij) / total_weight + 0.5 * l2_lambda * np.sum(w ** 2)
    sigmoid = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
    pair_grad = (sigmoid * (wins_i + wins_j) - wins_i) / total_weight
    n_tasks = len(acts)
    task_grad = (
        np.bincount(idx_i, weights=pair_grad, minlength=n_tasks)
        - np.bincount(idx_j, weights=pair_grad, minlength=n_tasks)
    )
    return loss, acts.T @ task_grad + l2_lambda * w


def fit_bt(acts, idx_i, idx_j, wins_i, wins_j, total_weight, l2_lambda):
    w0 = np.zeros(acts.shape[1])
    result = minimize(
        bt_loss_and_grad, w0, args=(acts, idx_i, idx_j, wins_i, wins_j, total_weight, l2_lambda),
        method="L-BFGS-B", jac=True, options={"maxiter": 500},
    )
    return result.x


def fit_thurstonian_fast(task_a_idx, task_b_idx, winner_idx, task_id_list, task_set):
    mask = np.isin(task_a_idx, list(task_set)) & np.isin(task_b_idx, list(task_set))
    a_filt = task_a_idx[mask]
    b_filt = task_b_idx[mask]
    w_filt = winner_idx[mask]

    task_idx_sorted = sorted(task_set)
    idx_to_new = {old: new for new, old in enumerate(task_idx_sorted)}
    tasks = [Task(prompt="", origin=OriginDataset.WILDCHAT, id=task_id_list[i], metadata={}) for i in task_idx_sorted]

    n = len(tasks)
    wins_matrix = np.zeros((n, n))
    for k in range(len(a_filt)):
        ai = idx_to_new[a_filt[k]]
        bi = idx_to_new[b_filt[k]]
        wi = idx_to_new[w_filt[k]]
        if wi == ai:
            wins_matrix[ai, bi] += 1
        else:
            wins_matrix[bi, ai] += 1

    data = PairwiseData(tasks=tasks, wins=wins_matrix)
    result = fit_thurstonian(data, max_iter=300)

    mu_full = np.full(len(task_id_list), np.nan)
    for new_idx, old_idx in enumerate(task_idx_sorted):
        mu_full[old_idx] = result.mu[new_idx]
    return mu_full


def reconstruct_iteration_order(measurements_json_path: Path) -> list[list[tuple[str, str]]]:
    """Reconstruct which pairs belong to which AL iteration.

    Returns list of 9 lists, each containing (task_a, task_b) canonical pairs
    for that iteration. Pairs are ordered by their first measurement index.
    """
    with open(measurements_json_path) as f:
        measurements = json.load(f)

    # Map each canonical pair to its first measurement index
    pair_first_idx: dict[tuple[str, str], int] = {}
    for i, m in enumerate(measurements):
        pair = (min(m["a"], m["b"]), max(m["a"], m["b"]))
        if pair not in pair_first_idx:
            pair_first_idx[pair] = i

    # Sort pairs by first measurement index
    sorted_pairs = sorted(pair_first_idx.items(), key=lambda x: x[1])

    # Assign to iterations based on known sizes
    iterations: list[list[tuple[str, str]]] = []
    offset = 0
    for size in ITER_SIZES:
        iteration_pairs = [p for p, _ in sorted_pairs[offset:offset + size]]
        iterations.append(iteration_pairs)
        offset += size

    assert offset == len(sorted_pairs), f"Expected {offset} == {len(sorted_pairs)}"
    return iterations


def pairs_to_indices(pairs_list: list[tuple[str, str]], id_to_idx: dict[str, int]) -> set[tuple[int, int]]:
    """Convert string pair list to set of (idx_low, idx_high) tuples."""
    result = set()
    for a, b in pairs_list:
        ai, bi = id_to_idx[a], id_to_idx[b]
        result.add((min(ai, bi), max(ai, bi)))
    return result


def filter_aggregated_by_pair_set(
    all_pairs: np.ndarray, all_wins_i: np.ndarray, all_total: np.ndarray,
    pair_set: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only aggregated pairs that are in pair_set."""
    mask = np.array([
        (int(all_pairs[i, 0]), int(all_pairs[i, 1])) in pair_set
        for i in range(len(all_pairs))
    ])
    return all_pairs[mask], all_wins_i[mask], all_total[mask]


def filter_measurements_by_pair_set(
    task_a_idx: np.ndarray, task_b_idx: np.ndarray, winner_idx: np.ndarray,
    pair_set: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only raw measurements whose canonical pair is in pair_set."""
    low = np.minimum(task_a_idx, task_b_idx)
    high = np.maximum(task_a_idx, task_b_idx)
    mask = np.array([
        (int(low[i]), int(high[i])) in pair_set
        for i in range(len(task_a_idx))
    ])
    return task_a_idx[mask], task_b_idx[mask], winner_idx[mask]


def evaluate_bt_scaled(acts, train_set, test_set, train_pairs, train_wins_i, train_total,
                       test_pairs, test_wins_i, test_total) -> float:
    scaler = StandardScaler()
    acts_scaled = np.copy(acts)
    acts_scaled[list(train_set)] = scaler.fit_transform(acts[list(train_set)])
    acts_scaled[list(test_set)] = scaler.transform(acts[list(test_set)])

    idx_i = train_pairs[:, 0]
    idx_j = train_pairs[:, 1]
    wins_j = train_total - train_wins_i
    total_w = float(np.sum(train_total))

    w = fit_bt(acts_scaled, idx_i, idx_j, train_wins_i, wins_j, total_w, BT_SCALED_LAMBDA)

    task_scores = acts_scaled @ w
    logits = task_scores[test_pairs[:, 0]] - task_scores[test_pairs[:, 1]]
    test_wins_j = test_total - test_wins_i
    correct = np.where(logits > 0, test_wins_i, test_wins_j)
    return float(np.sum(correct) / np.sum(test_total))


def evaluate_ridge(acts, task_a_idx, task_b_idx, winner_idx, task_id_list,
                   train_set, test_set, train_pair_set,
                   test_pairs, test_wins_i, test_total, n_tasks) -> float:
    # Filter measurements to only those in train_pair_set AND train tasks
    sub_a, sub_b, sub_w = filter_measurements_by_pair_set(
        task_a_idx, task_b_idx, winner_idx, train_pair_set
    )
    # Further filter to train tasks only
    mask = np.isin(sub_a, list(train_set)) & np.isin(sub_b, list(train_set))
    sub_a, sub_b, sub_w = sub_a[mask], sub_b[mask], sub_w[mask]

    if len(sub_a) < 100:
        return 0.5

    sub_task_set = set(np.unique(np.concatenate([sub_a, sub_b])).tolist())
    mu = fit_thurstonian_fast(sub_a, sub_b, sub_w, task_id_list, sub_task_set)

    valid_train = np.array([i for i in sorted(train_set) if not np.isnan(mu[i])])
    if len(valid_train) < 10:
        return 0.5

    scaler = StandardScaler()
    train_X = scaler.fit_transform(acts[valid_train])
    train_y = mu[valid_train]

    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(train_X, train_y)

    predicted = np.full(n_tasks, 0.0)
    predicted[valid_train] = model.predict(train_X)
    predicted[list(test_set)] = model.predict(scaler.transform(acts[list(test_set)]))

    return weighted_pairwise_accuracy(predicted, test_pairs, test_wins_i, test_total)


def run_experiment_5a(acts, task_a_idx, task_b_idx, winner_idx, task_id_list,
                      all_pairs, all_wins_i, all_total, folds,
                      iteration_pairs_idx: list[set[tuple[int, int]]]):
    """5a: Random subsample vs AL-order subsample at different fractions."""
    n_tasks = len(task_id_list)
    fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    n_seeds = 3

    # Build cumulative AL-order pair sets
    all_al_pairs_ordered: list[tuple[int, int]] = []
    for it_pairs in iteration_pairs_idx:
        all_al_pairs_ordered.extend(sorted(it_pairs))

    all_pair_set = set(all_al_pairs_ordered)
    total_pairs = len(all_pair_set)
    print(f"Total unique pairs: {total_pairs}")

    results = {"fractions": fractions, "n_seeds": n_seeds, "conditions": {}}

    for frac in fractions:
        n_select = int(frac * total_pairs)
        print(f"\n=== 5a: Fraction {frac} ({n_select} pairs) ===", flush=True)

        for condition in ["al_order", "random"]:
            accs_all_seeds = []

            for seed in range(n_seeds):
                if condition == "al_order":
                    # Take first n_select pairs in AL iteration order
                    selected_pair_set = set(all_al_pairs_ordered[:n_select])
                elif condition == "random":
                    rng = np.random.default_rng(seed + 100)
                    indices = rng.choice(total_pairs, size=n_select, replace=False)
                    selected_pair_set = {all_al_pairs_ordered[i] for i in indices}

                fold_bt_accs = []
                fold_ridge_accs = []

                for fold_idx, (train_idx, test_idx) in enumerate(folds):
                    train_set = set(train_idx.tolist())
                    test_set = set(test_idx.tolist())

                    test_pairs, test_wins_i, test_total = filter_pairs_by_tasks(
                        all_pairs, all_wins_i, all_total, test_set
                    )
                    if len(test_pairs) == 0:
                        continue

                    # Filter selected pairs to train tasks
                    train_pair_set = {
                        (a, b) for a, b in selected_pair_set
                        if a in train_set and b in train_set
                    }
                    train_pairs, train_wins_i_f, train_total_f = filter_aggregated_by_pair_set(
                        all_pairs, all_wins_i, all_total, train_pair_set
                    )

                    if len(train_pairs) < 10:
                        fold_bt_accs.append(0.5)
                        fold_ridge_accs.append(0.5)
                        continue

                    bt_acc = evaluate_bt_scaled(
                        acts, train_set, test_set,
                        train_pairs, train_wins_i_f, train_total_f,
                        test_pairs, test_wins_i, test_total,
                    )
                    fold_bt_accs.append(bt_acc)

                    ridge_acc = evaluate_ridge(
                        acts, task_a_idx, task_b_idx, winner_idx, task_id_list,
                        train_set, test_set, train_pair_set,
                        test_pairs, test_wins_i, test_total, n_tasks,
                    )
                    fold_ridge_accs.append(ridge_acc)

                bt_mean = float(np.mean(fold_bt_accs))
                ridge_mean = float(np.mean(fold_ridge_accs))
                accs_all_seeds.append({
                    "seed": seed,
                    "bt_fold_accs": fold_bt_accs,
                    "ridge_fold_accs": fold_ridge_accs,
                    "bt_mean": bt_mean,
                    "ridge_mean": ridge_mean,
                })
                print(f"  {condition} seed={seed}: BT={bt_mean:.4f}, Ridge={ridge_mean:.4f}", flush=True)

            key = f"{condition}_{frac}"
            results["conditions"][key] = accs_all_seeds

    return results


def run_experiment_5b(acts, task_a_idx, task_b_idx, winner_idx, task_id_list,
                      all_pairs, all_wins_i, all_total, folds,
                      iteration_pairs_idx: list[set[tuple[int, int]]]):
    """5b: Marginal value of AL-next vs random at different base sizes."""
    n_tasks = len(task_id_list)
    n_seeds = 3
    add_size = 2000

    # Build cumulative AL pair list
    all_al_pairs_ordered: list[tuple[int, int]] = []
    for it_pairs in iteration_pairs_idx:
        all_al_pairs_ordered.extend(sorted(it_pairs))

    # Cumulative iteration boundaries
    cum_sizes = np.cumsum(ITER_SIZES)  # [7500, 9500, 11500, 13500, 15500, 17500, 19500, 21500, 23500]

    # Base sizes where we can test: need at least add_size more AL pairs after the base
    # Base = first K pairs, then add next 2000 AL or 2000 random from remaining
    base_sizes = [5000, 7500, 10000, 15000, 20000]

    results = {"base_sizes": [], "add_size": add_size, "conditions": {}}

    for base_k in base_sizes:
        if base_k + add_size > len(all_al_pairs_ordered):
            print(f"Skipping base_k={base_k}: not enough remaining pairs", flush=True)
            continue

        results["base_sizes"].append(base_k)
        base_pair_set = set(all_al_pairs_ordered[:base_k])
        remaining_pairs = [p for p in all_al_pairs_ordered[base_k:]]
        al_next_pairs = set(all_al_pairs_ordered[base_k:base_k + add_size])

        print(f"\n=== 5b: Base K={base_k}, adding {add_size} ===", flush=True)

        for condition in ["base_only", "al_next", "random"]:
            accs_all_seeds = []

            for seed in range(n_seeds):
                if condition == "base_only":
                    selected = base_pair_set
                elif condition == "al_next":
                    selected = base_pair_set | al_next_pairs
                elif condition == "random":
                    rng = np.random.default_rng(seed + 200)
                    indices = rng.choice(len(remaining_pairs), size=add_size, replace=False)
                    random_add = {remaining_pairs[i] for i in indices}
                    selected = base_pair_set | random_add

                fold_bt_accs = []
                fold_ridge_accs = []

                for fold_idx, (train_idx, test_idx) in enumerate(folds):
                    train_set = set(train_idx.tolist())
                    test_set = set(test_idx.tolist())

                    test_pairs, test_wins_i, test_total = filter_pairs_by_tasks(
                        all_pairs, all_wins_i, all_total, test_set
                    )
                    if len(test_pairs) == 0:
                        continue

                    train_pair_set = {
                        (a, b) for a, b in selected
                        if a in train_set and b in train_set
                    }
                    train_pairs, train_wins_i_f, train_total_f = filter_aggregated_by_pair_set(
                        all_pairs, all_wins_i, all_total, train_pair_set
                    )

                    if len(train_pairs) < 10:
                        fold_bt_accs.append(0.5)
                        fold_ridge_accs.append(0.5)
                        continue

                    bt_acc = evaluate_bt_scaled(
                        acts, train_set, test_set,
                        train_pairs, train_wins_i_f, train_total_f,
                        test_pairs, test_wins_i, test_total,
                    )
                    fold_bt_accs.append(bt_acc)

                    ridge_acc = evaluate_ridge(
                        acts, task_a_idx, task_b_idx, winner_idx, task_id_list,
                        train_set, test_set, train_pair_set,
                        test_pairs, test_wins_i, test_total, n_tasks,
                    )
                    fold_ridge_accs.append(ridge_acc)

                bt_mean = float(np.mean(fold_bt_accs))
                ridge_mean = float(np.mean(fold_ridge_accs))
                accs_all_seeds.append({
                    "seed": seed,
                    "bt_fold_accs": fold_bt_accs,
                    "ridge_fold_accs": fold_ridge_accs,
                    "bt_mean": bt_mean,
                    "ridge_mean": ridge_mean,
                })
                print(f"  {condition} seed={seed}: BT={bt_mean:.4f}, Ridge={ridge_mean:.4f}", flush=True)

            key = f"{condition}_{base_k}"
            results["conditions"][key] = accs_all_seeds

    return results


def main():
    print("Loading data...", flush=True)
    task_a_idx, task_b_idx, winner_idx, task_id_list = load_measurements()
    acts = load_activations_layer(task_id_list)
    n_tasks = len(task_id_list)

    all_pairs, all_wins_i, all_total = aggregate_pairs(task_a_idx, task_b_idx, winner_idx)
    folds = get_task_folds(n_tasks, n_folds=5, seed=42)

    print("Reconstructing AL iteration order...", flush=True)
    iterations = reconstruct_iteration_order(MEASUREMENTS_JSON)
    id_to_idx = {tid: i for i, tid in enumerate(task_id_list)}

    iteration_pairs_idx = [pairs_to_indices(it, id_to_idx) for it in iterations]
    for i, it in enumerate(iteration_pairs_idx):
        print(f"  Iteration {i}: {len(it)} pairs")

    print("\n" + "=" * 60)
    print("EXPERIMENT 5a: Random vs AL-order subsampling")
    print("=" * 60)
    results_5a = run_experiment_5a(
        acts, task_a_idx, task_b_idx, winner_idx, task_id_list,
        all_pairs, all_wins_i, all_total, folds, iteration_pairs_idx,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 5b: Marginal value at different base sizes")
    print("=" * 60)
    results_5b = run_experiment_5b(
        acts, task_a_idx, task_b_idx, winner_idx, task_id_list,
        all_pairs, all_wins_i, all_total, folds, iteration_pairs_idx,
    )

    # Save results
    combined = {"experiment_5a": results_5a, "experiment_5b": results_5b}
    out_path = RESULTS_DIR / "experiment5_results.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n=== 5a SUMMARY ===")
    print(f"{'Fraction':<10} {'AL BT':>12} {'Rand BT':>12} {'AL Ridge':>12} {'Rand Ridge':>12}")
    for frac in results_5a["fractions"]:
        al_key = f"al_order_{frac}"
        rand_key = f"random_{frac}"
        al_bt = np.mean([s["bt_mean"] for s in results_5a["conditions"][al_key]])
        rand_bt = np.mean([s["bt_mean"] for s in results_5a["conditions"][rand_key]])
        al_ridge = np.mean([s["ridge_mean"] for s in results_5a["conditions"][al_key]])
        rand_ridge = np.mean([s["ridge_mean"] for s in results_5a["conditions"][rand_key]])
        print(f"{frac:<10.2f} {al_bt:>11.4f} {rand_bt:>11.4f} {al_ridge:>11.4f} {rand_ridge:>11.4f}")

    print("\n=== 5b SUMMARY ===")
    print(f"{'Base K':<10} {'Base BT':>12} {'AL+ BT':>12} {'Rand+ BT':>12} {'Base Ridge':>12} {'AL+ Ridge':>12} {'Rand+ Ridge':>12}")
    for base_k in results_5b["base_sizes"]:
        base_bt = np.mean([s["bt_mean"] for s in results_5b["conditions"][f"base_only_{base_k}"]])
        al_bt = np.mean([s["bt_mean"] for s in results_5b["conditions"][f"al_next_{base_k}"]])
        rand_bt = np.mean([s["bt_mean"] for s in results_5b["conditions"][f"random_{base_k}"]])
        base_ridge = np.mean([s["ridge_mean"] for s in results_5b["conditions"][f"base_only_{base_k}"]])
        al_ridge = np.mean([s["ridge_mean"] for s in results_5b["conditions"][f"al_next_{base_k}"]])
        rand_ridge = np.mean([s["ridge_mean"] for s in results_5b["conditions"][f"random_{base_k}"]])
        print(f"{base_k:<10} {base_bt:>11.4f} {al_bt:>11.4f} {rand_bt:>11.4f} {base_ridge:>11.4f} {al_ridge:>11.4f} {rand_ridge:>11.4f}")


if __name__ == "__main__":
    main()
