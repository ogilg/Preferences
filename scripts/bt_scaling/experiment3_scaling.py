"""Experiment 3: Scaling curves.

For fractions of train pairs, train Ridge (via Thurstonian) and BT,
evaluate held-out pairwise accuracy. 3 seeds per fraction.

Also includes BT+scaled variant since Experiment 1 showed it's competitive.
"""
import json
import sys
import warnings
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


def bt_weighted_accuracy(w, acts, idx_i, idx_j, wins_i, wins_j, total_weight):
    task_scores = acts @ w
    logits = task_scores[idx_i] - task_scores[idx_j]
    correct = np.where(logits > 0, wins_i, wins_j)
    return float(np.sum(correct) / total_weight)


def fit_thurstonian_fast(task_a_idx, task_b_idx, winner_idx, task_id_list, task_set):
    """Fast Thurstonian fit with higher tolerance."""
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


def subsample_pairs(all_pairs, all_wins_i, all_total, task_set, fraction, seed):
    """Subsample a fraction of train pairs."""
    train_pairs, train_wins_i, train_total = filter_pairs_by_tasks(all_pairs, all_wins_i, all_total, task_set)

    if fraction >= 1.0:
        return train_pairs, train_wins_i, train_total

    rng = np.random.default_rng(seed)
    n = len(train_pairs)
    n_select = max(1, int(n * fraction))
    indices = rng.choice(n, size=n_select, replace=False)

    return train_pairs[indices], train_wins_i[indices], train_total[indices]


def subsample_measurements_for_thurstonian(task_a_idx, task_b_idx, winner_idx, task_set, fraction, seed):
    """Subsample measurements by pairs (not individual comparisons)."""
    mask = np.isin(task_a_idx, list(task_set)) & np.isin(task_b_idx, list(task_set))
    a_filt = task_a_idx[mask]
    b_filt = task_b_idx[mask]
    w_filt = winner_idx[mask]

    if fraction >= 1.0:
        return a_filt, b_filt, w_filt

    # Group by pair
    pair_to_indices: dict[tuple[int, int], list[int]] = {}
    for i in range(len(a_filt)):
        key = (min(int(a_filt[i]), int(b_filt[i])), max(int(a_filt[i]), int(b_filt[i])))
        if key not in pair_to_indices:
            pair_to_indices[key] = []
        pair_to_indices[key].append(i)

    rng = np.random.default_rng(seed)
    all_keys = list(pair_to_indices.keys())
    n_select = max(1, int(len(all_keys) * fraction))
    selected_keys = rng.choice(len(all_keys), size=n_select, replace=False)

    selected_idx = []
    for k in selected_keys:
        selected_idx.extend(pair_to_indices[all_keys[k]])

    return a_filt[selected_idx], b_filt[selected_idx], w_filt[selected_idx]


def run_experiment():
    print("Loading data...", flush=True)
    task_a_idx, task_b_idx, winner_idx, task_id_list = load_measurements()
    acts = load_activations_layer(task_id_list)
    n_tasks = len(task_id_list)

    all_pairs, all_wins_i, all_total = aggregate_pairs(task_a_idx, task_b_idx, winner_idx)
    folds = get_task_folds(n_tasks, n_folds=5, seed=42)

    fractions = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    n_seeds = 3

    # Use HPs from Experiment 1
    ridge_alpha = 1374.0
    bt_lambda = 139.0
    bt_scaled_lambda = 0.193

    results = {
        "fractions": fractions,
        "n_seeds": n_seeds,
        "ridge_alpha": ridge_alpha,
        "bt_lambda": bt_lambda,
        "bt_scaled_lambda": bt_scaled_lambda,
        "ridge": [],
        "bt": [],
        "bt_scaled": [],
    }

    for frac in fractions:
        print(f"\n=== Fraction {frac} ===", flush=True)
        for seed in range(n_seeds):
            fold_ridge_accs = []
            fold_bt_accs = []
            fold_bt_scaled_accs = []

            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                train_set = set(train_idx.tolist())
                test_set = set(test_idx.tolist())

                test_pairs, test_wins_i, test_total = filter_pairs_by_tasks(all_pairs, all_wins_i, all_total, test_set)
                if len(test_pairs) == 0:
                    continue

                run_seed = seed * 1000 + fold_idx

                # Subsample train pairs
                sub_pairs, sub_wins_i, sub_total = subsample_pairs(
                    all_pairs, all_wins_i, all_total, train_set, frac, run_seed
                )

                # --- BT standard ---
                idx_i = sub_pairs[:, 0]
                idx_j = sub_pairs[:, 1]
                wins_j = sub_total - sub_wins_i
                bt_total = float(np.sum(sub_total))

                w_bt = fit_bt(acts, idx_i, idx_j, sub_wins_i, wins_j, bt_total, bt_lambda)
                bt_test = bt_weighted_accuracy(
                    w_bt, acts, test_pairs[:, 0], test_pairs[:, 1],
                    test_wins_i, test_total - test_wins_i, float(np.sum(test_total)),
                )
                fold_bt_accs.append(bt_test)

                # --- BT scaled ---
                scaler_bt = StandardScaler()
                acts_scaled = np.copy(acts)
                acts_scaled[list(train_set)] = scaler_bt.fit_transform(acts[list(train_set)])
                acts_scaled[list(test_set)] = scaler_bt.transform(acts[list(test_set)])

                w_bt_s = fit_bt(acts_scaled, idx_i, idx_j, sub_wins_i, wins_j, bt_total, bt_scaled_lambda)
                bt_s_test = bt_weighted_accuracy(
                    w_bt_s, acts_scaled, test_pairs[:, 0], test_pairs[:, 1],
                    test_wins_i, test_total - test_wins_i, float(np.sum(test_total)),
                )
                fold_bt_scaled_accs.append(bt_s_test)

                # --- Ridge on Thurstonian ---
                sub_a, sub_b, sub_w = subsample_measurements_for_thurstonian(
                    task_a_idx, task_b_idx, winner_idx, train_set, frac, run_seed
                )

                if len(sub_a) < 100:
                    fold_ridge_accs.append(0.5)
                    continue

                sub_task_set = set(np.unique(np.concatenate([sub_a, sub_b])).tolist())
                mu = fit_thurstonian_fast(sub_a, sub_b, sub_w, task_id_list, sub_task_set)

                valid_train = np.array([i for i in train_idx if not np.isnan(mu[i])])
                if len(valid_train) < 10:
                    fold_ridge_accs.append(0.5)
                    continue

                scaler = StandardScaler()
                train_X = scaler.fit_transform(acts[valid_train])
                train_y = mu[valid_train]

                model = Ridge(alpha=ridge_alpha)
                model.fit(train_X, train_y)

                predicted = np.full(n_tasks, 0.0)
                predicted[valid_train] = model.predict(train_X)
                predicted[list(test_set)] = model.predict(scaler.transform(acts[list(test_set)]))

                ridge_test = weighted_pairwise_accuracy(predicted, test_pairs, test_wins_i, test_total)
                fold_ridge_accs.append(ridge_test)

            mean_ridge = float(np.mean(fold_ridge_accs))
            mean_bt = float(np.mean(fold_bt_accs))
            mean_bt_s = float(np.mean(fold_bt_scaled_accs))
            print(f"  Seed {seed}: Ridge={mean_ridge:.4f}, BT={mean_bt:.4f}, BT_scaled={mean_bt_s:.4f}", flush=True)

            results["ridge"].append({"fraction": frac, "seed": seed, "fold_accs": fold_ridge_accs, "mean_acc": mean_ridge})
            results["bt"].append({"fraction": frac, "seed": seed, "fold_accs": fold_bt_accs, "mean_acc": mean_bt})
            results["bt_scaled"].append({"fraction": frac, "seed": seed, "fold_accs": fold_bt_scaled_accs, "mean_acc": mean_bt_s})

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    print(f"{'Fraction':<10} {'Ridge':>15} {'BT':>15} {'BT scaled':>15}", flush=True)
    for frac in fractions:
        ridge_means = [r["mean_acc"] for r in results["ridge"] if r["fraction"] == frac]
        bt_means = [r["mean_acc"] for r in results["bt"] if r["fraction"] == frac]
        bt_s_means = [r["mean_acc"] for r in results["bt_scaled"] if r["fraction"] == frac]
        print(f"{frac:<10.1f} {np.mean(ridge_means):>7.4f}±{np.std(ridge_means):.4f} {np.mean(bt_means):>7.4f}±{np.std(bt_means):.4f} {np.mean(bt_s_means):>7.4f}±{np.std(bt_s_means):.4f}", flush=True)

    out_path = Path("experiments/probe_science/bt_scaling/experiment3_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    run_experiment()
