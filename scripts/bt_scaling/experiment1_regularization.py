"""Experiment 1: Regularization audit.

Fine-grained sweep on Ridge and BT at full data with task-level 5-fold CV.
Four variants:
  1. Ridge on Thurstonian mu (standard)
  2. BT on raw pairs (standard)
  3. BT with StandardScaler
  4. Ridge on raw win-rates

HP selection on fold 0 via internal validation. Fixed HP for folds 1-4.
All evaluated on held-out test pairs.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*LinAlg.*")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.bt_scaling.data_loading import (
    aggregate_pairs,
    compute_win_rates,
    filter_measurements_by_tasks,
    filter_pairs_by_tasks,
    get_task_folds,
    load_activations_layer,
    load_measurements,
    load_thurstonian_scores,
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


def fit_thurstonian_from_measurements(task_a_idx, task_b_idx, winner_idx, task_id_list, task_set):
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
    result = fit_thurstonian(data)

    mu_full = np.full(len(task_id_list), np.nan)
    for new_idx, old_idx in enumerate(task_idx_sorted):
        mu_full[old_idx] = result.mu[new_idx]
    return mu_full


def sweep_bt(acts, train_pairs, train_wins_i, train_total, lambdas, fold_seed):
    """Sweep BT lambda with 80/20 pair split on train data. Return sweep results."""
    idx_i = train_pairs[:, 0]
    idx_j = train_pairs[:, 1]
    wins_j = train_total - train_wins_i

    rng = np.random.default_rng(fold_seed)
    val_mask = rng.random(len(train_pairs)) < 0.2
    tr_mask = ~val_mask

    tr_wi = train_wins_i[tr_mask]
    tr_wj = wins_j[tr_mask]
    tr_total = float(np.sum(tr_wi + tr_wj))
    val_wi = train_wins_i[val_mask]
    val_wj = wins_j[val_mask]
    val_total = float(np.sum(val_wi + val_wj))

    sweep = []
    for l2 in lambdas:
        w = fit_bt(acts, idx_i[tr_mask], idx_j[tr_mask], tr_wi, tr_wj, tr_total, l2)
        val_acc = bt_weighted_accuracy(w, acts, idx_i[val_mask], idx_j[val_mask], val_wi, val_wj, val_total)
        train_acc = bt_weighted_accuracy(w, acts, idx_i[tr_mask], idx_j[tr_mask], tr_wi, tr_wj, tr_total)
        sweep.append({"l2_lambda": float(l2), "val_acc": val_acc, "train_acc": train_acc})
        print(f"      l2={l2:.2e}: train={train_acc:.4f}, val={val_acc:.4f}")

    return sweep


def run_experiment():
    print("Loading data...", flush=True)
    task_a_idx, task_b_idx, winner_idx, task_id_list = load_measurements()
    acts = load_activations_layer(task_id_list)
    n_tasks = len(task_id_list)

    all_pairs, all_wins_i, all_total = aggregate_pairs(task_a_idx, task_b_idx, winner_idx)
    folds = get_task_folds(n_tasks, n_folds=5, seed=42)

    ridge_alphas = np.logspace(-1, 7, 30)
    bt_lambdas = np.logspace(-3, 5, 15)

    results = {
        "ridge_thurstonian": {"sweep": None, "best_hp": None, "folds": []},
        "ridge_winrate": {"sweep": None, "best_hp": None, "folds": []},
        "bt_standard": {"sweep": None, "best_hp": None, "folds": []},
        "bt_scaled": {"sweep": None, "best_hp": None, "folds": []},
    }

    # ============ PHASE 1: HP selection on fold 0 ============
    train_idx, test_idx = folds[0]
    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())

    train_pairs, train_wins_i, train_total = filter_pairs_by_tasks(all_pairs, all_wins_i, all_total, train_set)
    train_a, train_b, train_w = filter_measurements_by_tasks(task_a_idx, task_b_idx, winner_idx, train_set)

    print("Fitting Thurstonian on fold 0 train...", flush=True)
    train_mu = fit_thurstonian_from_measurements(task_a_idx, task_b_idx, winner_idx, task_id_list, train_set)

    print("Computing win rates on fold 0 train...", flush=True)
    train_winrate = compute_win_rates(train_a, train_b, train_w, n_tasks)

    scaler = StandardScaler()
    train_X = scaler.fit_transform(acts[train_idx])

    # --- Ridge Thurstonian HP sweep ---
    print("\n--- Ridge Thurstonian HP sweep (fold 0) ---", flush=True)
    train_y = train_mu[train_idx]
    ridge_sweep = []
    for alpha in ridge_alphas:
        cv_scores = cross_val_score(Ridge(alpha=alpha), train_X, train_y, cv=3, scoring="r2")
        mean_cv = float(np.mean(cv_scores))
        ridge_sweep.append({"alpha": float(alpha), "cv_r2": mean_cv})
    best_ridge_alpha = max(ridge_sweep, key=lambda x: x["cv_r2"])["alpha"]
    results["ridge_thurstonian"]["sweep"] = ridge_sweep
    results["ridge_thurstonian"]["best_hp"] = best_ridge_alpha
    print(f"  Best Ridge alpha: {best_ridge_alpha:.2e}", flush=True)

    # --- Ridge win-rate HP sweep ---
    print("\n--- Ridge win-rate HP sweep (fold 0) ---", flush=True)
    wr_y = train_winrate[train_idx]
    wr_sweep = []
    for alpha in ridge_alphas:
        cv_scores = cross_val_score(Ridge(alpha=alpha), train_X, wr_y, cv=3, scoring="r2")
        mean_cv = float(np.mean(cv_scores))
        wr_sweep.append({"alpha": float(alpha), "cv_r2": mean_cv})
    best_wr_alpha = max(wr_sweep, key=lambda x: x["cv_r2"])["alpha"]
    results["ridge_winrate"]["sweep"] = wr_sweep
    results["ridge_winrate"]["best_hp"] = best_wr_alpha
    print(f"  Best WR alpha: {best_wr_alpha:.2e}", flush=True)

    # --- BT standard HP sweep ---
    print("\n--- BT standard HP sweep (fold 0) ---", flush=True)
    bt_sweep = sweep_bt(acts, train_pairs, train_wins_i, train_total, bt_lambdas, fold_seed=0)
    best_bt_lambda = max(bt_sweep, key=lambda x: x["val_acc"])["l2_lambda"]
    results["bt_standard"]["sweep"] = bt_sweep
    results["bt_standard"]["best_hp"] = best_bt_lambda
    print(f"  Best BT lambda: {best_bt_lambda:.2e}", flush=True)

    # --- BT scaled HP sweep ---
    print("\n--- BT scaled HP sweep (fold 0) ---", flush=True)
    scaler_bt = StandardScaler()
    acts_scaled_fold0 = np.copy(acts)
    acts_scaled_fold0[list(train_set)] = scaler_bt.fit_transform(acts[list(train_set)])
    acts_scaled_fold0[list(test_set)] = scaler_bt.transform(acts[list(test_set)])

    bt_scaled_sweep = sweep_bt(acts_scaled_fold0, train_pairs, train_wins_i, train_total, bt_lambdas, fold_seed=0)
    best_bt_scaled_lambda = max(bt_scaled_sweep, key=lambda x: x["val_acc"])["l2_lambda"]
    results["bt_scaled"]["sweep"] = bt_scaled_sweep
    results["bt_scaled"]["best_hp"] = best_bt_scaled_lambda
    print(f"  Best BT scaled lambda: {best_bt_scaled_lambda:.2e}", flush=True)

    # ============ PHASE 2: Evaluate all folds with fixed HPs ============
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n=== Fold {fold_idx} ===", flush=True)
        train_set = set(train_idx.tolist())
        test_set = set(test_idx.tolist())

        train_pairs, train_wins_i, train_total = filter_pairs_by_tasks(all_pairs, all_wins_i, all_total, train_set)
        test_pairs, test_wins_i, test_total = filter_pairs_by_tasks(all_pairs, all_wins_i, all_total, test_set)
        train_a, train_b, train_w = filter_measurements_by_tasks(task_a_idx, task_b_idx, winner_idx, train_set)

        print(f"  Train: {len(train_idx)} tasks, {len(train_pairs)} pairs | Test: {len(test_idx)} tasks, {len(test_pairs)} pairs", flush=True)

        # Thurstonian on train fold
        print(f"  Fitting Thurstonian...", flush=True)
        train_mu = fit_thurstonian_from_measurements(task_a_idx, task_b_idx, winner_idx, task_id_list, train_set)
        train_winrate = compute_win_rates(train_a, train_b, train_w, n_tasks)

        scaler = StandardScaler()
        train_X = scaler.fit_transform(acts[train_idx])
        test_X = scaler.transform(acts[test_idx])

        # --- Ridge Thurstonian ---
        model = Ridge(alpha=best_ridge_alpha)
        model.fit(train_X, train_mu[train_idx])
        predicted = np.full(n_tasks, 0.0)
        predicted[train_idx] = model.predict(train_X)
        predicted[test_idx] = model.predict(test_X)
        test_acc = weighted_pairwise_accuracy(predicted, test_pairs, test_wins_i, test_total)
        train_acc = weighted_pairwise_accuracy(predicted, train_pairs, train_wins_i, train_total)
        results["ridge_thurstonian"]["folds"].append({"fold": fold_idx, "train_acc": train_acc, "test_acc": test_acc})
        print(f"  Ridge Thurstonian: train={train_acc:.4f}, test={test_acc:.4f}", flush=True)

        # --- Ridge win-rate ---
        model = Ridge(alpha=best_wr_alpha)
        model.fit(train_X, train_winrate[train_idx])
        predicted_wr = np.full(n_tasks, 0.0)
        predicted_wr[train_idx] = model.predict(train_X)
        predicted_wr[test_idx] = model.predict(test_X)
        wr_test = weighted_pairwise_accuracy(predicted_wr, test_pairs, test_wins_i, test_total)
        wr_train = weighted_pairwise_accuracy(predicted_wr, train_pairs, train_wins_i, train_total)
        results["ridge_winrate"]["folds"].append({"fold": fold_idx, "train_acc": wr_train, "test_acc": wr_test})
        print(f"  Ridge win-rate: train={wr_train:.4f}, test={wr_test:.4f}", flush=True)

        # --- BT standard ---
        idx_i = train_pairs[:, 0]
        idx_j = train_pairs[:, 1]
        wins_j = train_total - train_wins_i
        bt_total = float(np.sum(train_wins_i + wins_j))

        print(f"  Training BT standard...", flush=True)
        w_bt = fit_bt(acts, idx_i, idx_j, train_wins_i, wins_j, bt_total, best_bt_lambda)
        bt_train = bt_weighted_accuracy(w_bt, acts, idx_i, idx_j, train_wins_i, wins_j, bt_total)
        bt_test = bt_weighted_accuracy(
            w_bt, acts, test_pairs[:, 0], test_pairs[:, 1],
            test_wins_i, test_total - test_wins_i, float(np.sum(test_total)),
        )
        results["bt_standard"]["folds"].append({"fold": fold_idx, "train_acc": bt_train, "test_acc": bt_test})
        print(f"  BT standard: train={bt_train:.4f}, test={bt_test:.4f}", flush=True)

        # --- BT scaled ---
        scaler_bt = StandardScaler()
        acts_scaled = np.copy(acts)
        acts_scaled[list(train_set)] = scaler_bt.fit_transform(acts[list(train_set)])
        acts_scaled[list(test_set)] = scaler_bt.transform(acts[list(test_set)])

        print(f"  Training BT scaled...", flush=True)
        w_bt_s = fit_bt(acts_scaled, idx_i, idx_j, train_wins_i, wins_j, bt_total, best_bt_scaled_lambda)
        bt_s_train = bt_weighted_accuracy(w_bt_s, acts_scaled, idx_i, idx_j, train_wins_i, wins_j, bt_total)
        bt_s_test = bt_weighted_accuracy(
            w_bt_s, acts_scaled, test_pairs[:, 0], test_pairs[:, 1],
            test_wins_i, test_total - test_wins_i, float(np.sum(test_total)),
        )
        results["bt_scaled"]["folds"].append({"fold": fold_idx, "train_acc": bt_s_train, "test_acc": bt_s_test})
        print(f"  BT scaled: train={bt_s_train:.4f}, test={bt_s_test:.4f}", flush=True)

    # --- Summary ---
    print("\n\n=== SUMMARY ===", flush=True)
    for variant, data in results.items():
        accs = [d["test_acc"] for d in data["folds"]]
        print(f"{variant}: {np.mean(accs):.4f} ± {np.std(accs):.4f}  (HP={data['best_hp']})")

    out_path = Path("experiments/probe_science/bt_scaling/experiment1_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
