"""Experiment 6: Task scaling — does more tasks help?

Subsample tasks at different fractions, train probes on the subsampled tasks,
evaluate on all remaining tasks. Sweep regularization at each fraction.
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
    filter_pairs_by_tasks,
    load_activations_layer,
    load_measurements,
    weighted_pairwise_accuracy,
)
from src.fitting.thurstonian_fitting.thurstonian import PairwiseData, fit_thurstonian
from src.task_data.task import OriginDataset, Task

RESULTS_DIR = Path("experiments/probe_science/bt_scaling")

RIDGE_ALPHAS = np.logspace(-1, 7, 20)
BT_LAMBDAS = np.logspace(-3, 5, 15)


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


def sweep_ridge_alpha(train_X, train_y):
    """Sweep Ridge alpha via 3-fold CV on R^2. Returns best alpha."""
    best_alpha = RIDGE_ALPHAS[0]
    best_score = -np.inf
    for alpha in RIDGE_ALPHAS:
        cv_scores = cross_val_score(Ridge(alpha=alpha), train_X, train_y, cv=3, scoring="r2")
        mean_cv = float(np.mean(cv_scores))
        if mean_cv > best_score:
            best_score = mean_cv
            best_alpha = alpha
    return float(best_alpha)


def sweep_bt_lambda(acts_scaled, train_pairs, train_wins_i, train_total, sweep_seed):
    """Sweep BT lambda via 80/20 pair split. Returns best lambda."""
    idx_i = train_pairs[:, 0]
    idx_j = train_pairs[:, 1]
    wins_j = train_total - train_wins_i

    rng = np.random.default_rng(sweep_seed)
    val_mask = rng.random(len(train_pairs)) < 0.2
    tr_mask = ~val_mask

    tr_wi = train_wins_i[tr_mask]
    tr_wj = wins_j[tr_mask]
    tr_total = float(np.sum(tr_wi + tr_wj))
    val_wi = train_wins_i[val_mask]
    val_wj = wins_j[val_mask]
    val_total = float(np.sum(val_wi + val_wj))

    best_lambda = BT_LAMBDAS[0]
    best_acc = -np.inf
    for l2 in BT_LAMBDAS:
        w = fit_bt(acts_scaled, idx_i[tr_mask], idx_j[tr_mask], tr_wi, tr_wj, tr_total, l2)
        task_scores = acts_scaled @ w
        logits = task_scores[idx_i[val_mask]] - task_scores[idx_j[val_mask]]
        correct = np.where(logits > 0, val_wi, val_wj)
        val_acc = float(np.sum(correct) / val_total)
        if val_acc > best_acc:
            best_acc = val_acc
            best_lambda = l2
    return float(best_lambda)


def run_single(acts, task_a_idx, task_b_idx, winner_idx, task_id_list,
               all_pairs, all_wins_i, all_total,
               train_task_indices, test_task_indices, seed_label):
    """Run Ridge + BT on a single train/test split with HP sweeps."""
    n_tasks = len(task_id_list)
    train_set = set(train_task_indices.tolist())
    test_set = set(test_task_indices.tolist())

    train_pairs, train_wins_i_f, train_total_f = filter_pairs_by_tasks(
        all_pairs, all_wins_i, all_total, train_set
    )
    test_pairs, test_wins_i_f, test_total_f = filter_pairs_by_tasks(
        all_pairs, all_wins_i, all_total, test_set
    )

    n_train_tasks = len(train_task_indices)
    n_test_tasks = len(test_task_indices)
    n_train_pairs = len(train_pairs)
    n_test_pairs = len(test_pairs)

    if n_test_pairs == 0:
        print(f"    {seed_label}: No test pairs, skipping")
        return None

    # --- Ridge + Thurstonian ---
    mu = fit_thurstonian_fast(task_a_idx, task_b_idx, winner_idx, task_id_list, train_set)
    valid_train = np.array([i for i in train_task_indices if not np.isnan(mu[i])])

    ridge_acc = 0.5
    best_ridge_alpha = float("nan")
    if len(valid_train) >= 10:
        scaler = StandardScaler()
        train_X = scaler.fit_transform(acts[valid_train])
        train_y = mu[valid_train]

        best_ridge_alpha = sweep_ridge_alpha(train_X, train_y)

        model = Ridge(alpha=best_ridge_alpha)
        model.fit(train_X, train_y)

        predicted = np.full(n_tasks, 0.0)
        predicted[valid_train] = model.predict(train_X)
        predicted[list(test_set)] = model.predict(scaler.transform(acts[list(test_set)]))

        ridge_acc = weighted_pairwise_accuracy(predicted, test_pairs, test_wins_i_f, test_total_f)

    # --- BT + StandardScaler ---
    scaler_bt = StandardScaler()
    acts_scaled = np.copy(acts)
    acts_scaled[list(train_set)] = scaler_bt.fit_transform(acts[list(train_set)])
    acts_scaled[list(test_set)] = scaler_bt.transform(acts[list(test_set)])

    best_bt_lambda = sweep_bt_lambda(acts_scaled, train_pairs, train_wins_i_f, train_total_f, sweep_seed=42)

    idx_i = train_pairs[:, 0]
    idx_j = train_pairs[:, 1]
    wins_j = train_total_f - train_wins_i_f
    bt_total = float(np.sum(train_total_f))

    w_bt = fit_bt(acts_scaled, idx_i, idx_j, train_wins_i_f, wins_j, bt_total, best_bt_lambda)

    task_scores = acts_scaled @ w_bt
    logits = task_scores[test_pairs[:, 0]] - task_scores[test_pairs[:, 1]]
    test_wins_j = test_total_f - test_wins_i_f
    correct = np.where(logits > 0, test_wins_i_f, test_wins_j)
    bt_acc = float(np.sum(correct) / np.sum(test_total_f))

    print(f"    {seed_label}: {n_train_tasks} train tasks, {n_train_pairs} train pairs, "
          f"{n_test_tasks} test tasks, {n_test_pairs} test pairs | "
          f"Ridge={ridge_acc:.4f} (α={best_ridge_alpha:.0f}), BT={bt_acc:.4f} (λ={best_bt_lambda:.3f})")

    return {
        "ridge_acc": ridge_acc,
        "bt_acc": bt_acc,
        "best_ridge_alpha": best_ridge_alpha,
        "best_bt_lambda": best_bt_lambda,
        "n_train_tasks": n_train_tasks,
        "n_test_tasks": n_test_tasks,
        "n_train_pairs": n_train_pairs,
        "n_test_pairs": n_test_pairs,
    }


def main():
    print("Loading data...", flush=True)
    task_a_idx, task_b_idx, winner_idx, task_id_list = load_measurements()
    acts = load_activations_layer(task_id_list)
    n_tasks = len(task_id_list)

    all_pairs, all_wins_i, all_total = aggregate_pairs(task_a_idx, task_b_idx, winner_idx)

    fractions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    n_seeds = 3

    results = {"fractions": fractions, "n_seeds": n_seeds, "runs": []}

    for frac in fractions:
        n_train = int(frac * n_tasks)
        print(f"\n=== Fraction {frac} ({n_train} tasks) ===", flush=True)

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed + 300)

            if frac >= 1.0:
                # At 100%, no held-out tasks — use 5-fold CV instead
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                fold_results = []
                for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_tasks))):
                    r = run_single(
                        acts, task_a_idx, task_b_idx, winner_idx, task_id_list,
                        all_pairs, all_wins_i, all_total,
                        train_idx, test_idx, f"seed={seed} fold={fold_idx}",
                    )
                    if r is not None:
                        fold_results.append(r)

                if fold_results:
                    mean_ridge = float(np.mean([r["ridge_acc"] for r in fold_results]))
                    mean_bt = float(np.mean([r["bt_acc"] for r in fold_results]))
                    results["runs"].append({
                        "fraction": frac,
                        "seed": seed,
                        "ridge_acc": mean_ridge,
                        "bt_acc": mean_bt,
                        "n_train_tasks": n_tasks,
                        "fold_results": fold_results,
                    })
                    print(f"  seed={seed} (5-fold mean): Ridge={mean_ridge:.4f}, BT={mean_bt:.4f}")
            else:
                train_indices = rng.choice(n_tasks, size=n_train, replace=False)
                test_indices = np.setdiff1d(np.arange(n_tasks), train_indices)

                r = run_single(
                    acts, task_a_idx, task_b_idx, winner_idx, task_id_list,
                    all_pairs, all_wins_i, all_total,
                    train_indices, test_indices, f"seed={seed}",
                )
                if r is not None:
                    results["runs"].append({
                        "fraction": frac,
                        "seed": seed,
                        **r,
                    })

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    print(f"{'Fraction':<10} {'N tasks':<10} {'Ridge':>15} {'BT+scaled':>15} {'Best α':>12} {'Best λ':>12}")
    for frac in fractions:
        frac_runs = [r for r in results["runs"] if r["fraction"] == frac]
        ridge_accs = [r["ridge_acc"] for r in frac_runs]
        bt_accs = [r["bt_acc"] for r in frac_runs]

        if frac < 1.0:
            alphas = [r["best_ridge_alpha"] for r in frac_runs]
            lambdas = [r["best_bt_lambda"] for r in frac_runs]
        else:
            alphas = [np.mean([fr["best_ridge_alpha"] for fr in r["fold_results"]]) for r in frac_runs]
            lambdas = [np.mean([fr["best_bt_lambda"] for fr in r["fold_results"]]) for r in frac_runs]

        n_train = int(frac * n_tasks)
        print(f"{frac:<10.1f} {n_train:<10} "
              f"{np.mean(ridge_accs):>7.4f}±{np.std(ridge_accs):.4f} "
              f"{np.mean(bt_accs):>7.4f}±{np.std(bt_accs):.4f} "
              f"{np.mean(alphas):>11.0f} "
              f"{np.mean(lambdas):>11.3f}")

    out_path = RESULTS_DIR / "experiment6_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
