"""Experiment 2: Pair selection oracle.

Would BT select different pairs than Thurstonian?
Replay pair selection retroactively using iteration boundaries.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.bt_scaling.data_loading import (
    aggregate_pairs,
    filter_measurements_by_tasks,
    load_activations_layer,
    load_measurements,
    load_thurstonian_scores,
)
from src.fitting.thurstonian_fitting.thurstonian import PairwiseData, fit_thurstonian
from src.task_data.task import OriginDataset, Task


# BT fitting (same as experiment 1)
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
    from scipy.optimize import minimize
    w0 = np.zeros(acts.shape[1])
    result = minimize(
        bt_loss_and_grad, w0, args=(acts, idx_i, idx_j, wins_i, wins_j, total_weight, l2_lambda),
        method="L-BFGS-B", jac=True, options={"maxiter": 500},
    )
    return result.x


def load_iteration_boundaries():
    """Load the JSON that maps pairs to iteration numbers.

    The measurements JSON has iteration info embedded. Iteration 1 = first 7500 pairs,
    subsequent = 2000 pairs each.
    """
    measurements_json = Path("scripts/active_learning_calibration/measurements_fast.json")
    with open(measurements_json) as f:
        raw = json.load(f)

    # Each measurement has a, b, c fields. Group by pair to find iteration assignment.
    # Since pairs are measured 5 times each, we can infer iteration by order of appearance.
    pair_first_seen: dict[tuple[str, str], int] = {}
    for i, m in enumerate(raw):
        a, b = m["a"], m["b"]
        key = tuple(sorted([a, b]))
        if key not in pair_first_seen:
            pair_first_seen[key] = i

    # Sort pairs by first appearance
    pairs_by_order = sorted(pair_first_seen.items(), key=lambda x: x[1])

    # Iteration boundaries: iter 1 = 7500 pairs, subsequent = 2000 pairs
    iterations: dict[int, list[tuple[str, str]]] = {}
    pair_count = 0
    current_iter = 1
    iter_size = 7500  # first iteration

    for pair, _ in pairs_by_order:
        if pair_count >= iter_size:
            current_iter += 1
            pair_count = 0
            iter_size = 2000  # subsequent iterations

        if current_iter not in iterations:
            iterations[current_iter] = []
        iterations[current_iter].append(pair)
        pair_count += 1

    return iterations


def run_experiment():
    print("Loading data...")
    task_a_idx, task_b_idx, winner_idx, task_id_list = load_measurements()
    acts = load_activations_layer(task_id_list)
    n_tasks = len(task_id_list)
    id_to_idx = {tid: i for i, tid in enumerate(task_id_list)}

    print("Loading iteration boundaries...")
    iterations = load_iteration_boundaries()
    print(f"  {len(iterations)} iterations found")
    for it, pairs in sorted(iterations.items()):
        print(f"    Iteration {it}: {len(pairs)} pairs")

    results = []

    # For each iteration boundary, replay selection
    for target_iter in range(2, min(6, len(iterations) + 1)):
        print(f"\n=== Replaying iteration {target_iter} selection ===")

        # Pairs available up to (target_iter - 1)
        available_pairs_set: set[tuple[str, str]] = set()
        for it in range(1, target_iter):
            available_pairs_set.update(iterations[it])

        # All measured pairs (to know which ones were actually selected in target_iter)
        actual_selected = set(iterations[target_iter])

        # Build measurement mask for available pairs
        available_pairs_idx = set()
        for a_str, b_str in available_pairs_set:
            if a_str in id_to_idx and b_str in id_to_idx:
                available_pairs_idx.add((id_to_idx[a_str], id_to_idx[b_str]))

        mask = np.zeros(len(task_a_idx), dtype=bool)
        for i in range(len(task_a_idx)):
            key = tuple(sorted([int(task_a_idx[i]), int(task_b_idx[i])]))
            if key in available_pairs_idx:
                mask[i] = True

        avail_a = task_a_idx[mask]
        avail_b = task_b_idx[mask]
        avail_w = winner_idx[mask]

        # Aggregate available pairs
        pairs, wins_i, total = aggregate_pairs(avail_a, avail_b, avail_w)
        print(f"  Available pairs: {len(pairs)}, measurements: {int(np.sum(total))}")

        # --- Train BT probe on available data ---
        print("  Training BT probe (lambda=10.0)...")
        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]
        wins_j = total - wins_i
        total_weight = float(np.sum(total))

        w_bt = fit_bt(acts, idx_i, idx_j, wins_i, wins_j, total_weight, 10.0)

        # --- Fit Thurstonian on available data ---
        print("  Fitting Thurstonian...")
        avail_tasks = set(np.unique(np.concatenate([avail_a, avail_b])).tolist())
        task_idx_sorted = sorted(avail_tasks)
        idx_to_new = {old: new for new, old in enumerate(task_idx_sorted)}
        tasks = [Task(prompt="", origin=OriginDataset.WILDCHAT, id=task_id_list[i], metadata={}) for i in task_idx_sorted]

        n_fit = len(tasks)
        wins_matrix = np.zeros((n_fit, n_fit))
        for k in range(len(avail_a)):
            ai = idx_to_new[avail_a[k]]
            bi = idx_to_new[avail_b[k]]
            wi = idx_to_new[avail_w[k]]
            if wi == ai:
                wins_matrix[ai, bi] += 1
            else:
                wins_matrix[bi, ai] += 1

        pairwise_data = PairwiseData(tasks=tasks, wins=wins_matrix)
        thurst_result = fit_thurstonian(pairwise_data)

        # Map mu back to full indices
        mu_avail = np.full(n_tasks, np.nan)
        for new_idx, old_idx in enumerate(task_idx_sorted):
            mu_avail[old_idx] = thurst_result.mu[new_idx]

        # --- Score remaining (unmeasured but measurable) pairs by BT uncertainty ---
        # Remaining pairs = all possible pairs among tasks with activations, minus already measured
        all_measured_idx = set()
        for a_str, b_str in available_pairs_set:
            if a_str in id_to_idx and b_str in id_to_idx:
                key = tuple(sorted([id_to_idx[a_str], id_to_idx[b_str]]))
                all_measured_idx.add(key)

        # We only consider pairs that were actually measured at some point
        # (we can't know about unmeasured pairs in retrospect)
        remaining_pairs_set: set[tuple[str, str]] = set()
        for it in range(target_iter, len(iterations) + 1):
            remaining_pairs_set.update(iterations[it])

        remaining_pairs_idx = []
        for a_str, b_str in remaining_pairs_set:
            if a_str in id_to_idx and b_str in id_to_idx:
                i, j = id_to_idx[a_str], id_to_idx[b_str]
                remaining_pairs_idx.append((min(i, j), max(i, j)))

        remaining_pairs_idx = list(set(remaining_pairs_idx))
        print(f"  Remaining unmeasured pairs: {len(remaining_pairs_idx)}")

        if len(remaining_pairs_idx) == 0:
            print("  No remaining pairs, skipping...")
            continue

        remaining_arr = np.array(remaining_pairs_idx)

        # BT uncertainty: |w · (act_i - act_j)| — lower = more uncertain
        bt_scores = acts @ w_bt
        bt_logits = np.abs(bt_scores[remaining_arr[:, 0]] - bt_scores[remaining_arr[:, 1]])

        # Thurstonian ambiguity: |mu_i - mu_j| — lower = more ambiguous
        thurst_diffs = np.abs(mu_avail[remaining_arr[:, 0]] - mu_avail[remaining_arr[:, 1]])

        # Handle NaN in Thurstonian (tasks not in available set)
        valid_mask = ~np.isnan(thurst_diffs)
        bt_logits_valid = bt_logits[valid_mask]
        thurst_diffs_valid = thurst_diffs[valid_mask]
        remaining_valid = remaining_arr[valid_mask]
        print(f"  Valid remaining pairs (both tasks have Thurstonian scores): {len(remaining_valid)}")

        # --- BT selects top-2000 most uncertain (lowest |logit|) ---
        batch_size = min(2000, len(remaining_valid))
        bt_selected_idx = np.argsort(bt_logits_valid)[:batch_size]
        bt_selected_pairs = set(map(tuple, remaining_valid[bt_selected_idx].tolist()))

        # --- What Thurstonian AL actually selected ---
        actual_selected_idx = set()
        for a_str, b_str in actual_selected:
            if a_str in id_to_idx and b_str in id_to_idx:
                i, j = id_to_idx[a_str], id_to_idx[b_str]
                actual_selected_idx.add((min(i, j), max(i, j)))

        # --- Metrics ---
        overlap = bt_selected_pairs & actual_selected_idx
        overlap_pct = len(overlap) / min(len(bt_selected_pairs), len(actual_selected_idx)) * 100

        # Rank correlation of uncertainty vs ambiguity
        rho, p_value = spearmanr(bt_logits_valid, thurst_diffs_valid)

        # Task coverage overlap
        bt_tasks = set()
        for i, j in bt_selected_pairs:
            bt_tasks.add(i)
            bt_tasks.add(j)
        actual_tasks = set()
        for i, j in actual_selected_idx:
            actual_tasks.add(i)
            actual_tasks.add(j)
        task_overlap = len(bt_tasks & actual_tasks) / len(bt_tasks | actual_tasks) * 100

        print(f"  BT selected: {len(bt_selected_pairs)} pairs")
        print(f"  Actual selected: {len(actual_selected_idx)} pairs")
        print(f"  Pair overlap: {len(overlap)} ({overlap_pct:.1f}%)")
        print(f"  Rank correlation (BT uncertainty vs Thurstonian ambiguity): rho={rho:.4f}, p={p_value:.2e}")
        print(f"  Task coverage overlap: {task_overlap:.1f}%")

        results.append({
            "target_iteration": target_iter,
            "n_available_pairs": len(pairs),
            "n_remaining_pairs": len(remaining_valid),
            "bt_selected": len(bt_selected_pairs),
            "actual_selected": len(actual_selected_idx),
            "pair_overlap": len(overlap),
            "pair_overlap_pct": overlap_pct,
            "rank_correlation": rho,
            "rank_p_value": p_value,
            "task_coverage_overlap_pct": task_overlap,
            "bt_tasks": len(bt_tasks),
            "actual_tasks": len(actual_tasks),
        })

    # Save results
    out_path = Path("experiments/probe_science/bt_scaling/experiment2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
