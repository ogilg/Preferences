"""Validate data loading and basic sanity checks."""
import sys
sys.path.insert(0, ".")

from scripts.bt_scaling.data_loading import (
    load_measurements, load_activations_layer, load_thurstonian_scores,
    aggregate_pairs, get_task_folds, filter_pairs_by_tasks,
    compute_win_rates, weighted_pairwise_accuracy,
)
import numpy as np

print("Loading measurements...")
task_a_idx, task_b_idx, winner_idx, task_id_list = load_measurements()
n_tasks = len(task_id_list)
print(f"  {len(task_a_idx)} measurements, {n_tasks} tasks")

print("\nAggregating pairs...")
pairs, wins_i, total = aggregate_pairs(task_a_idx, task_b_idx, winner_idx)
print(f"  {len(pairs)} unique pairs")
print(f"  Total comparisons: {int(total.sum())}")
print(f"  Mean comparisons per pair: {total.mean():.1f}")

print("\nLoading activations...")
acts = load_activations_layer(task_id_list)
print(f"  Shape: {acts.shape}")
print(f"  Any zero rows: {(np.abs(acts).sum(axis=1) == 0).sum()}")

print("\nLoading Thurstonian scores...")
mu = load_thurstonian_scores(task_id_list)
print(f"  Shape: {mu.shape}")
print(f"  Range: [{mu.min():.3f}, {mu.max():.3f}]")
print(f"  Mean: {mu.mean():.3f}, Std: {mu.std():.3f}")

print("\nThurstonian pairwise accuracy (sanity check)...")
acc = weighted_pairwise_accuracy(mu, pairs, wins_i, total)
print(f"  Full data: {acc:.4f}")

print("\nWin rate pairwise accuracy...")
wr = compute_win_rates(task_a_idx, task_b_idx, winner_idx, n_tasks)
acc_wr = weighted_pairwise_accuracy(wr, pairs, wins_i, total)
print(f"  Full data: {acc_wr:.4f}")

print("\n5-fold CV setup...")
folds = get_task_folds(n_tasks, n_folds=5, seed=42)
for i, (train_idx, test_idx) in enumerate(folds):
    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())
    _, _, test_total = filter_pairs_by_tasks(pairs, wins_i, total, test_set)
    _, _, train_total = filter_pairs_by_tasks(pairs, wins_i, total, train_set)
    print(f"  Fold {i}: train={len(train_idx)} tasks, test={len(test_idx)} tasks, "
          f"train_pairs={len(train_total)}, test_pairs={len(test_total)}")

print("\nValidation complete!")
