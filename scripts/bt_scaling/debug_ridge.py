"""Debug why Ridge gives constant 0.532 at low fractions."""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.bt_scaling.data_loading import (
    aggregate_pairs,
    filter_pairs_by_tasks,
    get_task_folds,
    load_activations_layer,
    load_measurements,
    load_thurstonian_scores,
    weighted_pairwise_accuracy,
)
from src.fitting.thurstonian_fitting.thurstonian import PairwiseData, fit_thurstonian
from src.task_data.task import OriginDataset, Task

task_a_idx, task_b_idx, winner_idx, task_id_list = load_measurements()
acts = load_activations_layer(task_id_list)
n_tasks = len(task_id_list)
all_pairs, all_wins_i, all_total = aggregate_pairs(task_a_idx, task_b_idx, winner_idx)
folds = get_task_folds(n_tasks, n_folds=5, seed=42)
full_mu = load_thurstonian_scores(task_id_list)

# Test fold 0, fraction 0.1
train_idx, test_idx = folds[0]
train_set = set(train_idx.tolist())
test_set = set(test_idx.tolist())
test_pairs, test_wins_i, test_total = filter_pairs_by_tasks(all_pairs, all_wins_i, all_total, test_set)

# Subsample 10%
mask = np.isin(task_a_idx, list(train_set)) & np.isin(task_b_idx, list(train_set))
a_filt = task_a_idx[mask]
b_filt = task_b_idx[mask]
w_filt = winner_idx[mask]

pair_to_indices = {}
for i in range(len(a_filt)):
    key = (min(int(a_filt[i]), int(b_filt[i])), max(int(a_filt[i]), int(b_filt[i])))
    if key not in pair_to_indices:
        pair_to_indices[key] = []
    pair_to_indices[key].append(i)

rng = np.random.default_rng(0)
all_keys = list(pair_to_indices.keys())
n_select = int(len(all_keys) * 0.1)
selected_keys = rng.choice(len(all_keys), size=n_select, replace=False)
selected_idx = []
for k in selected_keys:
    selected_idx.extend(pair_to_indices[all_keys[k]])

sub_a = a_filt[selected_idx]
sub_b = b_filt[selected_idx]
sub_w = w_filt[selected_idx]

print(f"Subsampled: {len(sub_a)} measurements, {n_select} unique pairs")
sub_task_set = set(np.unique(np.concatenate([sub_a, sub_b])).tolist())
print(f"Tasks covered: {len(sub_task_set)} / {len(train_set)}")

# Fit Thurstonian
task_idx_sorted = sorted(sub_task_set)
idx_to_new = {old: new for new, old in enumerate(task_idx_sorted)}
tasks_list = [Task(prompt="", origin=OriginDataset.WILDCHAT, id=task_id_list[i], metadata={}) for i in task_idx_sorted]

n = len(tasks_list)
wins_matrix = np.zeros((n, n))
for k in range(len(sub_a)):
    ai = idx_to_new[sub_a[k]]
    bi = idx_to_new[sub_b[k]]
    wi = idx_to_new[sub_w[k]]
    if wi == ai:
        wins_matrix[ai, bi] += 1
    else:
        wins_matrix[bi, ai] += 1

data = PairwiseData(tasks=tasks_list, wins=wins_matrix)
result = fit_thurstonian(data, max_iter=200, gradient_tol=10.0, loss_tol=1e-6)

mu_full = np.full(n_tasks, np.nan)
for new_idx, old_idx in enumerate(task_idx_sorted):
    mu_full[old_idx] = result.mu[new_idx]

print(f"Thurstonian mu stats:")
valid_mu = mu_full[~np.isnan(mu_full)]
print(f"  Range: [{valid_mu.min():.4f}, {valid_mu.max():.4f}]")
print(f"  Mean: {valid_mu.mean():.4f}, Std: {valid_mu.std():.4f}")
print(f"  Converged: {result.converged}")
print(f"  Iterations: {result.n_iterations}")

# Train Ridge
valid_train = np.array([i for i in train_idx if not np.isnan(mu_full[i])])
print(f"Valid train tasks: {len(valid_train)}")

scaler = StandardScaler()
train_X = scaler.fit_transform(acts[valid_train])
train_y = mu_full[valid_train]
print(f"Train y stats: mean={train_y.mean():.4f}, std={train_y.std():.4f}")

model = Ridge(alpha=1374.0)
model.fit(train_X, train_y)
print(f"Ridge coef norm: {np.linalg.norm(model.coef_):.6f}")
print(f"Ridge intercept: {model.intercept_:.6f}")

predicted = np.full(n_tasks, 0.0)
predicted[valid_train] = model.predict(train_X)
predicted[list(test_set)] = model.predict(scaler.transform(acts[list(test_set)]))

acc = weighted_pairwise_accuracy(predicted, test_pairs, test_wins_i, test_total)
print(f"Test accuracy: {acc:.4f}")

# Compare with full-data Thurstonian
print(f"\n--- Full-data Thurstonian comparison ---")
print(f"Full mu stats: mean={full_mu.mean():.4f}, std={full_mu.std():.4f}")
# Correlation
from scipy.stats import pearsonr
common = ~np.isnan(mu_full) & ~np.isnan(full_mu)
r, _ = pearsonr(mu_full[common], full_mu[common])
print(f"Correlation subsampled vs full mu: {r:.4f}")
