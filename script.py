import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import numpy as np
from src.probes.data_loading import load_thurstonian_scores

# Paths
noprompt_dir = Path("results/experiments/mra_exp2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_mra_exp2_split_c_1000_task_ids")
midwest_dir = Path("results/experiments/mra_exp2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0_sys5d504504_mra_exp2_split_c_1000_task_ids")

# Load scores
noprompt_scores = load_thurstonian_scores(noprompt_dir)
midwest_scores = load_thurstonian_scores(midwest_dir)

print("Loaded scores:")
print(f"  Noprompt: {len(noprompt_scores)} tasks")
print(f"  Midwest: {len(midwest_scores)} tasks")

# Find common task IDs
common_tasks = set(noprompt_scores.keys()) & set(midwest_scores.keys())
print(f"  Common tasks: {len(common_tasks)}")

# Align scores
noprompt_aligned = np.array([noprompt_scores[tid] for tid in common_tasks])
midwest_aligned = np.array([midwest_scores[tid] for tid in common_tasks])
task_ids_aligned = list(common_tasks)

# Summary statistics
print("\nSummary Statistics:")
print(f"Noprompt:  mean={noprompt_aligned.mean():.4f}, std={noprompt_aligned.std():.4f}, min={noprompt_aligned.min():.4f}, max={noprompt_aligned.max():.4f}")
print(f"Midwest:   mean={midwest_aligned.mean():.4f}, std={midwest_aligned.std():.4f}, min={midwest_aligned.min():.4f}, max={midwest_aligned.max():.4f}")

# Compute correlations
pearson_r, pearson_p = pearsonr(noprompt_aligned, midwest_aligned)
spearman_r, spearman_p = spearmanr(noprompt_aligned, midwest_aligned)

print("\nCorrelations:")
print(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
print(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.2e})")

# Compute differences
diffs = noprompt_aligned - midwest_aligned

# Top 10 tasks where noprompt > midwest
top_10_idx = np.argsort(diffs)[-10:][::-1]
print("\nTop 10 tasks (noprompt > midwest, by difference):")
for i, idx in enumerate(top_10_idx, 1):
    tid = task_ids_aligned[idx]
    diff = diffs[idx]
    print(f"  {i:2d}. Task {tid}: noprompt={noprompt_aligned[idx]:7.4f}, midwest={midwest_aligned[idx]:7.4f}, diff={diff:7.4f}")

# Bottom 10 tasks where noprompt < midwest
bottom_10_idx = np.argsort(diffs)[:10]
print("\nBottom 10 tasks (noprompt < midwest, by difference):")
for i, idx in enumerate(bottom_10_idx, 1):
    tid = task_ids_aligned[idx]
    diff = diffs[idx]
    print(f"  {i:2d}. Task {tid}: noprompt={noprompt_aligned[idx]:7.4f}, midwest={midwest_aligned[idx]:7.4f}, diff={diff:7.4f}")
