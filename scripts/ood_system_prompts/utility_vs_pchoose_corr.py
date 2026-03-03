"""Correlation between Thurstonian utility deltas and p_choose deltas."""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.ood_system_prompts.analyze_utility_shifts import (
    load_experiment_utilities,
    compute_deltas,
)

PAIRWISE_PATH = Path("results/ood/hidden_preference/pairwise.json")

# Load Thurstonian
df, _ = load_experiment_utilities("exp1b")
thu_deltas = compute_deltas(df)

# Load p_choose
with open(PAIRWISE_PATH) as f:
    pairwise_data = json.load(f)

wins = defaultdict(lambda: defaultdict(int))
total = defaultdict(lambda: defaultdict(int))
n_comparisons = defaultdict(lambda: defaultdict(int))

for entry in pairwise_data["results"]:
    cid = entry["condition_id"]
    ta, tb = entry["task_a"], entry["task_b"]
    non_refusal = entry["n_a"] + entry["n_b"]
    wins[cid][ta] += entry["n_a"]
    wins[cid][tb] += entry["n_b"]
    total[cid][ta] += non_refusal
    total[cid][tb] += non_refusal
    n_comparisons[cid][ta] += 1
    n_comparisons[cid][tb] += 1

# Build matched arrays
thu_vals = []
pc_vals = []
labels = []

for cond in thu_deltas.columns:
    for tid in thu_deltas.index:
        thu_du = thu_deltas.loc[tid, cond]
        w_b = wins["baseline"].get(tid, 0)
        t_b = total["baseline"].get(tid, 0)
        w_c = wins[cond].get(tid, 0)
        t_c = total[cond].get(tid, 0)
        if t_b == 0 or t_c == 0:
            continue
        pc_delta = (w_c / t_c) - (w_b / t_b)
        thu_vals.append(thu_du)
        pc_vals.append(pc_delta)
        labels.append((cond, tid))

thu_arr = np.array(thu_vals)
pc_arr = np.array(pc_vals)

r, p = stats.pearsonr(thu_arr, pc_arr)
rho, rho_p = stats.spearmanr(thu_arr, pc_arr)

print(f"Overall correlation (n={len(thu_arr)} task-condition pairs with both measures):")
print(f"  Pearson r = {r:.4f} (p = {p:.2e})")
print(f"  Spearman ρ = {rho:.4f} (p = {rho_p:.2e})")

# Sign agreement
sign_agree = np.mean(np.sign(thu_arr) == np.sign(pc_arr))
# Exclude near-zero
mask = (np.abs(thu_arr) > 0.5) & (np.abs(pc_arr) > 0.02)
sign_agree_filtered = np.mean(np.sign(thu_arr[mask]) == np.sign(pc_arr[mask]))
print(f"  Sign agreement: {sign_agree:.3f} (all), {sign_agree_filtered:.3f} (|thu|>0.5, |pc|>0.02, n={mask.sum()})")

# Per-condition correlation
print(f"\nPer-condition correlations:")
print(f"  {'Condition':<40} {'r':>6} {'n':>4} {'n_comp/task':>12}")

for cond in sorted(thu_deltas.columns):
    t_list = []
    p_list = []
    comp_counts = []
    for tid in thu_deltas.index:
        w_b = wins["baseline"].get(tid, 0)
        t_b = total["baseline"].get(tid, 0)
        w_c = wins[cond].get(tid, 0)
        t_c = total[cond].get(tid, 0)
        if t_b == 0 or t_c == 0:
            continue
        t_list.append(thu_deltas.loc[tid, cond])
        p_list.append((w_c / t_c) - (w_b / t_b))
        comp_counts.append(n_comparisons[cond].get(tid, 0))

    if len(t_list) < 5:
        continue
    r_c, _ = stats.pearsonr(t_list, p_list)
    mean_comp = np.mean(comp_counts)
    print(f"  {cond:<40} {r_c:>6.3f} {len(t_list):>4} {mean_comp:>12.1f}")

# Specifically look at rainy_weather_neg comparison counts
print(f"\nComparison counts per task for rainy_weather_neg_persona:")
cond = "rainy_weather_neg_persona"
for tid in sorted(thu_deltas.index):
    if "rainy_weather" not in tid:
        continue
    nc = n_comparisons[cond].get(tid, 0)
    t_c = total[cond].get(tid, 0)
    print(f"  {tid}: {nc} unique pairs, {t_c} total trials")

# Compare to a well-behaved condition
print(f"\nComparison counts per task for cats_neg_persona (on-target):")
cond = "cats_neg_persona"
for tid in sorted(thu_deltas.index):
    if "cats" not in tid:
        continue
    nc = n_comparisons[cond].get(tid, 0)
    t_c = total[cond].get(tid, 0)
    print(f"  {tid}: {nc} unique pairs, {t_c} total trials")

# Also: what are the baseline p_choose values for rainy_weather tasks?
print(f"\nBaseline p_choose for rainy_weather tasks (higher = more popular at baseline):")
for tid in sorted(thu_deltas.index):
    if "rainy_weather" not in tid:
        continue
    w_b = wins["baseline"].get(tid, 0)
    t_b = total["baseline"].get(tid, 0)
    if t_b > 0:
        print(f"  {tid}: {w_b/t_b:.3f}")

print(f"\nBaseline p_choose for cats tasks:")
for tid in sorted(thu_deltas.index):
    if "cats" not in tid:
        continue
    w_b = wins["baseline"].get(tid, 0)
    t_b = total["baseline"].get(tid, 0)
    if t_b > 0:
        print(f"  {tid}: {w_b/t_b:.3f}")
