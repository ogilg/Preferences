"""
Analysis of Phase 1 Arm A interim results from the stated steering experiment.
Gemma-3-27b rates tasks 1-5 under activation steering at various positions and coefficients.
"""
import json
import numpy as np
from scipy import stats
from collections import defaultdict

DATA_PATH = "/workspace/repo/experiments/steering/stated_steering/results/phase1_arm_a.json"

with open(DATA_PATH) as f:
    data = json.load(f)

print("=" * 70)
print("PHASE 1 ARM A — STATED STEERING ANALYSIS")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# 1. Basic counts
# ─────────────────────────────────────────────────────────────
print("\n--- 1. BASIC COUNTS ---")

total_conditions = len(data)
task_ids = sorted({r["task_id"] for r in data})
positions = sorted({r["position"] for r in data})
coefficients = sorted({r["coefficient"] for r in data})

print(f"Total records:          {total_conditions}")
print(f"Unique tasks:           {len(task_ids)}")
print(f"Steering positions:     {positions}")
print(f"Coefficients ({len(coefficients)}):  {coefficients}")

# ─────────────────────────────────────────────────────────────
# 2. Per position: mean rating and parse rate at each coefficient
# ─────────────────────────────────────────────────────────────
print("\n--- 2. PER POSITION: MEAN RATING & PARSE RATE BY COEFFICIENT ---")

# Group by (position, coefficient)
by_pos_coef = defaultdict(list)
for r in data:
    key = (r["position"], r["coefficient"])
    by_pos_coef[key].append(r)

for pos in positions:
    print(f"\n  Position: {pos}")
    print(f"  {'Coefficient':>12}  {'Mean Rating':>12}  {'Parse Rate':>12}  {'N records':>10}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")
    for coef in coefficients:
        records = by_pos_coef[(pos, coef)]
        if not records:
            continue
        # Only include records with valid ratings
        valid = [r for r in records if r["mean_rating"] is not None]
        if not valid:
            print(f"  {coef:>12}  {'N/A':>12}  {'N/A':>12}  {len(records):>10}")
            continue
        mean_rating = np.mean([r["mean_rating"] for r in valid])
        mean_parse = np.mean([r["parse_rate"] for r in records])
        print(f"  {coef:>12}  {mean_rating:>12.4f}  {mean_parse:>12.4f}  {len(records):>10}")

# ─────────────────────────────────────────────────────────────
# 3. Per position: per-task slopes, then one-sample t-test vs 0
# ─────────────────────────────────────────────────────────────
print("\n--- 3. PER-TASK SLOPES & ONE-SAMPLE T-TEST (slope vs 0) ---")

# Group by (position, task_id)
by_pos_task = defaultdict(list)
for r in data:
    key = (r["position"], r["task_id"])
    by_pos_task[key].append(r)

pos_slopes = {}  # position -> list of (task_id, slope, baseline_mu)

for pos in positions:
    slopes = []
    task_ids_with_slopes = []
    baseline_mus = []

    for task_id in task_ids:
        records = by_pos_task[(pos, task_id)]
        # Only use records with valid mean_rating
        valid = [(r["coefficient"], r["mean_rating"]) for r in records if r["mean_rating"] is not None]
        if len(valid) < 3:
            continue
        coefs = np.array([v[0] for v in valid], dtype=float)
        ratings = np.array([v[1] for v in valid], dtype=float)

        result = stats.linregress(coefs, ratings)
        slope = result.slope

        # Baseline mu: rating at coef=0 (or task's stored mu, use mean_rating at coef=0)
        zero_records = [r for r in records if r["coefficient"] == 0 and r["mean_rating"] is not None]
        if zero_records:
            baseline = zero_records[0]["mean_rating"]
        else:
            baseline = records[0]["mu"]  # fall back to stored Thurstonian mu

        slopes.append(slope)
        task_ids_with_slopes.append(task_id)
        baseline_mus.append(baseline)

    pos_slopes[pos] = (task_ids_with_slopes, np.array(slopes), np.array(baseline_mus))

    slopes_arr = np.array(slopes)
    if len(slopes_arr) == 0:
        print(f"\n  Position: {pos} — no valid data")
        continue

    t_stat, p_val = stats.ttest_1samp(slopes_arr, popmean=0)
    print(f"\n  Position: {pos}")
    print(f"    N tasks with slopes:  {len(slopes_arr)}")
    print(f"    Mean slope:           {slopes_arr.mean():.6e}")
    print(f"    Std slope:            {slopes_arr.std():.6e}")
    print(f"    Min slope:            {slopes_arr.min():.6e}")
    print(f"    Max slope:            {slopes_arr.max():.6e}")
    print(f"    t-statistic:          {t_stat:.4f}")
    print(f"    p-value:              {p_val:.4e}")
    sig = "*** (p<0.001)" if p_val < 0.001 else "** (p<0.01)" if p_val < 0.01 else "* (p<0.05)" if p_val < 0.05 else "n.s."
    print(f"    Significance:         {sig}")

# ─────────────────────────────────────────────────────────────
# 4. Correlation of per-task slope with baseline mu
# ─────────────────────────────────────────────────────────────
print("\n--- 4. CORRELATION: PER-TASK SLOPE vs BASELINE MU (coef=0 rating) ---")

for pos in positions:
    task_ids_p, slopes_arr, mus_arr = pos_slopes[pos]
    if len(slopes_arr) < 3:
        print(f"\n  Position: {pos} — insufficient data")
        continue
    r, p = stats.pearsonr(mus_arr, slopes_arr)
    print(f"\n  Position: {pos}")
    print(f"    N tasks:              {len(slopes_arr)}")
    print(f"    Pearson r:            {r:.4f}")
    print(f"    p-value:              {p:.4e}")
    sig = "*** (p<0.001)" if p < 0.001 else "** (p<0.01)" if p < 0.01 else "* (p<0.05)" if p < 0.05 else "n.s."
    print(f"    Significance:         {sig}")
    if abs(r) > 0.3:
        direction = "positive" if r > 0 else "negative"
        print(f"    Interpretation: {direction} correlation — tasks with {'higher' if r > 0 else 'lower'} baseline are more steereable")

# ─────────────────────────────────────────────────────────────
# 5. task_tokens deep dive: signal check + slope histogram
# ─────────────────────────────────────────────────────────────
print("\n--- 5. TASK_TOKENS DEEP DIVE ---")

pos = "task_tokens"
_, slopes_arr, _ = pos_slopes[pos]

print(f"\n  Slope distribution for '{pos}':")
print(f"  N = {len(slopes_arr)}")

# Descriptive stats
percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]
pct_values = np.percentile(slopes_arr, percentiles)
print(f"\n  Percentiles:")
for p_val, v in zip(percentiles, pct_values):
    print(f"    {p_val:>3}th pct:  {v:.6e}")

# Count slopes in direction of positive vs negative steering
n_positive = (slopes_arr > 0).sum()
n_negative = (slopes_arr < 0).sum()
n_zero = (slopes_arr == 0).sum()
print(f"\n  Slope directions:")
print(f"    Positive slopes (task steereable upward):  {n_positive} ({100*n_positive/len(slopes_arr):.1f}%)")
print(f"    Negative slopes (task steereable downward): {n_negative} ({100*n_negative/len(slopes_arr):.1f}%)")
print(f"    Zero slopes:                                {n_zero} ({100*n_zero/len(slopes_arr):.1f}%)")

# Text histogram
print(f"\n  Text histogram of slopes (bins of 1e-5 width):")
hist_min = slopes_arr.min()
hist_max = slopes_arr.max()
n_bins = 20
bin_edges = np.linspace(hist_min, hist_max, n_bins + 1)
counts, edges = np.histogram(slopes_arr, bins=bin_edges)
max_count = counts.max()
bar_width = 40

for i in range(n_bins):
    bar_len = int(counts[i] / max_count * bar_width) if max_count > 0 else 0
    bar = "#" * bar_len
    center = (edges[i] + edges[i + 1]) / 2
    print(f"  [{center:+.3e}]  {bar:<{bar_width}}  {counts[i]}")

# ─────────────────────────────────────────────────────────────
# 6. Summary table
# ─────────────────────────────────────────────────────────────
print("\n--- 6. SUMMARY TABLE ---")
print(f"\n  {'Position':<15}  {'Mean Slope':>12}  {'t-stat':>8}  {'p-value':>10}  {'r(slope,mu)':>12}  {'sig':>15}")
print(f"  {'-'*15}  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*15}")

for pos in positions:
    task_ids_p, slopes_arr, mus_arr = pos_slopes[pos]
    if len(slopes_arr) < 3:
        print(f"  {pos:<15}  {'N/A':>12}")
        continue
    mean_slope = slopes_arr.mean()
    t_stat, p_val_t = stats.ttest_1samp(slopes_arr, popmean=0)
    r_corr, p_corr = stats.pearsonr(mus_arr, slopes_arr)
    sig_t = "***" if p_val_t < 0.001 else "**" if p_val_t < 0.01 else "*" if p_val_t < 0.05 else "n.s."
    sig_r = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "n.s."
    print(f"  {pos:<15}  {mean_slope:>12.4e}  {t_stat:>8.3f}  {p_val_t:>10.4e}  {r_corr:>12.4f} {sig_r:<5}  t: {sig_t}")

print("\n" + "=" * 70)
print("END OF ANALYSIS")
print("=" * 70)
