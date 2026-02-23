"""Correlate per-pair steering slope with baseline decidedness (spec Phase 2 analysis)."""
import json
import numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

with open('experiments/steering/replication/results/screening.json') as f:
    sc = json.load(f)
with open('experiments/steering/replication/results/steering_phase1.json') as f:
    st = json.load(f)

results = st['results']

# Compute per-pair baseline decidedness from screening
# Decidedness = |mean P(a) across orderings - 0.5|
pair_screening = defaultdict(lambda: {'n_a': 0, 'n_valid': 0})
for r in sc['results']:
    if r['n_valid'] > 0:
        pair_screening[r['pair_id']]['n_a'] += r['responses'].count('a')
        pair_screening[r['pair_id']]['n_valid'] += r['n_valid']

pair_decidedness = {}
for pair_id, d in pair_screening.items():
    if d['n_valid'] > 0:
        p_a = d['n_a'] / d['n_valid']
        pair_decidedness[pair_id] = abs(p_a - 0.5)

# Compute per-pair diff_ab slope at positive coefficients from steering
# For each pair: collect P(a) at each positive coef and at control
pair_p_a = defaultdict(lambda: defaultdict(list))  # pair_id -> coef -> [p_a values]
for t in results:
    if t['condition'] not in ('diff_ab', 'control'):
        continue
    coef = 0.0 if t['condition'] == 'control' else t['coefficient']
    if coef < 0:
        continue  # only positive coefs for slope
    for r in t['responses']:
        if r != 'parse_fail':
            pair_p_a[t['pair_id']][coef].append(1.0 if r == 'a' else 0.0)

# Compute per-pair slope = mean(diff_ab at +2641) - mean(control)
# Only use the moderate coefficient
MOD_COEF_APPROX = 2641  # approximate; actual is 2641.14...

pair_slopes = {}
for pair_id in pair_p_a:
    ctrl_vals = pair_p_a[pair_id].get(0.0, [])
    # Find the stored coef closest to 2641
    pos_coefs = [c for c in pair_p_a[pair_id] if c > 0]
    if not pos_coefs or not ctrl_vals:
        continue
    mod_coef = min(pos_coefs, key=lambda c: abs(c - MOD_COEF_APPROX))
    mod_vals = pair_p_a[pair_id][mod_coef]
    pair_slopes[pair_id] = np.mean(mod_vals) - np.mean(ctrl_vals)

# Restrict to borderline pairs that appear in both
common = set(pair_decidedness) & set(pair_slopes)
print(f'Pairs with both decidedness and slope: {len(common)}')

decidedness = np.array([pair_decidedness[p] for p in sorted(common)])
slopes = np.array([pair_slopes[p] for p in sorted(common)])

r, p = stats.pearsonr(decidedness, slopes)
print(f'Pearson r = {r:.3f}, p = {p:.4f}')
rs, ps = stats.spearmanr(decidedness, slopes)
print(f'Spearman rho = {rs:.3f}, p = {ps:.4f}')
print(f'Mean slope: {slopes.mean()*100:+.1f}pp')
print(f'Mean decidedness: {decidedness.mean():.3f}')

# Plot
sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(decidedness * 100, slopes * 100, alpha=0.6, edgecolors='white', linewidths=0.5, s=50)
m, b = np.polyfit(decidedness, slopes, 1)
x_range = np.linspace(decidedness.min(), decidedness.max(), 100)
ax.plot(x_range * 100, (m * x_range + b) * 100, 'r-', lw=2,
        label=f'r={r:.2f}, p={p:.3f}')
ax.axhline(0, color='grey', linestyle='--', lw=1)
ax.set_xlabel('Baseline decidedness |P(a) − 0.5| (%)', fontsize=12)
ax.set_ylabel('Per-pair shift at coef=+2641 (pp)', fontsize=12)
ax.set_title('Steering effect vs baseline decidedness (Phase 1 borderline pairs)', fontsize=12)
ax.set_xlim(0, 55)
ax.set_ylim(-60, 60)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('experiments/steering/replication/assets/plot_022226_phase1_decidedness_corr.png', dpi=150)
print('Saved plot_022226_phase1_decidedness_corr.png')
