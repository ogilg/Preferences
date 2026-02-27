"""Section 5 steerability plot: steering effect vs baseline decidedness.

Shows that more borderline pairs (baseline P(a) near 0.5) are more steerable.
Uses direction-agnostic measure: mean |effect| across ±1% to ±4% coefficients.
Unit of analysis: pair × ordering (matching the report's n=582, r=−0.637).

Usage:
    cd docs/lw_post && python plot_section5_steerability.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

REPO = Path(__file__).parent.parent.parent
ASSETS = Path(__file__).parent / "assets"
MEAN_NORM = 52_823

# Load raw data
data_path = REPO / "experiments/steering/replication/fine_grained/results/phase1_L31.jsonl"
records = [json.loads(line) for line in open(data_path)]

# Compute baseline P(a) per (pair_id, ordering) from control condition
ctrl_pa = {}  # (pair_id, ordering) -> P(a)
for r in records:
    if r["condition"] == "control":
        key = (r["pair_id"], r["ordering"])
        p_a = sum(1 for x in r["responses"] if x == "a") / len(r["responses"])
        ctrl_pa[key] = p_a

# Compute per (pair×ordering) |effect| at each moderate coefficient for diff_ab
MODERATE_RANGE = (0.01, 0.045)  # 1% to 4.5% of mean norm

po_effects = defaultdict(list)  # (pair_id, ordering) -> list of |effect_pp|

for r in records:
    if r["condition"] != "diff_ab":
        continue
    pct_norm = abs(r["coefficient"]) / MEAN_NORM
    if not (MODERATE_RANGE[0] <= pct_norm <= MODERATE_RANGE[1]):
        continue

    key = (r["pair_id"], r["ordering"])
    if key not in ctrl_pa:
        continue

    p_a_steered = sum(1 for x in r["responses"] if x == "a") / len(r["responses"])
    effect = (p_a_steered - ctrl_pa[key]) * 100  # in pp
    po_effects[key].append(abs(effect))

# Mean |effect| per pair×ordering
po_mean_effect = {k: np.mean(v) for k, v in po_effects.items() if len(v) > 0}

# Build arrays
keys = [k for k in po_mean_effect if k in ctrl_pa]
baseline_pa = np.array([ctrl_pa[k] for k in keys])
mean_abs_effect = np.array([po_mean_effect[k] for k in keys])
decidedness = np.abs(baseline_pa - 0.5)

r_val, p_val = pearsonr(decidedness, mean_abs_effect)
print(f"r(|baseline P(a) - 0.5|, mean |effect|) = {r_val:.3f}, p = {p_val:.2e}, n={len(keys)}")

# Bin by baseline P(a)
bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_means = []
bin_ns = []
for i in range(len(bins) - 1):
    lo, hi = bins[i], bins[i+1]
    if i == len(bins) - 2:
        mask = (baseline_pa >= lo) & (baseline_pa <= hi)
    else:
        mask = (baseline_pa >= lo) & (baseline_pa < hi)
    bin_means.append(np.mean(mean_abs_effect[mask]) if mask.sum() > 0 else np.nan)
    bin_ns.append(mask.sum())

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

# Scatter
jitter = np.random.RandomState(42).normal(0, 0.008, len(baseline_pa))
ax.scatter(baseline_pa + jitter, mean_abs_effect, alpha=0.15, s=12, color='#1565C0',
           edgecolors='none', zorder=2)

# Binned means
valid = [i for i in range(len(bin_ns)) if bin_ns[i] >= 3]
ax.plot([bin_centers[i] for i in valid], [bin_means[i] for i in valid],
        'o-', color='#C62828', markersize=8, linewidth=2, zorder=3, label='Bin mean')

for i in valid:
    ax.annotate(f'n={bin_ns[i]}', (bin_centers[i], bin_means[i]),
                textcoords='offset points', xytext=(0, 10), fontsize=7,
                ha='center', color='#555')

ax.set_xlabel('Baseline P(choose A) at coefficient = 0', fontsize=11)
ax.set_ylabel('Mean |steering effect| (%)', fontsize=11)
ax.set_title('Steerability vs. Baseline Decidedness', fontsize=13, fontweight='bold', pad=20)
ax.text(0.5, 1.02,
        f'More borderline pairs show larger steering effects.',
        transform=ax.transAxes, ha='center', fontsize=9, color='#555555')

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0, None)
ax.axvline(0.5, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)

ax.legend(fontsize=9, loc='upper right')

fig.tight_layout()
outfile = ASSETS / "plot_022626_s5_steerability_vs_decidedness.png"
fig.savefig(outfile, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved {outfile}")
