"""
Plot mean |steering effect| (averaged over reasonable coefficients ±1-4%)
against baseline ctrl_pa, for diff_ab at L31.

Uses absolute effects so positive and negative steering directions combine
rather than cancel. This gives a direction-agnostic measure of steerability.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path("experiments/steering/replication/fine_grained/results")
ASSETS_DIR = Path("experiments/steering/replication/fine_grained/assets")

# Load calibration to get multiplier mapping
with open(RESULTS_DIR / "calibration.json") as f:
    calib = json.load(f)

mean_norm = calib["ridge_L31"]["mean_norm"]

# Reasonable coefficient range: ±1% to ±4% of mean norm
REASONABLE_MULTIPLIERS = {-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04}
reasonable_coefs = {round(mean_norm * m, 2) for m in REASONABLE_MULTIPLIERS}

# Load phase 1 data
records = []
with open(RESULTS_DIR / "phase1_L31.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

# Build control P(a) per (pair_id, ordering)
ctrl_pa = {}
for r in records:
    if r["condition"] == "control":
        key = (r["pair_id"], r["ordering"])
        pa = sum(1 for x in r["responses"] if x == "a") / len(r["responses"])
        ctrl_pa[key] = pa

# For diff_ab at reasonable coefs, compute P(a) per (pair, ordering, coef)
# Then effect = P(a|steered) - P(a|control)
# For negative coefs, flip sign so "effect" = magnitude of change in expected direction
# Actually: just take |effect| for each coef, then average across coefs per pair×ordering

pair_effects = defaultdict(list)  # (pair_id, ordering) -> list of |effects|

for r in records:
    if r["condition"] != "diff_ab":
        continue
    # Check if this coefficient is in the reasonable range
    coef = r["coefficient"]
    # Match to nearest reasonable coef (floating point)
    multiplier = coef / mean_norm
    if not any(abs(multiplier - m) < 0.001 for m in REASONABLE_MULTIPLIERS):
        continue

    key = (r["pair_id"], r["ordering"])
    if key not in ctrl_pa:
        continue

    steered_pa = sum(1 for x in r["responses"] if x == "a") / len(r["responses"])
    effect = steered_pa - ctrl_pa[key]
    pair_effects[key].append(abs(effect))

# Average |effect| across reasonable coefs per pair×ordering
mean_abs_effects = {}
for key, effects in pair_effects.items():
    mean_abs_effects[key] = np.mean(effects)

# Prepare arrays
baseline = np.array([ctrl_pa[k] for k in mean_abs_effects])
steerability = np.array([mean_abs_effects[k] for k in mean_abs_effects])

# --- Plot 1: Scatter ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(baseline, steerability * 100, alpha=0.3, s=15, color="steelblue")
ax1.set_xlabel("Baseline P(a) (control, 10 resamples)")
ax1.set_ylabel("Mean |steering effect| (pp)\nacross ±1-4% coefs")
ax1.set_title("Steerability vs baseline decidedness (diff_ab, L31)")
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(0, 100)
ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)

# --- Plot 2: Binned means ---
bins = np.arange(0, 1.1, 0.1)
bin_centers = []
bin_means = []
bin_ses = []
bin_ns = []

for i in range(len(bins) - 1):
    lo, hi = bins[i], bins[i + 1]
    if i == len(bins) - 2:  # last bin includes upper edge
        mask = (baseline >= lo) & (baseline <= hi)
    else:
        mask = (baseline >= lo) & (baseline < hi)
    vals = steerability[mask] * 100
    if len(vals) > 0:
        bin_centers.append((lo + hi) / 2)
        bin_means.append(np.mean(vals))
        bin_ses.append(np.std(vals) / np.sqrt(len(vals)))
        bin_ns.append(len(vals))

ax2.errorbar(bin_centers, bin_means, yerr=[1.96 * s for s in bin_ses],
             fmt="o-", color="steelblue", capsize=4, markersize=8)

# Annotate with n
for x, y, n in zip(bin_centers, bin_means, bin_ns):
    ax2.annotate(f"n={n}", (x, y), textcoords="offset points",
                 xytext=(0, 12), ha="center", fontsize=8, color="gray")

ax2.set_xlabel("Baseline P(a) (control, 10 resamples)")
ax2.set_ylabel("Mean |steering effect| (pp)")
ax2.set_title("Binned steerability (diff_ab, L31, ±1-4% coefs)")
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(0, None)

plt.tight_layout()
plt.savefig(ASSETS_DIR / "plot_022526_steerability_symmetric.png", dpi=150, bbox_inches="tight")
print(f"Saved to {ASSETS_DIR / 'plot_022526_steerability_symmetric.png'}")

# Print table
print(f"\n{'Baseline P(a)':<15} {'n':>5} {'Mean |effect| (pp)':>20} {'SE':>8}")
print("-" * 55)
for c, m, s, n in zip(bin_centers, bin_means, bin_ses, bin_ns):
    lo = c - 0.05
    hi = c + 0.05
    print(f"[{lo:.1f}, {hi:.1f})      {n:>5} {m:>20.1f} {s:>8.1f}")

r = np.corrcoef(np.abs(baseline - 0.5), steerability)[0, 1]
print(f"\nPearson r(|ctrl_pa - 0.5|, mean |effect|) = {r:.3f}")
print(f"n = {len(baseline)}")
