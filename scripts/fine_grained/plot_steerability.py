"""
Steerability vs decidedness plot for the gemma-3-27b steering experiment.

Reads Phase 1 JSONL results and produces two plots:
  1. Binned summary: mean effect (pp) vs baseline P(a) bins
  2. Jittered scatter: per-ordering effect vs baseline P(a) with lowess line
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from scipy.stats import sem
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
JSONL_PATH = (
    "/workspace/repo/experiments/steering/replication/fine_grained/"
    "results/phase1_L31.jsonl"
)
ASSETS_DIR = (
    "/workspace/repo/experiments/steering/replication/fine_grained/assets/"
)
os.makedirs(ASSETS_DIR, exist_ok=True)

PEAK_COEF_TARGET = 1585.0  # +3% norm point

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
records = []
with open(JSONL_PATH) as fh:
    for line in fh:
        line = line.strip()
        if line:
            records.append(json.loads(line))

# Index by (pair_id, ordering)
from collections import defaultdict
by_key = defaultdict(list)
for r in records:
    key = (r["pair_id"], r["ordering"])
    by_key[key].append(r)

# ---------------------------------------------------------------------------
# Compute ctrl_pa and effect_pp for each pair×ordering
# ---------------------------------------------------------------------------
ctrl_pa_list = []
effect_pp_list = []
skipped = 0

for key, recs in by_key.items():
    # --- control at coef == 0 ---
    ctrl_recs = [
        r for r in recs
        if r["condition"] == "control" and r["coefficient"] == 0
    ]
    if not ctrl_recs:
        skipped += 1
        continue
    ctrl = ctrl_recs[0]
    responses = ctrl["responses"]
    valid = [resp for resp in responses if resp != "parse_fail"]
    if not valid:
        skipped += 1
        continue
    ctrl_pa = sum(1 for resp in valid if resp == "a") / len(valid)

    # --- diff_ab at peak positive coefficient ---
    diff_recs = [r for r in recs if r["condition"] == "diff_ab"]
    pos_diff = [r for r in diff_recs if r["coefficient"] > 0]
    if not pos_diff:
        skipped += 1
        continue
    peak_rec = min(pos_diff, key=lambda r: abs(r["coefficient"] - PEAK_COEF_TARGET))

    # p_a_steered from peak record responses
    peak_responses = peak_rec["responses"]
    peak_valid = [resp for resp in peak_responses if resp != "parse_fail"]
    if not peak_valid:
        skipped += 1
        continue
    p_a_steered = sum(1 for resp in peak_valid if resp == "a") / len(peak_valid)

    effect_pp = 100.0 * (p_a_steered - ctrl_pa)

    ctrl_pa_list.append(ctrl_pa)
    effect_pp_list.append(effect_pp)

ctrl_pa_arr = np.array(ctrl_pa_list)
effect_pp_arr = np.array(effect_pp_list)
n_total = len(ctrl_pa_arr)
print(f"Loaded {n_total} pair×orderings ({skipped} skipped)")

# ---------------------------------------------------------------------------
# Plot 1: Binned summary (bar chart with error bars)
# ---------------------------------------------------------------------------
bin_edges = np.arange(0.0, 1.1, 0.1)  # 0.0, 0.1, ..., 1.0
n_bins = len(bin_edges) - 1

bin_means = []
bin_cis = []
bin_ns = []
bin_centers = []

for i in range(n_bins):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    # Include right edge only for last bin
    if i < n_bins - 1:
        mask = (ctrl_pa_arr >= lo) & (ctrl_pa_arr < hi)
    else:
        mask = (ctrl_pa_arr >= lo) & (ctrl_pa_arr <= hi)
    vals = effect_pp_arr[mask]
    n = len(vals)
    bin_ns.append(n)
    bin_centers.append((lo + hi) / 2)
    if n > 0:
        bin_means.append(np.mean(vals))
    else:
        bin_means.append(np.nan)
    if n >= 3:
        # 95% CI = ±1.96 * SEM
        bin_cis.append(1.96 * sem(vals))
    else:
        bin_cis.append(np.nan)

bin_means = np.array(bin_means)
bin_cis = np.array(bin_cis)

# Color by decidedness: distance from 0.5
# Near 0.5 (undecided) → red/saturated; far from 0.5 (decided) → blue
def decidedness_color(center):
    dist = abs(center - 0.5)  # 0 = undecided, 0.5 = fully decided
    # Interpolate: 0 → red, 0.5 → blue
    t = dist / 0.5  # 0..1
    # red (1,0,0) → blue (0.2, 0.4, 0.8)
    r = 1.0 - t * 0.8
    g = 0.0 + t * 0.4
    b = 0.0 + t * 0.8
    alpha = 0.4 + t * 0.5  # more transparent for undecided, more opaque for decided
    return (r, g, b, alpha)

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.09  # slightly less than 0.1 for spacing

for i in range(n_bins):
    if np.isnan(bin_means[i]):
        continue
    color = decidedness_color(bin_centers[i])
    x_left = bin_edges[i]
    # Draw bar from x_left to x_left + bar_width (= bin_edges[i+1] - small gap)
    ax.bar(
        x_left + bar_width / 2,
        bin_means[i],
        width=bar_width,
        color=color[:3],
        alpha=color[3],
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
    )
    # Error bar (only if n >= 3 and CI is not nan)
    if not np.isnan(bin_cis[i]):
        ax.errorbar(
            x_left + bar_width / 2,
            bin_means[i],
            yerr=bin_cis[i],
            fmt="none",
            color="black",
            capsize=4,
            linewidth=1.2,
            zorder=3,
        )
    # n= label above/below bar
    label_y = bin_means[i] + (bin_cis[i] if not np.isnan(bin_cis[i]) else 0) + 0.5
    ax.text(
        x_left + bar_width / 2,
        label_y,
        f"n={bin_ns[i]}",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="black",
    )

ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=1)
ax.set_xlabel("Baseline P(a) at coef=0", fontsize=12)
ax.set_ylabel("Mean effect (percentage points)", fontsize=12)
ax.set_title(
    f"diff_ab effect at peak coef (+3% norm) by baseline P(a)\n"
    f"[L31, n={n_total} pair×orderings]",
    fontsize=12,
)
ax.set_xticks(bin_edges)
ax.set_xticklabels([f"{e:.1f}" for e in bin_edges], fontsize=9)
ax.set_xlim(-0.01, 1.01)

# Annotation: Pearson r (hardcoded as instructed)
ax.text(
    0.98, 0.03,
    "Pearson r(|ctrl_pa − 0.5|, effect) = −0.402, p < 0.001",
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=9,
    style="italic",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
)

# Colorbar legend (manual)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=(1.0, 0.0, 0.0, 0.45), edgecolor="black", label="Near 50/50 (undecided)"),
    Patch(facecolor=(0.2, 0.4, 0.8, 0.90), edgecolor="black", label="Near 0/1 (decided)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9, title="Decidedness")

plt.tight_layout()
out1 = os.path.join(ASSETS_DIR, "plot_022426_steerability_vs_ctrl_pa.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ---------------------------------------------------------------------------
# Plot 2: Jittered scatter with lowess
# ---------------------------------------------------------------------------
def lowess_numpy(x, y, frac=0.4):
    """Locally weighted scatterplot smoothing (LOWESS) implemented with numpy.

    Parameters
    ----------
    x, y : array-like, sorted by x
    frac  : float, fraction of data used for each local regression window

    Returns
    -------
    y_smooth : ndarray, smoothed y values at each x point
    """
    n = len(x)
    half_window = max(int(frac * n / 2), 1)
    y_smooth = np.empty(n)
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        xi = x[lo:hi]
        yi = y[lo:hi]
        # Tricube weights centred on x[i]
        dist = np.abs(xi - x[i])
        max_dist = dist.max() if dist.max() > 0 else 1.0
        u = dist / max_dist
        w = np.clip(1.0 - u ** 3, 0, None) ** 3
        # Weighted least-squares (degree 1)
        Xmat = np.column_stack([np.ones_like(xi), xi])
        W = np.diag(w)
        try:
            XtWX = Xmat.T @ W @ Xmat
            XtWy = Xmat.T @ W @ yi
            coef = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
            y_smooth[i] = coef[0] + coef[1] * x[i]
        except np.linalg.LinAlgError:
            y_smooth[i] = np.mean(yi)
    return y_smooth


# Jitter x values
rng = np.random.default_rng(42)
jitter = rng.uniform(-0.02, 0.02, size=n_total)
x_jittered = ctrl_pa_arr + jitter

# Lowess smooth on the original (un-jittered) values
sorted_idx = np.argsort(ctrl_pa_arr)
x_sorted = ctrl_pa_arr[sorted_idx]
y_sorted = effect_pp_arr[sorted_idx]
ly = lowess_numpy(x_sorted, y_sorted, frac=0.4)
lx = x_sorted

# Bootstrap 95% CI for lowess line
n_boot = 200
boot_curves = []
boot_x_grid = np.linspace(x_sorted.min(), x_sorted.max(), 200)

for _ in range(n_boot):
    idx_b = rng.integers(0, n_total, size=n_total)
    xb = ctrl_pa_arr[idx_b]
    yb = effect_pp_arr[idx_b]
    si = np.argsort(xb)
    yb_smooth = lowess_numpy(xb[si], yb[si], frac=0.4)
    # Interpolate onto grid
    interp = np.interp(boot_x_grid, xb[si], yb_smooth)
    boot_curves.append(interp)

boot_curves = np.array(boot_curves)
ci_low = np.percentile(boot_curves, 2.5, axis=0)
ci_high = np.percentile(boot_curves, 97.5, axis=0)

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    x_jittered, effect_pp_arr,
    alpha=0.3,
    color="steelblue",
    s=18,
    linewidths=0,
    zorder=2,
    label="pair×ordering",
)
ax.fill_between(
    boot_x_grid, ci_low, ci_high,
    color="steelblue",
    alpha=0.20,
    zorder=3,
    label="95% CI (bootstrap)",
)
ax.plot(lx, ly, color="navy", linewidth=2.0, zorder=4, label="Lowess (frac=0.4)")

ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.0, zorder=1, label="x=0.5")
ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=1, label="y=0")

ax.set_xlabel("Baseline P(a) [control, 10 resamples]", fontsize=12)
ax.set_ylabel("Effect (pp) at peak coefficient", fontsize=12)
ax.set_title(
    "Per-ordering steerability vs baseline P(a) [diff_ab, L31, +3% norm]",
    fontsize=12,
)
ax.set_xlim(-0.05, 1.05)
ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()
out2 = os.path.join(ASSETS_DIR, "plot_022426_steerability_scatter.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ---------------------------------------------------------------------------
# Verify files exist
# ---------------------------------------------------------------------------
for path in [out1, out2]:
    size_kb = os.path.getsize(path) / 1024
    print(f"  EXISTS: {path} ({size_kb:.1f} KB)")
