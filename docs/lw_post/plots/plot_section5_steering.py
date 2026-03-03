"""Section 5 steering plots for LW post.

Plot 1: Revealed preference dose-response (differential steering at L31)
Plot 2: Stated preference dose-response (by steering position, zoomed)

Both use % of mean activation norm on x-axis.
Coherence bars overlaid on both.

Usage:
    cd docs/lw_post && python plot_section5_steering.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).parent.parent.parent.parent
ASSETS = Path(__file__).parent.parent / "assets"

MEAN_NORM = 52_823  # L31 mean activation norm

# Load coherence data (last_token steering at L31)
coh = json.load(open(
    REPO / "experiments/steering/stated_steering/coherence_test/results/coherence_by_coefficient.json"
))
coh_coefs_raw = sorted(coh.keys(), key=float)
coh_pct_norms = [float(c) / MEAN_NORM * 100 for c in coh_coefs_raw]
coh_pcts = [coh[c]["coherent_pct"] * 100 for c in coh_coefs_raw]
COHERENCE_THRESHOLD = 90


def add_coherence_bars(ax, pct_norms, pcts, bar_width=1.0):
    """Add coherence bars on a twin y-axis. X-axis is % of mean norm."""
    ax2 = ax.twinx()
    colors = ['#C8E6C9' if p >= COHERENCE_THRESHOLD else '#FFCDD2' for p in pcts]
    ax2.bar(pct_norms, pcts, width=bar_width, color=colors, edgecolor='none',
            alpha=0.4, zorder=0)
    ax2.set_ylabel('Coherence (%)', fontsize=9, color='#999')
    ax2.set_ylim(0, 115)
    ax2.tick_params(axis='y', colors='#999', labelsize=8)
    return ax2


# ── Plot 1: Revealed preference dose-response ──────────────────────

analysis = json.load(open(
    REPO / "experiments/steering/replication/fine_grained/results/analysis.json"
))

diff = analysis["phase1_L31"]["diff_ab"]["by_coef"]

coefs = sorted(diff.keys(), key=float)
pct_norms = [float(c) / MEAN_NORM * 100 for c in coefs]
effects = [diff[c]["mean_effect_pp"] / 100 for c in coefs]
ses = [diff[c]["se_pp"] / 100 for c in coefs]

fig, ax = plt.subplots(figsize=(7, 4))

ax.errorbar(pct_norms, effects, yerr=[1.96 * s for s in ses],
            fmt='o', color='#1565C0', ecolor='#1565C0', elinewidth=1.2,
            capsize=3, markersize=7, zorder=3)

ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')

ax.set_xlabel('Steering coefficient (% of mean activation norm)', fontsize=10)
ax.set_ylabel('ΔP(choose task A)', fontsize=10)
ax.set_title('Revealed preference steering', fontsize=12, fontweight='bold', pad=18)
ax.text(0.5, 1.01, 'Steer +direction on task A tokens, −direction on task B tokens',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=9, color='#555',
        fontstyle='italic')

ax.set_ylim(-0.15, 0.18)
ax.set_xlim(-12, 12)

fig.tight_layout()
out1 = ASSETS / "plot_022626_s5_revealed_dose_response.png"
fig.savefig(out1, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved {out1}")


# ── Plot 2: Stated preference dose-response (3 formats, format replication) ──

stats = json.load(open(
    REPO / "experiments/steering/stated_steering/format_replication/results/statistics.json"
))

# Build lookup: (format, position) -> dose_response dict and parse_rate
dose_lookup = {}
parse_lookup = {}
for entry in stats:
    dose_lookup[(entry["format"], entry["position"])] = entry["dose_response"]
    parse_lookup[(entry["format"], entry["position"])] = entry["parse_rate"]

# Find which coefficients are below coherence threshold
incoherent_pct = set()
for c, data in coh.items():
    if data["coherent_pct"] * 100 < COHERENCE_THRESHOLD:
        incoherent_pct.add(round(float(c) / MEAN_NORM * 100, 1))

PARSE_THRESHOLD = 0.80

position_style = {
    'task_tokens': ('#42A5F5', 'Task tokens'),
    'generation': ('#66BB6A', 'During generation'),
    'last_token': ('#EF5350', 'Final task token'),
}

format_config = [
    ("qualitative_ternary", '"Rate as good, neutral, or bad"', (1.0, 3.0), "Mean rating (1–3)"),
    ("adjective_pick", "Pick from 10 adjectives\n(dreading → thrilled)", (3.0, 9.0), "Mean rating (1–10)"),
    ("anchored_simple_1_5", "Rate 1–5 with anchored examples", (3.0, 5.0), "Mean rating (1–5)"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

COHERENCE_BOUNDARY_PCT = 5.0  # ±5% of mean norm

for ax, (fmt, title, ylim, ylabel) in zip(axes, format_config):
    for pos, (color, label) in position_style.items():
        dr = dose_lookup[(fmt, pos)]
        coefs_raw = sorted(dr.keys(), key=float)
        # Filter by minimum parse rate only
        coefs_filtered = []
        for c in coefs_raw:
            n_expected = 2000  # 200 tasks × 10 samples
            if dr[c]["n"] / n_expected < PARSE_THRESHOLD:
                continue
            coefs_filtered.append(c)

        pct_xs = [float(c) / MEAN_NORM * 100 for c in coefs_filtered]
        means = [dr[c]["mean"] for c in coefs_filtered]

        # Split into coherent and incoherent segments
        coherent_x, coherent_y = [], []
        left_x, left_y = [], []
        right_x, right_y = [], []
        for x, y in zip(pct_xs, means):
            if abs(x) <= COHERENCE_BOUNDARY_PCT:
                coherent_x.append(x)
                coherent_y.append(y)
            elif x < -COHERENCE_BOUNDARY_PCT:
                left_x.append(x)
                left_y.append(y)
            else:
                right_x.append(x)
                right_y.append(y)

        # Full line faded for continuity
        ax.plot(pct_xs, means, '-o', color=color, linewidth=1.0, markersize=3,
                alpha=0.2, zorder=2)
        # Coherent region highlighted
        ax.plot(coherent_x, coherent_y, '-o', color=color, linewidth=1.5, markersize=4,
                zorder=3, label=label)

    ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
    ax.axvline(-COHERENCE_BOUNDARY_PCT, color='black', linestyle='--', alpha=0.3)
    ax.axvline(COHERENCE_BOUNDARY_PCT, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Steering coefficient (% of mean norm)', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylim(*ylim)
    ax.set_xlim(-12, 12)

axes[0].legend(fontsize=8, loc='upper left')

fig.suptitle('Stated preference steering (L31)', fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
out2 = ASSETS / "plot_022626_s5_stated_dose_response.png"
fig.savefig(out2, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved {out2}")
