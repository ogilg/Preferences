"""Appendix plot for GPT-OSS replication.

Grouped bar chart: per-topic within-topic (heldout) r vs cross-topic (HOO) r,
with refusal rate on secondary y-axis.
Style follows plot_022626_per_topic_hoo.png from section 3.

Data from experiments/gptoss_probes/gptoss_probes_report.md tables.

Usage:
    cd docs/lw_post && python plot_appendix_gptoss.py
"""

import matplotlib.pyplot as plt
import numpy as np

ASSETS = "assets"

# ── Data from report tables (L18) ──
# Sorted by within-topic r descending.
# sensitive_creative dropped (n=5, too noisy for heldout).

# (topic_label, heldout_n, heldout_r, hoo_n, hoo_r, refusal_rate)
# Refusal rates from report; topics not listed in refusal table have ~0% rate.
data = [
    ("Coding",              26,  0.889,   402, 0.759, 0.030),
    ("Knowledge QA",       183,  0.866,  2503, 0.801, 0.0),
    ("Persuasive Writing",  27,  0.860,   367, 0.739, 0.0),
    ("Fiction",             26,  0.856,   708, 0.719, 0.0),
    ("Summarization",       10,  0.854,   108, 0.767, 0.024),
    ("Other",                9,  0.842,    44, 0.791, 0.0),
    ("Content Generation",  59,  0.798,  1608, 0.735, 0.030),
    ("Math",               194,  0.784,  2677, 0.600, 0.004),
    ("Model Manipulation",  57,  0.445,   270, 0.646, 0.255),
    ("Security & Legal",    27,  0.441,   220, 0.512, 0.339),
    ("Harmful Request",    191,  0.258,  1012, 0.334, 0.350),
]

labels = [f"{t} (n={hn})" for t, hn, _, _, _, _ in data]
heldout_rs = [r for _, _, r, _, _, _ in data]
hoo_rs = [r for _, _, _, _, r, _ in data]
refusal_rates = [r for _, _, _, _, _, r in data]

x = np.arange(len(data))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))

# Match section 3 palette
ax.bar(x - width/2, heldout_rs, width, color='#4878A8', label='Within-topic (heldout)')
ax.bar(x + width/2, hoo_rs, width, color='#7A8B99', label='Cross-topic (HOO)')

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("Pearson r", fontsize=11)
ax.set_xlabel("Topic", fontsize=11)
ax.set_ylim(0, 1.0)

# Secondary y-axis for refusal rate
ax2 = ax.twinx()
ax2.plot(x, [r * 100 for r in refusal_rates], 'D', color='#C62828', markersize=6,
         label='Refusal rate', zorder=5)
ax2.set_ylabel("Refusal rate (%)", fontsize=11, color='#C62828')
ax2.set_ylim(0, 50)
ax2.tick_params(axis='y', labelcolor='#C62828')

ax.set_title(
    "GPT-OSS-120B: Per-Topic Probe Performance (L18)",
    fontsize=13, fontweight='bold', pad=12,
)
ax.text(0.5, 1.02,
        "Within-topic heldout r vs. cross-topic (hold-one-out) r. Refusal rate = % of pairs involving topic that were refused.",
        transform=ax.transAxes, ha='center', fontsize=9, color='#555555')

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9.5, loc='upper right')

fig.tight_layout()
outfile = f"{ASSETS}/plot_022626_appendix_heldout_vs_hoo.png"
fig.savefig(outfile, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved {outfile}")
