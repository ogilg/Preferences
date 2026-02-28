"""Appendix plot: heldout probe performance by depth, GPT-OSS vs Gemma-3 (raw only).

Data from experiment reports:
- experiments/training_probes/gptoss_probes/gptoss_probes_report.md
- experiments/training_probes/gemma3_10k_probes/gemma3_10k_probes_report.md

Usage:
    cd docs/lw_post && python plots/plot_appendix_depth_comparison.py
"""

import matplotlib.pyplot as plt
import numpy as np

ASSETS = "assets"

# GPT-OSS-120B (36 layers) — fractional depth = layer / 36
gptoss_layers = [3, 7, 10, 14, 18, 21, 25, 28, 32]
gptoss_depth = [l / 36 for l in gptoss_layers]
gptoss_raw = [0.855, 0.865, 0.873, 0.909, 0.915, 0.913, 0.910, 0.907, 0.904]

# Gemma-3-27B (62 layers) — fractional depth = layer / 62
gemma_layers = [15, 31, 37, 43, 49, 55]
gemma_depth = [l / 62 for l in gemma_layers]
gemma_raw = [0.748, 0.864, 0.853, 0.849, 0.845, 0.845]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(gptoss_depth, gptoss_raw, 'o-', color='#4878A8', linewidth=2, markersize=7,
        label='GPT-OSS-120B')
ax.plot(gemma_depth, gemma_raw, 's-', color='#2E7D32', linewidth=2, markersize=7,
        label='Gemma-3-27B')

ax.set_xlabel("Fractional Depth", fontsize=11)
ax.set_ylabel("Heldout Pearson r", fontsize=11)
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=10)
ax.set_title("Heldout Probe Performance by Depth: GPT-OSS vs Gemma-3",
             fontsize=13, fontweight='bold')

fig.tight_layout()
outfile = f"{ASSETS}/plot_022626_appendix_depth_comparison.png"
fig.savefig(outfile, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved {outfile}")
