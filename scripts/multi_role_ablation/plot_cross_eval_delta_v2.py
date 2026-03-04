"""Cross-persona probe transfer minus baseline utility correlation.

Same as plot_cross_eval_pearson_v2 but each cell is:
  probe_transfer_r - utility_correlation_r(train_persona, eval_persona)

This isolates what the probe adds beyond what utility similarity would predict.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.ood_system_prompts.analyze_mra_utilities import load_persona_utilities

RESULTS_PATH = "results/experiments/mra_exp3/probes/mra_8persona_results.json"
OUTPUT_PATH = "docs/lw_post/assets/plot_030426_s5_cross_eval_delta_heatmap.png"

LAYERS = ["L31", "L43", "L55"]
PERSONAS = ["noprompt", "aesthete", "midwest", "villain", "sadist"]
PERSONA_LABELS = ["Default", "Aesthete", "Midwest", "Villain", "Sadist"]

# Load probe transfer results
with open(RESULTS_PATH) as f:
    results = json.load(f)

# Compute baseline utility correlations between all persona pairs
print("Loading utilities...")
utils = {p: load_persona_utilities(p) for p in PERSONAS}

utility_corr = np.zeros((len(PERSONAS), len(PERSONAS)))
for i, p1 in enumerate(PERSONAS):
    for j, p2 in enumerate(PERSONAS):
        shared = sorted(set(utils[p1]) & set(utils[p2]))
        v1 = np.array([utils[p1][t] for t in shared])
        v2 = np.array([utils[p2][t] for t in shared])
        utility_corr[i, j] = pearsonr(v1, v2)[0]
        if i != j:
            print(f"  {p1} <-> {p2}: r={utility_corr[i, j]:.3f} (n={len(shared)})")

# Plot delta heatmaps
fig = plt.figure(figsize=(18, 5.5))
gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.4)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
cax = fig.add_subplot(gs[0, 3])

for ax, layer in zip(axes, LAYERS):
    cross_eval = results["phase1"][layer]["cross_eval"]
    probe_matrix = np.array([
        [cross_eval[train][eval_p]["pearson_r"] for eval_p in PERSONAS]
        for train in PERSONAS
    ])

    # Delta: probe transfer - utility correlation
    # Rows = train persona, cols = eval persona
    # Utility correlation is symmetric, but for the "baseline" we use
    # corr(train_utils, eval_utils) which is the same regardless of direction
    delta = probe_matrix - utility_corr

    im = ax.imshow(delta, cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="equal")

    for i in range(len(PERSONAS)):
        for j in range(len(PERSONAS)):
            val = delta[i, j]
            color = "white" if abs(val) > 0.35 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(len(PERSONAS)))
    ax.set_xticklabels(PERSONA_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(PERSONAS)))
    ax.set_yticklabels(PERSONA_LABELS)
    ax.set_xlabel("Eval persona")
    ax.set_ylabel("Train persona")
    ax.set_title(layer)

fig.suptitle("Probe transfer minus utility correlation (Δr)", fontsize=13)
fig.colorbar(im, cax=cax, label="Δr (probe − utility)")
fig.subplots_adjust(top=0.85, bottom=0.2)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUTPUT_PATH}")
