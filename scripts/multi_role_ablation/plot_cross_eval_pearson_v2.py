"""Cross-persona probe transfer heatmap: Pearson r, all 3 layers, v2 (train=2000, eval=250)."""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RESULTS_PATH = "results/experiments/mra_exp3/probes/mra_8persona_results.json"
OUTPUT_PATH = "docs/lw_post/assets/plot_030426_s5_cross_eval_heatmap.png"

LAYERS = ["L31", "L43", "L55"]
PERSONAS = ["noprompt", "aesthete", "midwest", "villain", "sadist"]
PERSONA_LABELS = ["Default", "Aesthete", "Midwest", "Villain", "Sadist"]

with open(RESULTS_PATH) as f:
    results = json.load(f)

fig = plt.figure(figsize=(18, 5.5))
gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.4)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
cax = fig.add_subplot(gs[0, 3])

for ax, layer in zip(axes, LAYERS):
    cross_eval = results["phase1"][layer]["cross_eval"]
    matrix = np.array([
        [cross_eval[train][eval_p]["pearson_r"] for eval_p in PERSONAS]
        for train in PERSONAS
    ])

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="equal")

    for i in range(len(PERSONAS)):
        for j in range(len(PERSONAS)):
            val = matrix[i, j]
            color = "white" if val < 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(len(PERSONAS)))
    ax.set_xticklabels(PERSONA_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(PERSONAS)))
    ax.set_yticklabels(PERSONA_LABELS)
    ax.set_xlabel("Eval persona")
    ax.set_ylabel("Train persona")
    ax.set_title(layer)

fig.suptitle("Cross-persona probe transfer (Pearson r)", fontsize=13)
fig.colorbar(im, cax=cax, label="Pearson r")
fig.subplots_adjust(top=0.85, bottom=0.2)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT_PATH}")
