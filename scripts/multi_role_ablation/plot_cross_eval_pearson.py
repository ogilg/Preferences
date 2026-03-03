"""Cross-persona probe transfer heatmap using Pearson r (L31 only)."""

import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_PATH = "results/experiments/mra_exp2/probes/mra_results.json"
OUTPUT_PATH = "docs/lw_post/assets/plot_030226_s5_cross_eval_heatmap.png"

PERSONAS = ["noprompt", "villain", "aesthete", "midwest"]

with open(RESULTS_PATH) as f:
    results = json.load(f)

cross_eval = results["phase1"]["L31"]["cross_eval"]
matrix = np.array([
    [cross_eval[train][eval_p]["pearson_r"] for eval_p in PERSONAS]
    for train in PERSONAS
])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1.0, aspect="equal")

for i in range(len(PERSONAS)):
    for j in range(len(PERSONAS)):
        val = matrix[i, j]
        color = "white" if val < 0.3 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=12, color=color)

ax.set_xticks(range(len(PERSONAS)))
ax.set_xticklabels(PERSONAS, rotation=45, ha="right", fontsize=11)
ax.set_yticks(range(len(PERSONAS)))
ax.set_yticklabels(PERSONAS, fontsize=11)
ax.set_xlabel("Eval persona", fontsize=12)
ax.set_ylabel("Train persona", fontsize=12)
ax.set_title("Cross-persona probe transfer (Pearson r, L31)", fontsize=13)
fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT_PATH}")
