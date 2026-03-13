"""Heatmaps: Cohen's d by layer × selector position, one per probe."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "truth_probes" / "error_prefill" / "error_prefill_followup_results.json"
ASSETS = ROOT / "experiments" / "truth_probes" / "error_prefill" / "assets"

with open(RESULTS) as f:
    data = json.load(f)

assistant_data = data["assistant_selectors_no_lying"]

SELECTORS = ["assistant_tb:-5", "assistant_tb:-4", "assistant_tb:-3", "assistant_tb:-2", "assistant_tb:-1", "assistant_mean"]
SELECTOR_LABELS = ["tb:-5\n<end_of_turn>", "tb:-4\n\\n", "tb:-3\n<start_of_turn>", "tb:-2\nuser", "tb:-1\n\\n", "mean\n(content)"]
PROBES = ["tb-2", "tb-5", "task_mean"]
PROBE_LABELS = ["Probe A (last-token)", "Probe B (EOT)", "Probe C (mean-pooled)"]
LAYERS = ["25", "32", "39", "46", "53"]

# Build d matrices: one per probe, shape (n_selectors, n_layers)
# Use "neutral" follow-up (all are identical for assistant selectors)
matrices = {}
for probe in PROBES:
    mat = np.zeros((len(SELECTORS), len(LAYERS)))
    for i, sel in enumerate(SELECTORS):
        sel_data = assistant_data[sel][probe]
        for j, layer in enumerate(LAYERS):
            mat[i, j] = sel_data["neutral"][layer]["cohens_d"]
    matrices[probe] = mat

# Plot: 3 heatmaps in a vertical stack for maximum readability
vmin = -0.5
vmax = 3.5

fig, axes = plt.subplots(3, 1, figsize=(9, 12))

for ax, probe, probe_label, mat in zip(axes, PROBES, PROBE_LABELS, matrices.values()):
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f"L{l}" for l in LAYERS], fontsize=10)
    ax.set_yticks(range(len(SELECTORS)))
    ax.set_yticklabels(SELECTOR_LABELS, fontsize=9)
    ax.set_title(probe_label, fontsize=12, fontweight="bold", pad=8)

    for i in range(len(SELECTORS)):
        for j in range(len(LAYERS)):
            val = mat[i, j]
            color = "white" if val < 0.5 or val > 2.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=color)

axes[-1].set_xlabel("Layer", fontsize=11)
fig.suptitle("Cohen's d (correct vs incorrect) by token position and layer", fontsize=13, fontweight="bold")
cbar_ax = fig.add_axes([0.15, 0.02, 0.6, 0.015])
fig.colorbar(im, cax=cbar_ax, label="Cohen's d", orientation="horizontal")
plt.subplots_adjust(hspace=0.4, bottom=0.07)
plt.savefig(ASSETS / "plot_031226_assistant_selector_layer_heatmaps.png", dpi=150, bbox_inches="tight")
print(f"Saved to {ASSETS / 'plot_031226_assistant_selector_layer_heatmaps.png'}")
