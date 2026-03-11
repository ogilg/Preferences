"""Violin plot of preference probe scores for true vs false claims."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.probes.core.activations import load_activations
from src.probes.core.evaluate import score_with_probe

ROOT = Path(__file__).resolve().parents[2]

# Paths
raw_acts_path = ROOT / "activations/gemma_3_27b_creak_raw/activations_turn_boundary:-2.npz"
repeat_acts_path = ROOT / "activations/gemma_3_27b_creak_repeat/activations_turn_boundary:-2.npz"
probe_path = ROOT / "results/probes/heldout_eval_gemma3_tb-2/probes/probe_ridge_L32.npy"
labels_path = ROOT / "src/task_data/data/creak.jsonl"
out_path = ROOT / "experiments/truth_probes/assets/plot_031126_truth_probe_score_distributions.png"

# Load probe
probe_weights = np.load(probe_path)

# Load labels
labels_by_id: dict[str, str] = {}
with open(labels_path) as f:
    for line in f:
        row = json.loads(line)
        labels_by_id[row["ex_id"]] = row["label"]

# Load and score both conditions
conditions = {
    "raw": raw_acts_path,
    "repeat": repeat_acts_path,
}

scored: dict[str, dict[str, list[float]]] = {}
for name, path in conditions.items():
    task_ids, layer_acts = load_activations(path, layers=[32])
    scores = score_with_probe(probe_weights, layer_acts[32])

    true_scores = []
    false_scores = []
    for tid, s in zip(task_ids, scores):
        label = labels_by_id[tid]
        if label == "true":
            true_scores.append(float(s))
        else:
            false_scores.append(float(s))

    scored[name] = {"true": true_scores, "false": false_scores}

# Plot
COLOR_FALSE = "#c45c5c"
COLOR_TRUE = "#5c8fb8"

fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150, sharey=True)

panel_info = [
    ("raw", "Raw: claim as user message (d=+0.670)"),
    ("repeat", "Repeat: 'Please say the following statement' (d=+2.180)"),
]

for ax, (name, title) in zip(axes, panel_info):
    false_vals = scored[name]["false"]
    true_vals = scored[name]["true"]

    parts = ax.violinplot(
        [false_vals, true_vals],
        positions=[0, 1],
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLOR_FALSE if i == 0 else COLOR_TRUE)
        body.set_alpha(0.7)

    # Mean lines
    for i, (vals, color) in enumerate(
        [(false_vals, COLOR_FALSE), (true_vals, COLOR_TRUE)]
    ):
        mean = np.mean(vals)
        ax.hlines(mean, i - 0.25, i + 0.25, colors=color, linewidths=2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["False", "True"])
    ax.set_title(title, fontsize=10)

axes[0].set_ylabel("Preference probe score (tb-2, layer 32)")

fig.suptitle("Does the preference direction separate true from false claims?", fontsize=12)
fig.tight_layout()
fig.savefig(out_path)
print(f"Saved to {out_path}")
