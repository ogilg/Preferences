"""Plot diversity ablation from MRA 8-persona results (Pearson r, L31).

Filters to 5 target personas: default, villain, aesthete, midwest, sadist.
Conditions: 1x2000, 2x1000, 3x667, 4x500 (leave-one-out from 5 personas).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

RESULTS_PATH = "results/experiments/mra_exp3/probes/mra_8persona_results.json"
OUTPUT_PATH = "docs/lw_post/assets/plot_030426_s5_diversity_ablation.png"

TARGET_PERSONAS = {"noprompt", "villain", "aesthete", "midwest", "sadist"}

PERSONA_COLORS = {
    "noprompt": "#5C6BC0",
    "villain": "#E53935",
    "aesthete": "#8E24AA",
    "midwest": "#43A047",
    "sadist": "#FF6F00",
}

PERSONA_DISPLAY = {
    "noprompt": "Default",
    "villain": "Villain",
    "aesthete": "Aesthete",
    "midwest": "Midwest",
    "sadist": "Sadist",
}

with open(RESULTS_PATH) as f:
    results = json.load(f)

conditions_l31 = results["phase2"]["L31"]["conditions"]

cond_keys = ["1x2000", "2x1000", "3x667", "4x500"]
labels = ["Train on 1 persona\n(2000)", "Train on 2 personas\n(1000 each)", "Train on 3 personas\n(667 each)",
          "Train on 4 personas\n(500 each)"]

# Filter to entries where eval persona is one of our 5 and all train personas
# are from the remaining 4
groups = {k: [] for k in cond_keys}
for entry in conditions_l31:
    if entry["condition"] not in groups:
        continue
    eval_p = entry["eval_persona"]
    train_ps = set(entry["train_personas"])
    if eval_p not in TARGET_PERSONAS:
        continue
    if not train_ps.issubset(TARGET_PERSONAS - {eval_p}):
        continue
    groups[entry["condition"]].append({
        "r": entry["pearson_r"],
        "eval_persona": eval_p,
    })

# Print summary
for label, key in zip(labels, cond_keys):
    vals = [e["r"] for e in groups[key]]
    print(f"{label.replace(chr(10), ' ')}: n={len(vals)}, mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(labels))
rng = np.random.RandomState(42)

# Plot individual points colored by eval persona
for i, key in enumerate(cond_keys):
    entries = groups[key]
    for entry in entries:
        jitter = rng.uniform(-0.15, 0.15)
        color = PERSONA_COLORS[entry["eval_persona"]]
        ax.scatter(i + jitter, entry["r"], color=color, alpha=0.6, s=40, zorder=3,
                   edgecolors="white", linewidths=0.5)

# Compute means and plot connected line
means = [np.mean([e["r"] for e in groups[k]]) for k in cond_keys]
ses = [np.std([e["r"] for e in groups[k]]) / np.sqrt(len(groups[k])) for k in cond_keys]

ax.plot(x, means, "o-", color="black", markersize=8, linewidth=1.5, zorder=4)
ax.errorbar(x, means, yerr=ses, fmt="none", color="black",
            capsize=3, elinewidth=0.8, zorder=4)

# Legend for eval personas
for persona in ["noprompt", "villain", "aesthete", "midwest", "sadist"]:
    color = PERSONA_COLORS[persona]
    ax.scatter([], [], color=color, s=40, label=f"eval: {PERSONA_DISPLAY[persona]}",
               edgecolors="white", linewidths=0.5)
ax.legend(loc="upper left", fontsize=9, frameon=True)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(-0.2, 1.0)
ax.set_ylabel("Pearson r (held-out persona)", fontsize=11)
ax.set_title("Probe performance vs persona diversity (L31)", fontsize=13)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUTPUT_PATH}")
