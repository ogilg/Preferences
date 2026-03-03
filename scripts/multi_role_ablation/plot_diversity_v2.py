"""Plot diversity ablation from MRA v2 results (Pearson r, L31)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

RESULTS_PATH = "results/experiments/mra_exp2/probes_v2/mra_results_v2.json"
OUTPUT_PATH = "docs/lw_post/assets/plot_030226_s5_diversity_ablation.png"

PERSONA_COLORS = {
    "noprompt": "#5C6BC0",
    "villain": "#E53935",
    "aesthete": "#8E24AA",
    "midwest": "#43A047",
}

with open(RESULTS_PATH) as f:
    results = json.load(f)

conditions_l31 = results["phase2"]["L31"]["conditions"]

cond_keys = ["A_1x2000", "B_2x1000", "C_3x667", "D_4x500"]
labels = ["1 persona\n(2000)", "2 personas\n(1000 each)", "3 personas\n(667 each)",
          "4 personas\n(500 each)"]
# Note: A/B/C exclude eval persona from training (OOD); D/E include it (in-distribution)

# Group by condition, keeping eval persona info
groups = {k: [] for k in cond_keys}
for entry in conditions_l31:
    if entry["condition"] not in groups:
        continue
    groups[entry["condition"]].append({
        "r": entry["pearson_r"],
        "eval_persona": entry["eval_persona"],
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
for persona, color in PERSONA_COLORS.items():
    ax.scatter([], [], color=color, s=40, label=f"eval: {persona}", edgecolors="white", linewidths=0.5)
ax.legend(loc="upper left", fontsize=9, frameon=True)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Pearson r (held-out persona)", fontsize=11)
ax.set_title("Persona diversity vs data quantity (L31)", fontsize=13)

# Vertical line separating OOD from in-distribution
ax.axvline(x=2.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.text(1, 0.05, "Train on n, eval on 1", ha="center", fontsize=8, color="gray")
ax.text(3, 0.05, "Train on 4, eval on 4", ha="center", fontsize=8, color="gray")

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUTPUT_PATH}")
