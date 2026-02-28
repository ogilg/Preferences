import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "results/experiments/mra_exp2/probes/mra_results.json"
OUTPUT_PATH = "experiments/probe_generalization/multi_role_ablation/assets/plot_022726_diversity_ablation.png"

LAYERS = ["L31", "L43", "L55"]

EVAL_PERSONA_COLORS = {
    "noprompt": "#1f77b4",
    "villain": "#d62728",
    "aesthete": "#9467bd",
    "midwest": "#2ca02c",
}

with open(RESULTS_PATH) as f:
    results = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax_idx, layer in enumerate(LAYERS):
    ax = axes[ax_idx]
    conditions = results["phase2"][layer]["conditions"]

    # Group by n_train_personas
    by_n: dict[int, list[dict]] = defaultdict(list)
    for c in conditions:
        by_n[c["n_train_personas"]].append(c)

    # Plot individual points colored by eval persona
    all_x = []
    all_y = []
    for n in sorted(by_n.keys()):
        entries = by_n[n]
        for e in entries:
            seed_val = abs(hash(str(e["train_personas"]) + e["eval_persona"])) % (2**31)
            jitter = np.random.default_rng(seed_val).uniform(-0.15, 0.15)
            x = n + jitter
            y = e["r2_adjusted"]
            all_x.append(n)
            all_y.append(y)
            ax.scatter(
                x, y,
                color=EVAL_PERSONA_COLORS[e["eval_persona"]],
                alpha=0.4,
                s=40,
                zorder=3,
            )

    # Compute and plot means per n
    mean_x = []
    mean_y = []
    for n in sorted(by_n.keys()):
        vals = [e["r2_adjusted"] for e in by_n[n]]
        mean_x.append(n)
        mean_y.append(np.mean(vals))

    ax.plot(mean_x, mean_y, "k-o", markersize=8, linewidth=2, zorder=5, label="Mean")

    ax.set_title(layer, fontsize=14)
    ax.set_xlabel("")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", alpha=0.3)

    # Vertical dashed line before N=4 to mark the data-size break
    ax.axvline(3.5, color="gray", linestyle="--", alpha=0.4)

    if ax_idx == 0:
        ax.set_ylabel("R² (adjusted) on held-out persona", fontsize=12)

# Legend for eval personas
legend_handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=p)
    for p, c in EVAL_PERSONA_COLORS.items()
]
legend_handles.append(
    plt.Line2D([0], [0], color="k", marker="o", markersize=8, linewidth=2, label="Mean")
)
axes[-1].legend(handles=legend_handles, title="Eval persona", loc="lower right", fontsize=9)

fig.suptitle("Effect of Persona Diversity on Cross-Persona Generalization", fontsize=14, y=1.02)
fig.text(
    0.5, -0.02,
    "Number of training personas (total training data ≈ 2000 tasks for N=1,2,3; N=4 uses ~4000 tasks, unmatched)",
    ha="center", fontsize=11,
)

plt.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT_PATH}")
