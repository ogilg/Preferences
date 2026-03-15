import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "results/experiments/mra_exp3/midway_bias/midway_bias_results.json"
OUTPUT_PATH = "experiments/probe_generalization/multi_role_ablation/assets/plot_031526_midway_ratio_per_persona.png"

FOCUS_TOPICS = ["harmful_request", "math", "knowledge_qa", "fiction", "coding", "content_generation"]
NON_DEFAULT_PERSONAS = ["villain", "aesthete", "midwest", "provocateur", "trickster", "autocrat", "sadist"]
SELECTOR = "turn_boundary:-2"

with open(RESULTS_PATH) as f:
    data = json.load(f)

# Filter to selector and OOD entries for non-default personas
filtered = [
    e for e in data
    if e["selector"] == SELECTOR
    and not e["is_in_dist"]
    and e["eval_persona"] in NON_DEFAULT_PERSONAS
]

# For each persona and N, compute per-entry median midway ratio across focus topics,
# then collect all such values for scatter + median line.
# Key: (persona, n_personas) -> list of median_midway_ratios (one per entry)
persona_n_values: dict[tuple[str, int], list[float]] = defaultdict(list)

for entry in filtered:
    persona = entry["eval_persona"]
    n = entry["n_personas"]
    topic_ratios = []
    for topic in FOCUS_TOPICS:
        if topic in entry["topics"]:
            topic_ratios.append(entry["topics"][topic]["midway_ratio"])
    if not topic_ratios:
        continue
    median_ratio = float(np.median(topic_ratios))
    persona_n_values[(persona, n)].append(median_ratio)

fig, axes = plt.subplots(2, 4, figsize=(14, 8))
axes_flat = axes.flatten()

for idx, persona in enumerate(NON_DEFAULT_PERSONAS):
    ax = axes_flat[idx]

    ns_present = sorted(set(n for (p, n) in persona_n_values if p == persona))
    medians = []
    for n in ns_present:
        values = persona_n_values[(persona, n)]
        medians.append(float(np.median(values)))
        # Scatter individual points (clipped for display)
        clipped = np.clip(values, -0.5, 2.0)
        ax.scatter(
            [n] * len(clipped), clipped,
            color="steelblue", alpha=0.15, s=12, zorder=1, edgecolors="none",
        )

    ax.plot(ns_present, np.clip(medians, -0.5, 2.0), "o-", color="steelblue", markersize=5, zorder=2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(-0.5, 2.0)
    ax.set_xlim(0.5, 8.5)
    ax.set_xticks(range(1, 9))
    ax.set_title(persona)
    ax.set_xlabel("N")
    ax.set_ylabel("Median midway ratio")

# Hide last subplot (2x4 grid, 7 personas)
axes_flat[7].set_visible(False)

fig.suptitle("Per-persona midway ratio (OOD, tb:-2)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {OUTPUT_PATH}")
