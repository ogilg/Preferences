import json
import matplotlib.pyplot as plt
import numpy as np

# Load analysis results
with open("experiments/probe_generalization/persona_ood/phase3/analysis_results.json") as f:
    results = json.load(f)

# Load persona groupings
with open("experiments/probe_generalization/persona_ood/v2_config.json") as f:
    v2_config = json.load(f)

with open("experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json") as f:
    enriched_prompts = json.load(f)

original_names = {p["name"] for p in v2_config["personas"] if p["part"] == "A"}
enriched_names = set(enriched_prompts.keys())

per_persona = results["demean/ridge_L31"]["per_persona"]

# Build sorted list of (name, r, group)
entries = []
for name, stats in per_persona.items():
    if name in original_names:
        group = "original"
    elif name in enriched_names:
        group = "enriched"
    else:
        raise ValueError(f"Unknown persona: {name}")
    entries.append((name, stats["r"], group))

entries.sort(key=lambda x: x[1], reverse=True)

names = [e[0] for e in entries]
r_values = [e[1] for e in entries]
colors = ["#1f77b4" if e[2] == "original" else "#ff7f0e" for e in entries]

fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

y_pos = np.arange(len(names))
ax.barh(y_pos, r_values, color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.invert_yaxis()
ax.set_xlabel("Pearson r")
ax.set_xlim(0, max(r_values) * 1.1)
ax.set_title("Per-persona probe-behavior correlation (demean/ridge_L31)")

# Threshold line
ax.axvline(x=0.2, color="red", linestyle="--", linewidth=1, label="r = 0.2 threshold")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#1f77b4", label="Original"),
    Patch(facecolor="#ff7f0e", label="Enriched"),
    plt.Line2D([0], [0], color="red", linestyle="--", label="r = 0.2 threshold"),
]
ax.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()
plt.savefig(
    "experiments/probe_generalization/persona_ood/phase3/assets/plot_021826_per_persona_correlation.png"
)
print("Saved plot.")
