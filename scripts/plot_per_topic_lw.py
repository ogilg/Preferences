"""Per-topic cross-topic generalisation breakdown for LW post.

Grouped bar chart: one group per topic, three bars (IT, PT, sentence-transformer).
Sorted by IT-PT delta (descending). Delta annotated above each group.
All numbers loaded from HOO summary JSONs.

Colour scheme (shared with plot_cross_topic_lw.py):
  IT: blue (#2978B5)
  PT: slate (#607D8B)
  ST: amber (#C9A833)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GEMMA3_IT_HOO = Path("results/probes/gemma3_10k_hoo_topic/hoo_summary.json")
GEMMA3_PT_HOO = Path("results/probes/gemma3_pt_10k_hoo_topic/hoo_summary.json")
ST_HOO = Path("results/probes/st_10k_hoo_topic/hoo_summary.json")

LW_ASSETS = Path("docs/lw_post/assets")

BEST_LAYERS = {"gemma3_it": 31, "gemma3_pt": 31, "st": 0}

COLORS = {"it": "#2978B5", "pt": "#607D8B", "st": "#C9A833"}

PRETTY = {
    "coding": "Coding",
    "content_generation": "Content Generation",
    "fiction": "Fiction",
    "harmful_request": "Harmful Request",
    "knowledge_qa": "Knowledge QA",
    "math": "Math",
    "model_manipulation": "Model Manipulation",
    "other": "Other",
    "persuasive_writing": "Persuasive Writing",
    "security_legal": "Security & Legal",
    "sensitive_creative": "Sensitive Creative",
    "summarization": "Summarization",
}


def load_per_topic_hoo(path: Path, layer: int):
    with open(path) as f:
        summary = json.load(f)
    key = f"ridge_L{layer}"
    result = {}
    for fold in summary["folds"]:
        topic = fold["held_out_groups"][0]
        metrics = fold["layers"][key]
        result[topic] = metrics["hoo_r"]
    return result, summary["group_sizes"]


g3_it, group_sizes = load_per_topic_hoo(GEMMA3_IT_HOO, BEST_LAYERS["gemma3_it"])
g3_pt, _ = load_per_topic_hoo(GEMMA3_PT_HOO, BEST_LAYERS["gemma3_pt"])
st, _ = load_per_topic_hoo(ST_HOO, BEST_LAYERS["st"])

# Sort topics by IT-PT delta (descending)
topics = sorted(g3_it.keys(), key=lambda t: g3_it[t] - g3_pt[t], reverse=True)

it_vals = [g3_it[t] for t in topics]
pt_vals = [g3_pt[t] for t in topics]
st_vals = [st[t] for t in topics]
deltas = [g3_it[t] - g3_pt[t] for t in topics]

labels = [f"$\\bf{{{PRETTY.get(t, t).replace(' ', '~')}}}$ (n={group_sizes[t]})" for t in topics]

x = np.arange(len(topics))
width = 0.25

fig, ax = plt.subplots(figsize=(17, 7.5))
ax.bar(x - width, it_vals, width, label="Gemma-3 IT", color=COLORS["it"], edgecolor="black", linewidth=0.5)
ax.bar(x, pt_vals, width, label="Gemma-3 PT", color=COLORS["pt"], edgecolor="black", linewidth=0.5)
ax.bar(x + width, st_vals, width, label="Sentence-transf.", color=COLORS["st"], edgecolor="black", linewidth=0.5)

# Bracket + delta annotation between IT and PT bars
for i, delta in enumerate(deltas):
    it_top = it_vals[i]
    pt_top = pt_vals[i]
    x_it = x[i] - width
    x_pt = x[i]
    mid_x = (x_it + x_pt) / 2
    top = max(it_top, pt_top) + 0.02
    bracket_h = 0.015
    # Horizontal line with vertical ticks
    ax.plot([x_it, x_it, x_pt, x_pt], [it_top + 0.01, top + bracket_h, top + bracket_h, pt_top + 0.01],
            color="#333333", linewidth=0.8)
    ax.text(mid_x, top + bracket_h + 0.01, f"Δ={delta:.2f}", ha="center", va="bottom", fontsize=7, color="#333333")

ax.set_ylim(0, 1.08)
ax.set_ylabel("Pearson r on held-out topic")
ax.set_xlabel("Held-out topic")
fig.suptitle("Leave-One-Topic-Out Probe Generalization",
             fontsize=13, y=1.02)
fig.text(0.5, 0.98, "Probe trained on 11 topics, evaluated on the held-out topic. Sorted by Instruct–Pre-trained gap.",
         ha="center", va="top", fontsize=10, color="#555555")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
LW_ASSETS.mkdir(parents=True, exist_ok=True)
plot_path = LW_ASSETS / "plot_022626_per_topic_hoo.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {plot_path}")
