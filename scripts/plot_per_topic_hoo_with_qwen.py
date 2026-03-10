"""Per-topic cross-topic generalisation: Gemma-3 IT, Gemma-3 PT, Qwen3-Embedding-8B.

Similar to plot_022626_per_topic_hoo.png but replaces sentence-transformer with Qwen3-Emb-8B.
Only shows topics present in all three HOO runs (9 of 12).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GEMMA3_IT_HOO = Path("results/probes/gemma3_10k_hoo_topic/hoo_summary.json")
GEMMA3_PT_HOO = Path("results/probes/gemma3_pt_10k_hoo_topic/hoo_summary.json")
QWEN_HOO = Path("results/probes/qwen3_emb_8b_hoo_topic/hoo_summary.json")

OUT_DIR = Path("experiments/probe_science/qwen_embedding/assets")

BEST_LAYERS = {"gemma3_it": 31, "gemma3_pt": 31, "qwen": 0}

COLORS = {"it": "#2978B5", "pt": "#607D8B", "qwen": "#C9A833"}

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
    "value_conflict": "Value Conflict",
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


g3_it, g3_sizes = load_per_topic_hoo(GEMMA3_IT_HOO, BEST_LAYERS["gemma3_it"])
g3_pt, _ = load_per_topic_hoo(GEMMA3_PT_HOO, BEST_LAYERS["gemma3_pt"])
qwen, qwen_sizes = load_per_topic_hoo(QWEN_HOO, BEST_LAYERS["qwen"])

# Only topics present in all three
common_topics = sorted(set(g3_it) & set(g3_pt) & set(qwen))
# Sort by IT-PT delta descending
common_topics.sort(key=lambda t: g3_it[t] - g3_pt[t], reverse=True)

it_vals = [g3_it[t] for t in common_topics]
pt_vals = [g3_pt[t] for t in common_topics]
qwen_vals = [qwen[t] for t in common_topics]
deltas = [g3_it[t] - g3_pt[t] for t in common_topics]

# Label with both sample sizes since they differ
labels = [
    f"$\\bf{{{PRETTY[t].replace(' ', '~')}}}$\n(G:{g3_sizes[t]}, Q:{qwen_sizes[t]})"
    for t in common_topics
]

x = np.arange(len(common_topics))
width = 0.25

fig, ax = plt.subplots(figsize=(17, 7.5))
ax.bar(x - width, it_vals, width, label="Gemma-3 IT (L31)", color=COLORS["it"], edgecolor="black", linewidth=0.5)
ax.bar(x, pt_vals, width, label="Gemma-3 PT (L31)", color=COLORS["pt"], edgecolor="black", linewidth=0.5)
ax.bar(x + width, qwen_vals, width, label="Qwen3-Emb-8B (4096d)", color=COLORS["qwen"], edgecolor="black", linewidth=0.5)

# Bracket + delta annotation between IT and PT bars
for i, delta in enumerate(deltas):
    it_top = it_vals[i]
    pt_top = pt_vals[i]
    x_it = x[i] - width
    x_pt = x[i]
    mid_x = (x_it + x_pt) / 2
    top = max(it_top, pt_top) + 0.02
    bracket_h = 0.015
    ax.plot(
        [x_it, x_it, x_pt, x_pt],
        [it_top + 0.01, top + bracket_h, top + bracket_h, pt_top + 0.01],
        color="#333333", linewidth=0.8,
    )
    ax.text(mid_x, top + bracket_h + 0.01, f"\u0394={delta:.2f}", ha="center", va="bottom", fontsize=7, color="#333333")

ax.set_ylim(0, 1.08)
ax.set_ylabel("Pearson r on held-out topic")
ax.set_xlabel("Held-out topic")
fig.suptitle("Leave-One-Topic-Out Probe Generalization", fontsize=13, y=1.02)
fig.text(
    0.5, 0.98,
    "Sorted by Instruct\u2013Pre-trained gap. G/Q = Gemma/Qwen task counts (different topic ontologies).\n"
    "Missing from Qwen: model_manipulation, security_legal, sensitive_creative.",
    ha="center", va="top", fontsize=9, color="#555555",
)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
plot_path = OUT_DIR / "plot_030926_per_topic_hoo_with_qwen.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {plot_path}")
