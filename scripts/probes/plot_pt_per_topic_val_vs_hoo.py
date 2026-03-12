"""Per-topic bar chart: CV r (on training topics) vs HOO r for PT task_mean at L31."""

import json
import numpy as np
import matplotlib.pyplot as plt

with open("results/probes/gemma3_pt_10k_hoo_topic_task_mean/hoo_summary.json") as f:
    hoo = json.load(f)

key = "ridge_L31"
topics_data = {}
for fold in hoo["folds"]:
    if key not in fold.get("layers", {}):
        continue
    topic = fold["held_out_groups"][0]
    entry = fold["layers"][key]
    topics_data[topic] = {
        "val_r": entry["val_r"],
        "hoo_r": entry["hoo_r"],
        "n": entry["hoo_n_samples"],
    }

# Sort by val_r - hoo_r gap (descending)
topics = sorted(topics_data.keys(), key=lambda t: topics_data[t]["val_r"] - topics_data[t]["hoo_r"], reverse=True)

pretty = {
    "coding": "Coding",
    "content_generation": "Content Gen.",
    "fiction": "Fiction",
    "harmful_request": "Harmful Request",
    "knowledge_qa": "Knowledge QA",
    "math": "Math",
    "model_manipulation": "Model Manip.",
    "other": "Other",
    "persuasive_writing": "Persuasive Writing",
    "security_legal": "Security & Legal",
    "sensitive_creative": "Sensitive Creative",
    "summarization": "Summarization",
    "value_conflict": "Value Conflict",
}

val_vals = [topics_data[t]["val_r"] for t in topics]
hoo_vals = [topics_data[t]["hoo_r"] for t in topics]
ns = [topics_data[t]["n"] for t in topics]
labels = [f"{pretty.get(t, t)}\n(n={ns[i]})" for i, t in enumerate(topics)]
deltas = [val_vals[i] - hoo_vals[i] for i in range(len(topics))]

color_val = "#B0BEC5"  # light grey — test set
color_hoo = "#546E7A"  # dark grey — HOO

fig, ax = plt.subplots(figsize=(14, 5.5))

x = np.arange(len(topics))
width = 0.38

bars_val = ax.bar(x - width / 2, val_vals, width, label="CV on training topics", color=color_val)
bars_hoo = ax.bar(x + width / 2, hoo_vals, width, label="Leave-one-topic-out", color=color_hoo)

# Delta annotations
for i in range(len(topics)):
    top = max(val_vals[i], hoo_vals[i])
    ax.annotate(
        f"\u0394={deltas[i]:.2f}",
        xy=(x[i], top + 0.03),
        ha="center", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8),
    )
    ax.plot(
        [x[i] - width / 2, x[i] - width / 2, x[i] + width / 2, x[i] + width / 2],
        [val_vals[i] + 0.005, top + 0.025, top + 0.025, hoo_vals[i] + 0.005],
        color="black", linewidth=0.7,
    )

ax.set_ylabel("Pearson r", fontsize=11)
ax.set_xlabel("Held-out topic", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
ax.legend(loc="upper right", fontsize=10)
ax.set_title("Gemma-3 PT task_mean (L31): CV on Training Topics vs Leave-One-Topic-Out\nSorted by generalisation gap", fontsize=12)
ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(
    "experiments/eot_probes/turn_boundary_sweep/assets/plot_031026_pt_val_vs_hoo_per_topic.png",
    dpi=150, bbox_inches="tight",
)
print("Saved.")
