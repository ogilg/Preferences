"""Per-topic HOO breakdown for IT vs PT task_mean probes, matching LW post style."""

import json
import numpy as np
import matplotlib.pyplot as plt

# --- Load HOO data ---

def load_per_topic_hoo(path, layer):
    with open(path) as f:
        hoo = json.load(f)
    key = f"ridge_L{layer}"
    result = {}
    for fold in hoo["folds"]:
        if key not in fold.get("layers", {}):
            continue
        topic = fold["held_out_groups"][0]
        entry = fold["layers"][key]
        result[topic] = {"hoo_r": entry["hoo_r"], "n": entry["hoo_n_samples"]}
    return result

it_hoo = load_per_topic_hoo("results/probes/gemma3_10k_hoo_topic_task_mean/hoo_summary.json", 32)
pt_hoo = load_per_topic_hoo("results/probes/gemma3_pt_10k_hoo_topic_task_mean/hoo_summary.json", 31)

# --- Sort by IT-PT gap (descending) ---
topics = sorted(it_hoo.keys(), key=lambda t: it_hoo[t]["hoo_r"] - pt_hoo.get(t, {"hoo_r": 0})["hoo_r"], reverse=True)

# Pretty names
pretty = {
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

it_vals = [it_hoo[t]["hoo_r"] for t in topics]
pt_vals = [pt_hoo[t]["hoo_r"] for t in topics]
ns = [it_hoo[t]["n"] for t in topics]
labels = [f"{pretty.get(t, t)}\n(n={ns[i]})" for i, t in enumerate(topics)]
deltas = [it_vals[i] - pt_vals[i] for i in range(len(topics))]

# --- Colours matching LW post ---
color_it = "#1565C0"   # blue
color_pt = "#78909C"   # grey

# --- Plot ---
fig, ax = plt.subplots(figsize=(14, 5.5))

x = np.arange(len(topics))
width = 0.38

bars_it = ax.bar(x - width / 2, it_vals, width, label="Gemma-3 IT (L32)", color=color_it)
bars_pt = ax.bar(x + width / 2, pt_vals, width, label="Gemma-3 PT (L31)", color=color_pt)

# Delta annotations
for i in range(len(topics)):
    top = max(it_vals[i], pt_vals[i])
    ax.annotate(
        f"\u0394={deltas[i]:.2f}",
        xy=(x[i], top + 0.03),
        ha="center", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8),
    )
    # Bracket lines
    ax.plot([x[i] - width / 2, x[i] - width / 2, x[i] + width / 2, x[i] + width / 2],
            [it_vals[i] + 0.005, top + 0.025, top + 0.025, pt_vals[i] + 0.005],
            color="black", linewidth=0.7)

ax.set_ylabel("Pearson r on held-out topic", fontsize=11)
ax.set_xlabel("Held-out topic", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
ax.legend(loc="upper right", fontsize=10)
ax.set_title("Leave-One-Topic-Out: task_mean probes\nSorted by Instruct\u2013Pre-trained gap", fontsize=12)
ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("experiments/eot_probes/turn_boundary_sweep/assets/plot_031026_per_topic_hoo_task_mean.png", dpi=150, bbox_inches="tight")
print("Saved.")
