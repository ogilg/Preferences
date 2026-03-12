"""Bar chart comparing IT vs PT task_mean probe performance at best layer."""

import json
import numpy as np
import matplotlib.pyplot as plt

# --- Load data ---

def load_hoo_mean_r(path, layers):
    with open(path) as f:
        hoo = json.load(f)
    result = {}
    fold_data = {}
    for layer in layers:
        key = f"ridge_L{layer}"
        rs, ns = [], []
        for fold in hoo["folds"]:
            if key in fold.get("layers", {}):
                rs.append(fold["layers"][key]["hoo_r"])
                ns.append(fold["layers"][key]["hoo_n_samples"])
        rs, ns = np.array(rs), np.array(ns)
        result[layer] = float(np.average(rs, weights=ns))
        fold_data[layer] = {"rs": rs, "ns": ns}
    return result, fold_data

# IT task_mean heldout (best layer = 25)
with open("results/probes/heldout_eval_gemma3_task_mean/manifest.json") as f:
    it_tm = json.load(f)
it_tm_r = {p["layer"]: p["final_r"] for p in it_tm["probes"]}

# PT task_mean heldout (best layer = 31)
with open("results/probes/gemma3_pt_10k_heldout_task_mean/manifest.json") as f:
    pt_tm = json.load(f)
pt_tm_r = {p["layer"]: p["final_r"] for p in pt_tm["probes"]}

# HOO
it_tm_hoo, it_tm_hoo_folds = load_hoo_mean_r("results/probes/gemma3_10k_hoo_topic_task_mean/hoo_summary.json", [25, 32, 39, 46, 53])
pt_tm_hoo, pt_tm_hoo_folds = load_hoo_mean_r("results/probes/gemma3_pt_10k_hoo_topic_task_mean/hoo_summary.json", [15, 31, 37, 43, 49, 55])

# --- Bar data (best layer each) ---
bars = [
    ("Gemma-3 IT\n(L32)", it_tm_r[32], it_tm_hoo[32], it_tm_hoo_folds[32]),
    ("Gemma-3 PT\n(L31)", pt_tm_r[31], pt_tm_hoo[31], pt_tm_hoo_folds[31]),
]

labels = [b[0] for b in bars]
heldout_vals = [b[1] for b in bars]
hoo_vals = [b[2] for b in bars]
hoo_stds = [np.sqrt(np.average((b[3]["rs"] - b[2])**2, weights=b[3]["ns"])) for b in bars]

# --- Colours matching LW post: per-model colours, light=heldout, dark=HOO ---
colors_heldout = ["#90CAF9", "#B0BEC5"]  # IT blue light, PT grey light
colors_hoo = ["#1565C0", "#546E7A"]      # IT blue dark, PT grey dark

# --- Plot ---
fig, ax = plt.subplots(figsize=(5, 5))

x = np.arange(len(labels))
width = 0.35

rects1 = ax.bar(x - width / 2, heldout_vals, width, label="Test set (2,000 utilities)",
                color=colors_heldout, edgecolor="white")
rects2 = ax.bar(x + width / 2, hoo_vals, width, label="Leave-one-topic-out",
                color=colors_hoo, edgecolor="white",
                yerr=hoo_stds, capsize=4, error_kw={"linewidth": 1.2})

ax.set_ylabel("Pearson r", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_title("task_mean probes: Instruct vs Pre-trained", fontsize=12)

# Value labels
for rect in rects1:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=10)
for i, rect in enumerate(rects2):
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, h + hoo_stds[i] + 0.02, f"{h:.2f}",
            ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("experiments/eot_probes/turn_boundary_sweep/assets/plot_031026_it_vs_pt_task_mean_bar.png", dpi=150, bbox_inches="tight")
print("Saved.")
