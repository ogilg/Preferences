import json
import os

import matplotlib.pyplot as plt
import numpy as np

ASSETS_DIR = "experiments/gemma2_10k_probes/assets"
GEMMA2_HOO_PATH = "results/probes/gemma2_10k_hoo_topic/hoo_summary.json"
GEMMA3_HOO_PATH = "results/probes/gemma3_10k_hoo_topic/hoo_summary.json"

os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Plot 1: Heldout r by layer ────────────────────────────────────────────────

g2_depths = [11/46, 23/46, 27/46, 32/46, 36/46, 41/46]
g2_raw_r  = [0.710, 0.767, 0.762, 0.740, 0.732, 0.731]
g2_dem_r  = [0.548, 0.610, 0.610, 0.571, 0.563, 0.566]

g3_depths = [15/62, 31/62, 37/62, 43/62, 49/62, 55/62]
g3_raw_r  = [0.748, 0.864, 0.853, 0.849, 0.845, 0.845]
g3_dem_r  = [0.602, 0.761, 0.738, 0.729, 0.716, 0.721]

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(g2_depths, g2_raw_r, color="steelblue",  linestyle="-",  linewidth=2, label="Gemma-2 raw")
ax.plot(g2_depths, g2_dem_r, color="steelblue",  linestyle="--", linewidth=2, label="Gemma-2 demeaned")
ax.plot(g3_depths, g3_raw_r, color="darkorange", linestyle="-",  linewidth=2, label="Gemma-3 IT raw")
ax.plot(g3_depths, g3_dem_r, color="darkorange", linestyle="--", linewidth=2, label="Gemma-3 IT demeaned")

# Stars on best layer per model per condition
best_g2_raw_idx = int(np.argmax(g2_raw_r))
best_g2_dem_idx = int(np.argmax(g2_dem_r))
best_g3_raw_idx = int(np.argmax(g3_raw_r))
best_g3_dem_idx = int(np.argmax(g3_dem_r))

ax.plot(g2_depths[best_g2_raw_idx], g2_raw_r[best_g2_raw_idx], "*", color="steelblue",  markersize=14, zorder=5)
ax.plot(g2_depths[best_g2_dem_idx], g2_dem_r[best_g2_dem_idx], "*", color="steelblue",  markersize=14, zorder=5)
ax.plot(g3_depths[best_g3_raw_idx], g3_raw_r[best_g3_raw_idx], "*", color="darkorange", markersize=14, zorder=5)
ax.plot(g3_depths[best_g3_dem_idx], g3_dem_r[best_g3_dem_idx], "*", color="darkorange", markersize=14, zorder=5)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Fractional Layer Depth")
ax.set_ylabel("Heldout r")
ax.set_title("Heldout Performance by Layer")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

fig.tight_layout()
plot1_path = os.path.join(ASSETS_DIR, "plot_021926_heldout_r_by_layer.png")
fig.savefig(plot1_path, dpi=150)
plt.close(fig)
print(f"Saved: {plot1_path}")

# ── Plot 2: Cross-topic HOO comparison ───────────────────────────────────────

with open(GEMMA2_HOO_PATH) as f:
    g2_hoo = json.load(f)

with open(GEMMA3_HOO_PATH) as f:
    g3_hoo = json.load(f)

# Mean HOO r per layer from layer_summary
g2_layers_int = [11, 23, 27, 32, 36, 41]
g2_layer_depths = [l / 46 for l in g2_layers_int]
g2_mean_hoo = [g2_hoo["layer_summary"][str(l)]["ridge"]["mean_hoo_r"] for l in g2_layers_int]

g3_layers_int = [15, 31, 37, 43, 49, 55]
g3_layer_depths = [l / 62 for l in g3_layers_int]
g3_mean_hoo = [g3_hoo["layer_summary"][str(l)]["ridge"]["mean_hoo_r"] for l in g3_layers_int]

# Per-topic HOO r at best layer: Gemma-2 L23, Gemma-3 L31
def extract_per_topic_hoo(hoo_data, layer_key):
    result = {}
    for fold in hoo_data["folds"]:
        groups = fold["held_out_groups"]
        if layer_key in fold["layers"]:
            hoo_r = fold["layers"][layer_key]["hoo_r"]
            for g in groups:
                result[g] = hoo_r
    return result

g2_topic_hoo = extract_per_topic_hoo(g2_hoo, "ridge_L23")
g3_topic_hoo = extract_per_topic_hoo(g3_hoo, "ridge_L31")

all_topics = sorted(set(g2_topic_hoo) & set(g3_topic_hoo))
# Sort by Gemma-3 r descending
all_topics.sort(key=lambda t: g3_topic_hoo[t], reverse=True)

g2_vals = [g2_topic_hoo[t] for t in all_topics]
g3_vals = [g3_topic_hoo[t] for t in all_topics]

topic_labels = [t.replace("_", "\n") for t in all_topics]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: mean HOO r by fractional depth
ax1.plot(g2_layer_depths, g2_mean_hoo, color="steelblue",  linestyle="-", linewidth=2, marker="o", label="Gemma-2")
ax1.plot(g3_layer_depths, g3_mean_hoo, color="darkorange", linestyle="-", linewidth=2, marker="o", label="Gemma-3 IT")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel("Fractional Layer Depth")
ax1.set_ylabel("Mean Topic HOO r")
ax1.set_title("Mean Topic HOO r by Layer")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: per-topic grouped bars
x = np.arange(len(all_topics))
bar_width = 0.35
ax2.bar(x - bar_width / 2, g2_vals, bar_width, color="steelblue",  label="Gemma-2 L23", alpha=0.85)
ax2.bar(x + bar_width / 2, g3_vals, bar_width, color="darkorange", label="Gemma-3 L31", alpha=0.85)

ax2.set_ylim(0, 1)
ax2.set_xticks(x)
ax2.set_xticklabels(topic_labels, fontsize=8)
ax2.set_ylabel("HOO r")
ax2.set_title("Per-Topic HOO r (Gemma-2 L23 vs Gemma-3 L31)")
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plot2_path = os.path.join(ASSETS_DIR, "plot_021926_hoo_topic.png")
fig.savefig(plot2_path, dpi=150)
plt.close(fig)
print(f"Saved: {plot2_path}")
