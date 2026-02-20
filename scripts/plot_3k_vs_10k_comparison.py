"""Compare 3k vs 10k probe training with matched evaluation methodology."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("experiments/gemma3_10k_probes/scaling_comparison/assets")

LAYERS = [15, 31, 37, 43, 49, 55]

# 3k heldout (4k eval) — r and acc
raw_3k_r = [0.7075, 0.8411, 0.8300, 0.8274, 0.8205, 0.8168]
raw_3k_acc = [0.6818, 0.7487, 0.7436, 0.7358, 0.7378, 0.7310]
dem_3k_r = [0.5271, 0.6986, 0.6652, 0.6536, 0.6414, 0.6452]
dem_3k_acc = [0.6154, 0.6547, 0.6503, 0.6479, 0.6463, 0.6428]

# 10k heldout (4k eval) — r and acc
raw_10k_r = [0.7483, 0.8643, 0.8527, 0.8485, 0.8449, 0.8449]
raw_10k_acc = [0.6983, 0.7684, 0.7540, 0.7520, 0.7507, 0.7502]
dem_10k_r = [0.6016, 0.7609, 0.7378, 0.7285, 0.7158, 0.7206]
dem_10k_acc = [0.6406, 0.6886, 0.6817, 0.6756, 0.6721, 0.6682]

# Derived: R²
raw_3k_r2 = [r**2 for r in raw_3k_r]
raw_10k_r2 = [r**2 for r in raw_10k_r]
dem_3k_r2 = [r**2 for r in dem_3k_r]
dem_10k_r2 = [r**2 for r in dem_10k_r]

# HOO summaries
with open("results/probes/gemma3_3k_hoo_topic_v1/hoo_summary.json") as f:
    hoo_3k = json.load(f)
with open("results/probes/gemma3_10k_hoo_topic/hoo_summary.json") as f:
    hoo_10k = json.load(f)

hoo_layers = sorted(hoo_10k["layer_summary"].keys(), key=int)


def get_hoo_metrics(hoo_data):
    val_r = [hoo_data["layer_summary"][l]["ridge"]["mean_val_r"] for l in hoo_layers]
    hoo_r = [hoo_data["layer_summary"][l]["ridge"]["mean_hoo_r"] for l in hoo_layers]
    hoo_std = [hoo_data["layer_summary"][l]["ridge"]["std_hoo_r"] for l in hoo_layers]
    return val_r, hoo_r, hoo_std


hoo_3k_val, hoo_3k_r, hoo_3k_std = get_hoo_metrics(hoo_3k)
hoo_10k_val, hoo_10k_r, hoo_10k_std = get_hoo_metrics(hoo_10k)


def plot_comparison(y_3k, y_10k, ylabel, title, filename, marker="o", color_10k="#2196F3",
                    ylim=(0, 1.0), annotate=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(LAYERS, y_3k, f"{marker}--", color="#9E9E9E", label="3k train", linewidth=2, markersize=8)
    ax.plot(LAYERS, y_10k, f"{marker}-", color=color_10k, label="10k train", linewidth=2, markersize=8)
    if annotate:
        for i, l in enumerate(LAYERS):
            delta = y_10k[i] - y_3k[i]
            ax.annotate(f"+{delta:.3f}", (l, y_10k[i]), textcoords="offset points",
                        xytext=(8, 5), fontsize=8, color=color_10k)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(*ylim)
    ax.set_xticks(LAYERS)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


# --- Heldout: raw r, R², acc ---
plot_comparison(raw_3k_r, raw_10k_r, "Pearson r (heldout)",
                "Raw Heldout r: 3k vs 10k", "plot_021926_raw_heldout_r.png")
plot_comparison(raw_3k_r2, raw_10k_r2, "R² (heldout)",
                "Raw Heldout R²: 3k vs 10k", "plot_021926_raw_heldout_r2.png")
plot_comparison(raw_3k_acc, raw_10k_acc, "Pairwise accuracy (heldout)",
                "Raw Heldout Accuracy: 3k vs 10k", "plot_021926_raw_heldout_acc.png",
                ylim=(0.5, 0.85))

# --- Heldout: demeaned r, R², acc ---
plot_comparison(dem_3k_r, dem_10k_r, "Pearson r (heldout)",
                "Demeaned Heldout r: 3k vs 10k", "plot_021926_dem_heldout_r.png",
                marker="s", color_10k="#FF9800")
plot_comparison(dem_3k_r2, dem_10k_r2, "R² (heldout)",
                "Demeaned Heldout R²: 3k vs 10k", "plot_021926_dem_heldout_r2.png",
                marker="s", color_10k="#FF9800")
plot_comparison(dem_3k_acc, dem_10k_acc, "Pairwise accuracy (heldout)",
                "Demeaned Heldout Accuracy: 3k vs 10k", "plot_021926_dem_heldout_acc.png",
                marker="s", color_10k="#FF9800", ylim=(0.5, 0.75))

# --- HOO: r across layers ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(LAYERS, hoo_3k_r, yerr=hoo_3k_std, fmt="s--", color="#9E9E9E",
            label="3k HOO r", linewidth=2, markersize=8, capsize=4)
ax.errorbar(LAYERS, hoo_10k_r, yerr=hoo_10k_std, fmt="s-", color="#E91E63",
            label="10k HOO r", linewidth=2, markersize=8, capsize=4)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Pearson r (held-out topic)", fontsize=12)
ax.set_title("HOO Cross-Topic r: 3k vs 10k Training", fontsize=13)
ax.set_ylim(0, 1.0)
ax.set_xticks(LAYERS)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "plot_021926_hoo_layers_comparison.png", dpi=150)
plt.close()
print("Saved hoo_layers_comparison")

# --- HOO: per-topic at L31 ---
fold_3k = {}
for fold in hoo_3k["folds"]:
    group = fold["held_out_groups"][0]
    fold_3k[group] = fold["layers"]["ridge_L31"]["hoo_r"]

fold_10k = {}
for fold in hoo_10k["folds"]:
    group = fold["held_out_groups"][0]
    fold_10k[group] = fold["layers"]["ridge_L31"]["hoo_r"]

topics = sorted(fold_10k.keys(), key=lambda g: fold_10k[g], reverse=True)
r_3k_topics = [fold_3k[t] for t in topics]
r_10k_topics = [fold_10k[t] for t in topics]

x = np.arange(len(topics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width / 2, r_3k_topics, width, color="#9E9E9E", label="3k train")
ax.bar(x + width / 2, r_10k_topics, width, color="#4CAF50", label="10k train")
ax.set_xticks(x)
ax.set_xticklabels(topics, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Pearson r (held-out)", fontsize=12)
ax.set_title("HOO r by Topic (L31): 3k vs 10k Training", fontsize=13)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "plot_021926_hoo_per_topic_comparison.png", dpi=150)
plt.close()
print("Saved hoo_per_topic_comparison")

print("All plots saved to", OUT)
