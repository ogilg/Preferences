import json
import matplotlib.pyplot as plt
import numpy as np

BASE = "/Users/oscargilg/Dev/MATS/Preferences-gptoss-probes"
ASSETS = f"{BASE}/experiments/gptoss_probes/assets"

# Load data
with open(f"{BASE}/results/probes/gptoss_120b_10k_heldout_std_raw/manifest.json") as f:
    raw = json.load(f)
with open(f"{BASE}/results/probes/gptoss_120b_10k_heldout_std_demeaned/manifest.json") as f:
    demeaned = json.load(f)
with open(f"{BASE}/results/probes/gptoss_120b_10k_hoo_topic/hoo_summary.json") as f:
    hoo = json.load(f)
with open(f"{BASE}/results/probes/gemma3_10k_heldout_std_raw/manifest.json") as f:
    gemma_raw = json.load(f)
with open(f"{BASE}/results/probes/gemma3_10k_hoo_topic/hoo_summary.json") as f:
    gemma_hoo = json.load(f)

# --- Plot 1: Per-layer r (raw vs demeaned) ---
layers = [p["layer"] for p in raw["probes"]]
depths = [l / 36 for l in layers]
raw_r = [p["final_r"] for p in raw["probes"]]
dem_r = [p["final_r"] for p in demeaned["probes"]]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(layers))
width = 0.35
bars1 = ax.bar(x - width / 2, raw_r, width, label="Raw", color="#2E86C1", alpha=0.85)
bars2 = ax.bar(x + width / 2, dem_r, width, label="Topic-demeaned", color="#E67E22", alpha=0.85)
ax.set_xlabel("Layer")
ax.set_ylabel("Heldout Pearson r")
ax.set_title("GPT-OSS-120B: Per-Layer Probe Performance (10k train, ~814 eval)")
ax.set_xticks(x)
ax.set_xticklabels([f"L{l}" for l in layers])
ax.set_ylim(0, 1.0)
ax.legend()
ax.axhline(y=0.3, color="gray", linestyle="--", alpha=0.5, label="Threshold")

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(f"{ASSETS}/plot_022426_layer_r_raw_vs_demeaned.png", dpi=150)
print(f"Saved plot 1")
plt.close()

# --- Plot 2: HOO by topic at L18 (GPT-OSS vs Gemma-3) ---
topics_gptoss = []
hoo_r_gptoss = []
hoo_r_gemma = []
ns = []

# Build per-topic data at L18 for GPT-OSS
gptoss_by_topic = {}
for fold in hoo["folds"]:
    topic = fold["held_out_groups"][0]
    lr = fold["layers"]["ridge_L18"]
    gptoss_by_topic[topic] = (lr["hoo_r"], lr["hoo_n_samples"])

# Gemma-3 at L31 (best)
gemma_by_topic = {}
for fold in gemma_hoo["folds"]:
    topic = fold["held_out_groups"][0]
    lr = fold["layers"]["ridge_L31"]
    gemma_by_topic[topic] = (lr["hoo_r"], lr["hoo_n_samples"])

# Sort by GPT-OSS hoo_r descending
all_topics = sorted(gptoss_by_topic.keys(), key=lambda t: gptoss_by_topic[t][0], reverse=True)

for t in all_topics:
    topics_gptoss.append(t)
    hoo_r_gptoss.append(gptoss_by_topic[t][0])
    ns.append(gptoss_by_topic[t][1])
    hoo_r_gemma.append(gemma_by_topic[t][0] if t in gemma_by_topic else 0)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(topics_gptoss))
width = 0.35
bars1 = ax.bar(x - width / 2, hoo_r_gptoss, width, label="GPT-OSS-120B (L18)", color="#2E86C1", alpha=0.85)
bars2 = ax.bar(x + width / 2, hoo_r_gemma, width, label="Gemma-3-27B (L31)", color="#27AE60", alpha=0.85)

ax.set_xlabel("Topic")
ax.set_ylabel("Held-out Pearson r")
ax.set_title("Cross-Topic Generalisation (HOO): GPT-OSS-120B vs Gemma-3-27B")
ax.set_xticks(x)
labels = [f"{t}\n(n={n})" for t, n in zip(topics_gptoss, ns)]
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylim(0, 1.0)
ax.legend()
ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(f"{ASSETS}/plot_022426_hoo_by_topic.png", dpi=150)
print(f"Saved plot 2")
plt.close()

# --- Plot 3: Layer profile comparison GPT-OSS vs Gemma-3 (raw heldout) ---
gemma_layers = [p["layer"] for p in gemma_raw["probes"]]
gemma_depths = [l / 62 for l in gemma_layers]
gemma_r = [p["final_r"] for p in gemma_raw["probes"]]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(depths, raw_r, "o-", color="#2E86C1", label="GPT-OSS-120B (raw)", linewidth=2, markersize=6)
ax.plot(gemma_depths, gemma_r, "s-", color="#27AE60", label="Gemma-3-27B (raw)", linewidth=2, markersize=6)
ax.set_xlabel("Fractional Depth")
ax.set_ylabel("Heldout Pearson r")
ax.set_title("Probe Performance by Depth: GPT-OSS vs Gemma-3")
ax.set_ylim(0, 1.0)
ax.set_xlim(0, 1.0)
ax.legend()
plt.tight_layout()
plt.savefig(f"{ASSETS}/plot_022426_depth_comparison.png", dpi=150)
print(f"Saved plot 3")
plt.close()
