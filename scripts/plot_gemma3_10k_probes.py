"""Generate plots for gemma3 10k probe training report."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("experiments/gemma3_10k_probes/assets")

LAYERS = [15, 31, 37, 43, 49, 55]

# 10k heldout results (hardcoded from run output)
RAW_R = [0.7483, 0.8643, 0.8527, 0.8485, 0.8449, 0.8449]
DEMEAN_R = [0.6016, 0.7609, 0.7378, 0.7285, 0.7158, 0.7206]

with open("results/probes/gemma3_10k_hoo_topic/hoo_summary.json") as f:
    hoo = json.load(f)

hoo_layers = sorted(hoo["layer_summary"].keys(), key=int)
hoo_val_r = [hoo["layer_summary"][l]["ridge"]["mean_val_r"] for l in hoo_layers]
hoo_hoo_r = [hoo["layer_summary"][l]["ridge"]["mean_hoo_r"] for l in hoo_layers]
hoo_hoo_std = [hoo["layer_summary"][l]["ridge"]["std_hoo_r"] for l in hoo_layers]

fold_data = []
for fold in hoo["folds"]:
    group = fold["held_out_groups"][0]
    metrics = fold["layers"]["ridge_L31"]
    fold_data.append((group, metrics["hoo_r"], metrics["hoo_n_samples"]))
fold_data.sort(key=lambda x: x[1], reverse=True)


# --- Plot 1: Heldout r across layers (raw vs demeaned) ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(LAYERS, RAW_R, "o-", color="#2196F3", label="Raw", linewidth=2, markersize=8)
ax.plot(LAYERS, DEMEAN_R, "s-", color="#FF9800", label="Topic-demeaned", linewidth=2, markersize=8)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Pearson r (heldout)", fontsize=12)
ax.set_title("Gemma3-27B 10k: Heldout Probe Performance by Layer", fontsize=13)
ax.set_ylim(0, 1.0)
ax.set_xticks(LAYERS)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "plot_021926_heldout_r_by_layer.png", dpi=150)
plt.close()
print("Saved heldout_r_by_layer")


# --- Plot 2: HOO val vs hoo r across layers ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(LAYERS, hoo_val_r, "o-", color="#4CAF50", label="In-distribution r", linewidth=2, markersize=8)
ax.errorbar(LAYERS, hoo_hoo_r, yerr=hoo_hoo_std, fmt="s-", color="#E91E63",
            label="Held-out topic r", linewidth=2, markersize=8, capsize=4)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Pearson r", fontsize=12)
ax.set_title("Gemma3-27B 10k: HOO Cross-Topic Generalisation", fontsize=13)
ax.set_ylim(0, 1.0)
ax.set_xticks(LAYERS)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "plot_021926_hoo_generalisation.png", dpi=150)
plt.close()
print("Saved hoo_generalisation")


# --- Plot 3: Per-topic HOO at L31 ---
groups = [f[0] for f in fold_data]
rs = [f[1] for f in fold_data]
ns = [f[2] for f in fold_data]
mean_r = np.mean(rs)
excl_math = [r for g, r in zip(groups, rs) if g != "math"]
mean_excl_math = np.mean(excl_math)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#E91E63" if r < 0.6 else "#FF9800" if r < 0.75 else "#4CAF50" for r in rs]
ax.bar(range(len(groups)), rs, color=colors)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(groups, ns)], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Pearson r (held-out)", fontsize=12)
ax.set_title("Gemma3-27B 10k: HOO r by Topic (L31)", fontsize=13)
ax.set_ylim(0, 1.0)
ax.axhline(y=mean_r, color="gray", linestyle="--", alpha=0.7, label=f"Mean: {mean_r:.3f}")
ax.axhline(y=mean_excl_math, color="#2196F3", linestyle=":", alpha=0.7,
           label=f"Mean excl. math: {mean_excl_math:.3f}")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "plot_021926_hoo_per_topic_L31.png", dpi=150)
plt.close()
print("Saved hoo_per_topic_L31")

print("All plots saved to", OUT)
