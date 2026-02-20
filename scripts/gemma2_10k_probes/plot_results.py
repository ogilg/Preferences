import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ASSETS_DIR = "experiments/gemma2_10k_probes/assets"

# --- Plot 1: Heldout r by layer ---

gemma2_layers = [11, 23, 27, 32, 36, 41]
gemma2_total_layers = 46
gemma2_r = [0.7103, 0.7672, 0.7615, 0.7402, 0.7319, 0.7310]

gemma3_layers = [15, 31, 37, 43, 49, 55]
gemma3_total_layers = 62
gemma3_r = [0.7483, 0.8643, 0.8527, 0.8485, 0.8449, 0.8449]

gemma2_frac = [l / gemma2_total_layers for l in gemma2_layers]
gemma3_frac = [l / gemma3_total_layers for l in gemma3_layers]

gemma2_best_idx = int(np.argmax(gemma2_r))
gemma3_best_idx = int(np.argmax(gemma3_r))

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(gemma2_frac, gemma2_r, color="steelblue", marker="o", label="Gemma-2 base 10k")
ax.plot(gemma3_frac, gemma3_r, color="darkorange", marker="o", label="Gemma-3 IT 10k")

ax.plot(
    gemma2_frac[gemma2_best_idx],
    gemma2_r[gemma2_best_idx],
    marker="*",
    markersize=14,
    color="steelblue",
    zorder=5,
    linestyle="none",
)
ax.plot(
    gemma3_frac[gemma3_best_idx],
    gemma3_r[gemma3_best_idx],
    marker="*",
    markersize=14,
    color="darkorange",
    zorder=5,
    linestyle="none",
)

ax.set_xlabel("Fractional Depth (layer / total layers)")
ax.set_ylabel("Heldout r")
ax.set_title("Heldout Performance by Layer (10k Training)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f"{ASSETS_DIR}/plot_021926_heldout_r_by_layer.png", dpi=150)
plt.close(fig)
print(f"Saved: {ASSETS_DIR}/plot_021926_heldout_r_by_layer.png")

# --- Plot 2: Dataset HOO ---

hoo_layers = [11, 23, 27, 32, 36, 41]
hoo_mean_r = [0.3121, 0.3529, 0.3234, 0.2739, 0.2827, 0.2997]
hoo_best_layer_label = "L23"

datasets = ["alpaca", "bailbench", "competition_math", "stresstest", "wildchat"]
dataset_r_at_best = [0.4101, 0.4538, 0.1672, 0.3967, 0.3368]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: mean HOO r by layer
ax1.plot(hoo_layers, hoo_mean_r, color="steelblue", marker="o")
best_hoo_idx = int(np.argmax(hoo_mean_r))
ax1.plot(
    hoo_layers[best_hoo_idx],
    hoo_mean_r[best_hoo_idx],
    marker="*",
    markersize=14,
    color="steelblue",
    zorder=5,
    linestyle="none",
)
ax1.set_xlabel("Layer")
ax1.set_ylabel("Mean HOO r")
ax1.set_title("Mean HOO r by Layer (Gemma-2 base 10k)")
ax1.set_ylim(0, 1.0)
ax1.grid(True, alpha=0.3)

# Right: per-dataset HOO r at best layer
x = np.arange(len(datasets))
bars = ax2.bar(x, dataset_r_at_best, color="steelblue", alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, rotation=20, ha="right")
ax2.set_ylabel("HOO r")
ax2.set_title(f"Per-Dataset HOO r at {hoo_best_layer_label}")
ax2.set_ylim(0, 1.0)
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle("Cross-Dataset Generalization (Dataset HOO, L23 best)")
fig.tight_layout()
fig.savefig(f"{ASSETS_DIR}/plot_021926_dataset_hoo.png", dpi=150)
plt.close(fig)
print(f"Saved: {ASSETS_DIR}/plot_021926_dataset_hoo.png")
