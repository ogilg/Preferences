import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUTPUT_DIR = "experiments/probe_generalization/token_selection/assets"

# --- Final metrics ---
layers = [31, 43, 55]
layer_labels = ["L31", "L43", "L55"]

prompt_last_r   = [0.8411, 0.8274, 0.8168]
prompt_last_acc = [0.7487, 0.7358, 0.7310]

prompt_mean_r   = [0.7109, 0.6761, 0.6610]
prompt_mean_acc = [0.7068, 0.6934, 0.6887]

# --- Alpha sweep data (exact values from JSON) ---
# prompt_last sweep uses the provided values
pl_sweep = {
    31: [(0.1, 0.6727), (0.464, 0.6992), (2.154, 0.7143), (10, 0.7379),
         (46.4, 0.7787), (215.4, 0.8193), (1000, 0.8402), (4641.6, 0.8380),
         (21544.3, 0.8171), (100000, 0.7768)],
    43: [(0.1, 0.6729), (0.464, 0.6979), (2.154, 0.7113), (10, 0.7301),
         (46.4, 0.7663), (215.4, 0.8025), (1000, 0.8199), (4641.6, 0.8153),
         (21544.3, 0.7896), (100000, 0.7442)],
    55: [(0.1, 0.6522), (0.464, 0.6737), (2.154, 0.6876), (10, 0.7047),
         (46.4, 0.7422), (215.4, 0.7881), (1000, 0.8132), (4641.6, 0.8111),
         (21544.3, 0.7856), (100000, 0.7379)],
}

# prompt_mean sweep — exact values from heldout_eval.json
pm_sweep = {
    31: [(0.1,     0.22848815830448402),
         (0.46416, 0.27277443526219536),
         (2.15443, 0.37796041144313675),
         (10.0,    0.5107712315538141),
         (46.4159, 0.6146027522531788),
         (215.443, 0.674217878330592),
         (1000.0,  0.6971658229960296),
         (4641.59, 0.6855182553063378),
         (21544.3, 0.6451074306166819),
         (100000,  0.5575204179254627)],
    43: [(0.1,     0.28383036038952847),
         (0.46416, 0.3313854745557617),
         (2.15443, 0.41795055798750314),
         (10.0,    0.5169070122082938),
         (46.4159, 0.5931344154120447),
         (215.443, 0.6447271352185335),
         (1000.0,  0.66614790530473),
         (4641.59, 0.6545114932229182),
         (21544.3, 0.5972132919698604),
         (100000,  0.4852825535822595)],
    55: [(0.1,     0.29954244991137424),
         (0.46416, 0.343144613730453),
         (2.15443, 0.42544052167269236),
         (10.0,    0.5147766215464749),
         (46.4159, 0.5903188055524093),
         (215.443, 0.6406396742860548),
         (1000.0,  0.6573660980827597),
         (4641.59, 0.6403885782093705),
         (21544.3, 0.5819840832260691),
         (100000,  0.4764150600798343)],
}

LAYER_COLORS = {31: "#1f77b4", 43: "#ff7f0e", 55: "#2ca02c"}

# ============================================================
# Plot 1: Side-by-side grouped bar chart
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Token Selection: prompt_last vs prompt_mean", fontsize=14, fontweight="bold")

x = np.arange(len(layer_labels))
bar_width = 0.35

for ax, last_vals, mean_vals, ylabel, title_suffix in [
    (axes[0], prompt_last_r,   prompt_mean_r,   "Pearson r",        "Pearson r by Layer"),
    (axes[1], prompt_last_acc, prompt_mean_acc, "Pairwise Accuracy", "Pairwise Accuracy by Layer"),
]:
    bars_last = ax.bar(x - bar_width / 2, last_vals, bar_width,
                       label="prompt_last", color="#1f77b4", alpha=0.85)
    bars_mean = ax.bar(x + bar_width / 2, mean_vals, bar_width,
                       label="prompt_mean", color="#ff7f0e", alpha=0.85)

    # Value labels on bars
    for bar in bars_last:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)
    for bar in bars_mean:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title_suffix)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()
out1 = f"{OUTPUT_DIR}/plot_021826_final_metrics_comparison.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig)

# ============================================================
# Plot 2: Alpha sweep curves
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title("Alpha Sweep: prompt_last vs prompt_mean", fontsize=13, fontweight="bold")

for layer in layers:
    color = LAYER_COLORS[layer]

    pl_alphas = [pt[0] for pt in pl_sweep[layer]]
    pl_rs     = [pt[1] for pt in pl_sweep[layer]]
    ax.plot(pl_alphas, pl_rs, color=color, linestyle="-",  linewidth=1.8,
            marker="o", markersize=4, label=f"prompt_last  L{layer}")

    pm_alphas = [pt[0] for pt in pm_sweep[layer]]
    pm_rs     = [pt[1] for pt in pm_sweep[layer]]
    ax.plot(pm_alphas, pm_rs, color=color, linestyle="--", linewidth=1.8,
            marker="s", markersize=4, label=f"prompt_mean L{layer}")

ax.set_xscale("log")
ax.set_xlabel("Alpha (regularization)")
ax.set_ylabel("Sweep Pearson r (CV)")
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.legend(fontsize=8.5, ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.4)

fig.tight_layout()
out2 = f"{OUTPUT_DIR}/plot_021826_alpha_sweep_comparison.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig)
