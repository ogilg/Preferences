import matplotlib.pyplot as plt
import numpy as np

# Gemma-2 27B Base, L23
alphas = [1.0, 4.64, 21.5, 100, 464, 2154, 10000, 46416, 215443, 1000000]

g2_train = [0.9999, 0.9998, 0.9988, 0.9928, 0.9711, 0.9257, 0.8598, 0.7864, 0.7147, 0.6619]
g2_val = [0.585, 0.609, 0.647, 0.701, 0.759, 0.789, 0.782, 0.737, 0.637, 0.408]

# Gemma-3 27B IT, L31
g3_train = [0.9999, 0.9997, 0.9978, 0.9895, 0.9694, 0.9367, 0.8962, 0.8512, 0.7983, 0.7472]
g3_val = [0.727, 0.742, 0.773, 0.818, 0.851, 0.863, 0.854, 0.824, 0.765, 0.630]

# Best alphas (highest val R²)
g2_best_idx = int(np.argmax(g2_val))
g3_best_idx = int(np.argmax(g3_val))

fig, ax = plt.subplots(figsize=(8, 5))

# Gemma-2 Base
ax.plot(alphas, g2_train, "--", color="#2196F3", alpha=0.6, label="Gemma-2 27B Base L23 — train")
ax.plot(alphas, g2_val, "-", color="#2196F3", label="Gemma-2 27B Base L23 — val")
ax.plot(
    alphas[g2_best_idx], g2_val[g2_best_idx], "o",
    color="#2196F3", markersize=9, zorder=5,
    label=f"Best α={alphas[g2_best_idx]:.0f} (val R²={g2_val[g2_best_idx]:.3f})",
)

# Gemma-3 IT
ax.plot(alphas, g3_train, "--", color="#E91E63", alpha=0.6, label="Gemma-3 27B IT L31 — train")
ax.plot(alphas, g3_val, "-", color="#E91E63", label="Gemma-3 27B IT L31 — val")
ax.plot(
    alphas[g3_best_idx], g3_val[g3_best_idx], "o",
    color="#E91E63", markersize=9, zorder=5,
    label=f"Best α={alphas[g3_best_idx]:.0f} (val R²={g3_val[g3_best_idx]:.3f})",
)

ax.set_xscale("log")
ax.set_xlabel("Ridge α (log scale)")
ax.set_ylabel("R²")
ax.set_title("Alpha Sweep: Gemma-2 27B Base (L23) vs Gemma-3 27B IT (L31)")
ax.legend(fontsize=8, loc="lower left")
ax.grid(True, alpha=0.3)

fig.tight_layout()
out = "/workspace/repo/experiments/probe_science/content_orthogonal/gemma2base/base_probes/assets/plot_021726_alpha_sweep_comparison.png"
fig.savefig(out, dpi=200)
print(f"Saved to {out}")
