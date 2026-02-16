import json
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "experiments/probe_science/active_learning_calibration/analysis_1_results.json"
OUT_PATH = "experiments/probe_science/active_learning_calibration/assets/plot_021626_iteration_truncation.png"

with open(DATA_PATH) as f:
    results = json.load(f)

iterations = np.array([r["n_iterations"] for r in results])
rank_corr = np.array([r["rank_correlation_with_full"] for r in results])
pairwise_acc = np.array([r["pairwise_accuracy_mean"] for r in results])
pairwise_std = np.array([r["pairwise_accuracy_std"] for r in results])
cv_r2 = np.array([r["cv_r2_mean"] for r in results])

fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# --- Top panel: rank correlation ---
ax_top.plot(iterations, rank_corr, color="black", marker="o", markersize=5, linewidth=1.5)
ax_top.set_ylabel("Rank correlation with full data")
ax_top.set_ylim(0.85, 1.01)
ax_top.spines["top"].set_visible(False)
ax_top.spines["right"].set_visible(False)

# --- Bottom panel: pairwise accuracy (left) + CV R² (right) ---
color_acc = "#1f77b4"
color_r2 = "#ff7f0e"

ax_bot.errorbar(
    iterations, pairwise_acc, yerr=pairwise_std,
    color=color_acc, marker="o", markersize=5, linewidth=1.5,
    capsize=3, label="Pairwise accuracy",
)
ax_bot.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
ax_bot.set_ylabel("Pairwise accuracy", color=color_acc)
ax_bot.tick_params(axis="y", labelcolor=color_acc)
ax_bot.set_xlabel("Active learning iterations")
ax_bot.spines["top"].set_visible(False)

ax_r2 = ax_bot.twinx()
ax_r2.plot(
    iterations, cv_r2,
    color=color_r2, marker="s", markersize=5, linewidth=1.5,
    label="CV R²",
)
ax_r2.set_ylabel("Probe CV R²", color=color_r2)
ax_r2.tick_params(axis="y", labelcolor=color_r2)
ax_r2.spines["top"].set_visible(False)

# Combine legends from both axes
lines_bot, labels_bot = ax_bot.get_legend_handles_labels()
lines_r2, labels_r2 = ax_r2.get_legend_handles_labels()
ax_bot.legend(lines_bot + lines_r2, labels_bot + labels_r2, loc="lower right", frameon=False)

ax_bot.set_xticks(iterations)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"Saved to {OUT_PATH}")
