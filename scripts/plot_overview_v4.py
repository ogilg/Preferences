import matplotlib.pyplot as plt
import numpy as np

labels = ["Exp 1b\n(Hidden)", "Exp 1c\n(Crossed)", "Exp 1d\n(Competing)"]
x = np.arange(len(labels))
bar_color = "#6675B0"
width = 0.5

r_means = [0.634, 0.768, 0.756]
r_ses = [0.05, 0.02, 0.02]
r_baseline = 0.896

acc_means = [0.660, 0.767, 0.777]
acc_ses = [0.02, 0.02, 0.01]
acc_baseline = 0.77

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: Pearson r
ax1.bar(x, r_means, width, yerr=r_ses, color=bar_color, capsize=4, error_kw={"color": "black", "linewidth": 1.5})
ax1.axhline(r_baseline, color="black", linestyle="--", linewidth=1.2, label="Baseline probe (r = 0.90)")
ax1.set_ylabel("Pearson r", fontsize=12)
ax1.set_title("Pearson r", fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11)
ax1.set_ylim(0, 1.0)
ax1.legend(fontsize=10, loc="upper right")

# Right panel: Pairwise accuracy
ax2.bar(x, acc_means, width, yerr=acc_ses, color=bar_color, capsize=4, error_kw={"color": "black", "linewidth": 1.5})
ax2.axhline(acc_baseline, color="black", linestyle="--", linewidth=1.2, label="Baseline probe (acc = 0.77)")
ax2.set_ylabel("Pairwise accuracy", fontsize=12)
ax2.set_title("Pairwise accuracy", fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=11)
ax2.set_ylim(0.5, 1.0)
ax2.legend(fontsize=10, loc="upper right")

fig.suptitle("Condition probe performance across experiments", fontsize=16, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out = "/Users/oscargilg/Dev/MATS/Preferences/experiments/ood_system_prompts/utility_fitting/assets/plot_022828_overview_v4.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved to {out}")
