import matplotlib.pyplot as plt
import numpy as np

# Data from 10-seed robustness test
labels = ["Original\n(n=80)", "Augmented\n(n=160)", "Paraphrase\n(n=80)"]
colors = ["#4477AA", "#44AA77", "#EE7733"]  # blue, green, orange

test_r2_mean = [0.7464, 0.7595, 0.7580]
test_r2_std = [0.1031, 0.0858, 0.0881]

pairwise_acc_mean = [0.8321, 0.8295, 0.8337]
pairwise_acc_std = [0.0370, 0.0445, 0.0472]

cv_r2_mean = [-12.8437, 0.8613, -21.3949]
cv_r2_std = [1.8304, 0.0182, 2.4211]

x = np.arange(len(labels))
bar_width = 0.6

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# --- Panel 1: Test R² ---
ax = axes[0]
bars = ax.bar(x, test_r2_mean, bar_width, yerr=test_r2_std, capsize=5,
              color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Test R²")
ax.set_title("Test R²")
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.axhline(0, color="grey", linewidth=0.5)
for i, (m, s) in enumerate(zip(test_r2_mean, test_r2_std)):
    ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

# --- Panel 2: Pairwise Accuracy ---
ax = axes[1]
bars = ax.bar(x, pairwise_acc_mean, bar_width, yerr=pairwise_acc_std, capsize=5,
              color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Pairwise Accuracy")
ax.set_title("Pairwise Accuracy")
ax.set_ylim(0.5, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.axhline(0.5, color="grey", linewidth=0.5, linestyle="--", label="Chance")
ax.legend(fontsize=8, loc="lower right")
for i, (m, s) in enumerate(zip(pairwise_acc_mean, pairwise_acc_std)):
    ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

# --- Panel 3: CV R² (broken axis style via annotation) ---
ax = axes[2]
bars = ax.bar(x, cv_r2_mean, bar_width, yerr=cv_r2_std, capsize=5,
              color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("CV R²")
ax.set_title("CV R² (cross-validation)")
ax.set_ylim(-25, 2)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.axhline(0, color="grey", linewidth=0.5)
for i, (m, s) in enumerate(zip(cv_r2_mean, cv_r2_std)):
    if m > 0:
        ax.text(i, m + s + 0.5, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    else:
        # Place label inside the bar, near the top (just below zero line)
        ax.text(i, -1.5, f"{m:.1f}", ha="center", va="top", fontsize=8, fontweight="bold")

fig.suptitle("Probe Performance: Paraphrase Augmentation", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()

out_path = "/workspace/repo/experiments/probe_science/paraphrase_augmentation/assets/plot_021826_probe_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
