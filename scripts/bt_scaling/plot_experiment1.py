import json
import numpy as np
import matplotlib.pyplot as plt

with open("experiments/probe_science/bt_scaling/experiment1_results.json") as f:
    results = json.load(f)

# --- Plot 1: BT Lambda Sweep ---
fig, ax = plt.subplots(figsize=(10, 6))

for variant, color, label in [
    ("bt_standard", "tab:blue", "BT standard"),
    ("bt_scaled", "tab:orange", "BT scaled"),
]:
    sweep = results[variant]["sweep"]
    lambdas = [s["l2_lambda"] for s in sweep]
    val_acc = [s["val_acc"] for s in sweep]
    train_acc = [s["train_acc"] for s in sweep]
    best_lambda = results[variant]["best_hp"]

    ax.plot(lambdas, val_acc, "-", color=color, label=f"{label} val")
    ax.plot(lambdas, train_acc, "--", color=color, label=f"{label} train")
    ax.axvline(best_lambda, color=color, linestyle=":", alpha=0.7, label=f"{label} best λ={best_lambda:.2f}")

ax.set_xscale("log")
ax.set_xlabel("L2 Lambda")
ax.set_ylabel("Accuracy")
ax.set_title("BT Lambda Sweep (Fold 0)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("experiments/probe_science/bt_scaling/assets/plot_021626_bt_lambda_sweep.png", dpi=150)
plt.close(fig)
print("Saved plot 1")

# --- Plot 2: Summary bar chart ---
variant_configs = [
    ("ridge_thurstonian", "Ridge Thurstonian", "tab:blue"),
    ("ridge_winrate", "Ridge win-rate", "lightblue"),
    ("bt_standard", "BT standard", "tab:orange"),
    ("bt_scaled", "BT scaled", "tab:red"),
]

means = []
stderrs = []
labels = []
colors = []

for key, label, color in variant_configs:
    accs = [f["test_acc"] for f in results[key]["folds"]]
    means.append(np.mean(accs))
    stderrs.append(np.std(accs, ddof=1) / np.sqrt(len(accs)))
    labels.append(label)
    colors.append(color)

fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(labels))
bars = ax.bar(x, means, yerr=stderrs, capsize=5, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(0.5, color="black", linestyle="--", alpha=0.5, label="Chance")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.65, 0.78)
ax.set_ylabel("Held-out Pairwise Accuracy")
ax.set_title("Held-out Pairwise Accuracy by Method")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{mean:.4f}",
            ha="center", va="bottom", fontsize=10)

fig.tight_layout()
fig.savefig("experiments/probe_science/bt_scaling/assets/plot_021626_regularization_summary.png", dpi=150)
plt.close(fig)
print("Saved plot 2")
