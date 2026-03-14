import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4.5, 4))

categories = ["First slot\n(position)", "Second slot\n(label)"]
observed = [98.5, 1.5]
colors = ["#4a90d9", "#e8963e"]

ax.bar(categories, observed, color=colors, edgecolor="white", width=0.55)
for i, v in enumerate(observed):
    ax.text(i, v + 2, f"{v}%", ha="center", fontsize=13, fontweight="bold")

ax.set_ylabel("Patched choice (%)", fontsize=11)
ax.set_title("Label swap: which slot does\nthe model pick after patching?", fontsize=11, fontweight="bold")
ax.set_ylim(0, 115)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

plt.tight_layout()
plt.savefig(
    "docs/logs/assets/plot_031226_label_swap_test.png",
    dpi=200,
    bbox_inches="tight",
)
print("Saved.")
