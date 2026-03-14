import matplotlib.pyplot as plt

labels = ["Genuine\ncontent flip", "Label-only\nflip", "Refusal", "Other"]
counts = [144, 41, 10, 5]
total = sum(counts)
pcts = [100 * c / total for c in counts]
colors = ["#4a90d9", "#e8963e", "#e74c3c", "#aaaaaa"]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, pcts, color=colors, edgecolor="white", linewidth=1.5, width=0.6)

for i, (pct, count) in enumerate(zip(pcts, counts)):
    ax.text(i, pct + 1.5, f"{pct:.0f}%\n({count})", ha="center", fontsize=11, fontweight="bold")

ax.set_ylabel("Share of flipped orderings (%)", fontsize=11)
ax.set_ylim(0, 90)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

plt.tight_layout()
plt.savefig(
    "docs/logs/assets/plot_031226_flip_quality.png",
    dpi=200,
    bbox_inches="tight",
)
print("Saved.")
