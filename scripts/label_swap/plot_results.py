import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

matplotlib.rcParams.update({"font.size": 11})

ASSETS = "/workspace/repo/experiments/patching/eot_transfer/label_swap/assets"

# ---------- Plot 1: Four-way classification bar chart ----------

categories = [
    "Full position\n(label + exec \u2192 position)",
    "Full content\n(label + exec \u2192 content)",
    "Dissociation\n(label\u2192pos, exec\u2192content)",
    "Dissociation\n(label\u2192content, exec\u2192pos)",
]
counts = [236, 349, 156, 218]
total = 959
percentages = [c / total * 100 for c in counts]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(categories, counts, color=colors, edgecolor="white", height=0.6)

for bar, count, pct in zip(bars, counts, percentages):
    ax.text(
        bar.get_width() + 5,
        bar.get_y() + bar.get_height() / 2,
        f"{count}  ({pct:.1f}%)",
        va="center",
        fontsize=10,
    )

ax.set_xlim(0, 450)
ax.set_xlabel("Count")
ax.set_title("EOT Label Swap: Four-Way Classification (n=959)")
ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{ASSETS}/plot_030726_four_way_classification.png", dpi=150)
plt.close(fig)
print("Saved plot 1: four_way_classification")

# ---------- Plot 2: 2x2 dissociation matrix ----------

matrix = np.array([[236, 156], [218, 349]])
annot = np.array([
    [f"236\n(24.6%)", f"156\n(16.3%)"],
    [f"218\n(22.7%)", f"349\n(36.4%)"],
])

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    matrix,
    annot=annot,
    fmt="",
    cmap="Blues",
    xticklabels=["Position-following", "Content-following"],
    yticklabels=["Position-following", "Content-following"],
    linewidths=1,
    linecolor="white",
    cbar_kws={"label": "Count"},
    ax=ax,
    vmin=0,
)
ax.set_xlabel("Executed content")
ax.set_ylabel("Stated label")
ax.set_title("Stated Label vs Executed Content")
fig.tight_layout()
fig.savefig(f"{ASSETS}/plot_030726_dissociation_matrix.png", dpi=150)
plt.close(fig)
print("Saved plot 2: dissociation_matrix")
