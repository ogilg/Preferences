import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DATA_PATH = "/workspace/repo/experiments/steering/program/open_ended_effects/generalization_new_prompts/analysis_results.json"
ASSETS_DIR = "/workspace/repo/experiments/steering/program/open_ended_effects/generalization_new_prompts/assets"

CATEGORY_COLORS = {
    "self_report": "#3274A1",
    "affect": "#C44E52",
    "meta_cognitive": "#8172B3",
    "task_completion": "#4C9F70",
    "neutral": "#999999",
}

CATEGORY_ORDER = ["self_report", "affect", "meta_cognitive", "task_completion", "neutral"]
CATEGORY_LABELS = {
    "self_report": "Self-report",
    "affect": "Affect",
    "meta_cognitive": "Meta-cognitive",
    "task_completion": "Task completion",
    "neutral": "Neutral",
}

DIMENSION_LABELS = {
    "emotional_engagement": "Engagement",
    "hedging": "Hedging",
    "elaboration": "Elaboration",
    "confidence": "Confidence",
}

with open(DATA_PATH) as f:
    data = json.load(f)

combined = data["combined_3000"]


# --- Plot 1: Per-prompt bar charts for engagement and confidence ---

fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=False)

for ax, dim_key, dim_label in [
    (axes[0], "emotional_engagement", "Engagement"),
    (axes[1], "confidence", "Confidence"),
]:
    details = combined[dim_key]["details"]
    # Sort by diff
    details_sorted = sorted(details, key=lambda d: d["diff"])

    prompt_ids = [d["prompt_id"] for d in details_sorted]
    diffs = [d["diff"] for d in details_sorted]
    colors = [CATEGORY_COLORS[d["category"]] for d in details_sorted]

    y_pos = np.arange(len(prompt_ids))
    ax.barh(y_pos, diffs, color=colors, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(prompt_ids, fontsize=8)
    ax.set_xlabel("Direction asymmetry (positive - negative steering)")
    ax.set_title(dim_label, fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Legend (shared)
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=CATEGORY_COLORS[cat], label=CATEGORY_LABELS[cat])
    for cat in CATEGORY_ORDER
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=5,
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle("Direction asymmetry by prompt (combined \u00b13000)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(
    f"{ASSETS_DIR}/plot_021426_direction_asymmetry.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)
print("Saved plot 1: direction asymmetry bar chart")


# --- Plot 2: Category heatmap ---

dimensions = ["emotional_engagement", "hedging", "elaboration", "confidence"]
dim_labels = [DIMENSION_LABELS[d] for d in dimensions]
cat_labels = [CATEGORY_LABELS[c] for c in CATEGORY_ORDER]

# Build matrix: rows=categories, cols=dimensions
matrix = np.zeros((len(CATEGORY_ORDER), len(dimensions)))
for j, dim_key in enumerate(dimensions):
    details = combined[dim_key]["details"]
    # Group by category
    by_cat: dict[str, list[float]] = {}
    for d in details:
        by_cat.setdefault(d["category"], []).append(d["diff"])
    for i, cat in enumerate(CATEGORY_ORDER):
        vals = by_cat.get(cat, [])
        matrix[i, j] = np.mean(vals) if vals else 0.0

# Determine symmetric color limits
vmax = np.max(np.abs(matrix))
vmax = max(vmax, 0.1)  # avoid degenerate range

fig, ax = plt.subplots(figsize=(7, 5))
cmap = plt.get_cmap("RdBu_r")
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

ax.set_xticks(np.arange(len(dimensions)))
ax.set_xticklabels(dim_labels, fontsize=11)
ax.set_yticks(np.arange(len(CATEGORY_ORDER)))
ax.set_yticklabels(cat_labels, fontsize=11)

# Annotate cells
for i in range(len(CATEGORY_ORDER)):
    for j in range(len(dimensions)):
        val = matrix[i, j]
        text_color = "white" if abs(val) > 0.6 * vmax else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=text_color)

ax.set_title("Direction asymmetry by category and dimension", fontsize=13, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Mean direction asymmetry")

fig.tight_layout()
fig.savefig(
    f"{ASSETS_DIR}/plot_021426_category_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)
print("Saved plot 2: category heatmap")
