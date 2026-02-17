import matplotlib.pyplot as plt
import numpy as np

# Data for stem_enthusiast persona
data = {
    "math": (0.2662, 21),
    "coding": (0.2296, 6),
    "security_legal": (0.2280, 2),
    "model_manipulation": (-0.0449, 3),
    "harmful_request": (-0.0522, 21),
    "other": (-0.0524, 1),
    "knowledge_qa": (-0.0804, 17),
    "content_generation": (-0.1087, 9),
    "persuasive_writing": (-0.1515, 3),
    "fiction": (-0.3108, 8),
}

# Expected positive/negative categories
expected_positive = {"math", "coding"}
expected_negative = {"fiction", "content_generation"}

# Sort by absolute delta magnitude
categories = sorted(data.keys(), key=lambda x: abs(data[x][0]), reverse=True)
deltas = [data[cat][0] for cat in categories]
ns = [data[cat][1] for cat in categories]

# Color assignment
colors = []
for cat in categories:
    if cat in expected_positive:
        if deltas[categories.index(cat)] > 0:
            colors.append("#2ecc71")  # green for expected-positive hits
        else:
            colors.append("#95a5a6")  # gray for expected-positive misses
    elif cat in expected_negative:
        if deltas[categories.index(cat)] < 0:
            colors.append("#e74c3c")  # red for expected-negative hits
        else:
            colors.append("#95a5a6")  # gray for expected-negative misses
    else:
        colors.append("#95a5a6")  # gray for other

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

# Create horizontal bar chart
y_pos = np.arange(len(categories))
bars = ax.barh(y_pos, deltas, color=colors, edgecolor="black", linewidth=0.5)

# Add vertical dashed line at x=0
ax.axvline(x=0, color="black", linestyle="--", linewidth=1)

# Add n labels next to bars
for i, (delta, n) in enumerate(zip(deltas, ns)):
    label_x = delta + (0.01 if delta > 0 else -0.01)
    ha = "left" if delta > 0 else "right"
    ax.text(label_x, i, f"n={n}", va="center", ha=ha, fontsize=9)

# Labels and title
ax.set_yticks(y_pos)
ax.set_yticklabels(categories)
ax.set_xlabel("Mean Δ(p_choose) vs baseline", fontsize=11)
ax.set_title("stem_enthusiast: Category-level preference shifts vs neutral baseline", fontsize=12, fontweight="bold")

# Grid for readability
ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
ax.set_axisbelow(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig(
    "/Users/oscargilg/Dev/MATS/Preferences/experiments/probe_generalization/persona_ood/assets/plot_021626_stem_category_deltas.png",
    dpi=300,
    bbox_inches="tight"
)
print("Plot saved to experiments/probe_generalization/persona_ood/assets/plot_021626_stem_category_deltas.png")
plt.close()
