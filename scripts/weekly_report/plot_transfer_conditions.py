import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

conditions = ["Control", "Swap both\ntasks", "Same-topic\nswap", "Cross-topic\nswap"]
flip_rates = [83.6, 30.6, 29.2, 12.1]
ci_low = [78.4, 26.1, 24.7, 8.7]
ci_high = [88.8, 35.2, 33.7, 15.5]

yerr_low = [r - lo for r, lo in zip(flip_rates, ci_low)]
yerr_high = [hi - r for r, hi in zip(flip_rates, ci_high)]

colors = ["#7f7f7f", "#4a90d9", "#4a90d9", "#4a90d9"]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(
    conditions, flip_rates, color=colors, edgecolor="white", linewidth=1.5, width=0.55
)
ax.errorbar(
    range(len(conditions)),
    flip_rates,
    yerr=[yerr_low, yerr_high],
    fmt="none",
    color="black",
    capsize=5,
    capthick=1.2,
    linewidth=1.2,
)

for i, rate in enumerate(flip_rates):
    ax.text(
        i, rate + 4, f"{rate:.0f}%",
        ha="center", va="bottom", fontsize=12, fontweight="bold",
    )

# Bracket: task-dependent component (Control → Swap both)
bracket_y = 100
ax.plot([0, 0, 1, 1], [bracket_y - 2, bracket_y, bracket_y, bracket_y - 2],
        color="#333", lw=1.5)
ax.text(0.5, bracket_y + 1, "~53 pp task-dependent",
        ha="center", va="bottom", fontsize=9, color="#333")

# Arrow: topic match matters (same-topic → cross-topic)
ax.annotate(
    "topic match\nmatters",
    xy=(2.85, 22), xytext=(3.4, 55),
    fontsize=9, color="#c0392b", ha="center",
    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3),
)

ax.set_ylabel("Flip rate (%)", fontsize=12)
ax.set_ylim(0, 115)
ax.set_xlim(-0.5, 3.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

plt.tight_layout()
plt.savefig(
    "docs/logs/assets/plot_031226_eot_transfer_conditions.png",
    dpi=200,
    bbox_inches="tight",
)
print("Saved.")
