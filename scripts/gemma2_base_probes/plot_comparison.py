"""Gemma-2 27B Base vs Gemma-3 27B IT: probe R² across fractional layer depth."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Data ---

layer_fracs = np.array([0.25, 0.50, 0.60, 0.70, 0.80, 0.90])

gemma2_base = {
    "label": "Gemma-2 27B Base (46 layers, n=2264)",
    "layers": [11, 23, 27, 32, 36, 41],
    "r2": np.array([0.747, 0.789, 0.773, 0.757, 0.762, 0.769]),
}

gemma3_it = {
    "label": "Gemma-3 27B IT (62 layers, n=3000)",
    "layers": [15, 31, 37, 43, 49, 55],
    "r2": np.array([0.705, 0.863, 0.849, 0.840, 0.838, 0.835]),
}

OUTPUT = Path(
    "experiments/probe_science/content_orthogonal/gemma2base"
    "/base_probes/assets/plot_021726_base_vs_it_comparison.png"
)

# --- Plot ---

fig, ax = plt.subplots(figsize=(7, 4.5))

for series, color, marker in [
    (gemma2_base, "#2274A5", "o"),
    (gemma3_it, "#D64933", "s"),
]:
    ax.plot(
        layer_fracs,
        series["r2"],
        color=color,
        marker=marker,
        markersize=7,
        linewidth=2,
        label=series["label"],
        zorder=3,
    )
    for frac, r2, layer in zip(layer_fracs, series["r2"], series["layers"]):
        ax.annotate(
            f"L{layer}\n{r2:.3f}",
            (frac, r2),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=7.5,
            color=color,
        )

ax.set_xlabel("Fractional layer depth", fontsize=11)
ax.set_ylabel("CV R²", fontsize=11)
ax.set_title("Preference probe R²: Gemma-2 Base vs Gemma-3 IT", fontsize=12, fontweight="bold")

ax.set_xlim(0.20, 0.95)
ax.set_ylim(0.60, 0.95)
ax.set_xticks(layer_fracs)
ax.set_xticklabels([f"{f:.2f}" for f in layer_fracs])

ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(axis="both", alpha=0.25, linewidth=0.5)

ax.annotate(
    "Note: Gemma-2 Base uses 2264 tasks; Gemma-3 IT uses 3000 tasks.\n"
    "Both use ridge probes with 5-fold CV on prompt-last activations.",
    xy=(0.50, 0.02),
    xycoords="axes fraction",
    fontsize=7.5,
    color="gray",
    va="bottom",
    ha="center",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved to {OUTPUT}")
