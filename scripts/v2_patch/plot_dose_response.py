"""Generate dose-response plots for v2 patch report."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/experiments/persona_vectors_v2")
ASSETS_DIR = Path("experiments/persona_vectors/follow_up/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

PERSONAS_COHERENT = ["creative_artist", "evil", "stem_nerd", "uncensored"]

# Load coherent dose-response summaries
data = {}
for persona in PERSONAS_COHERENT:
    path = RESULTS_DIR / persona / "steering" / "coherent_dose_response_summary.json"
    with open(path) as f:
        data[persona] = json.load(f)

# Load lazy from original dose_response.json
lazy_path = RESULTS_DIR / "lazy" / "steering" / "dose_response.json"
with open(lazy_path) as f:
    data["lazy"] = json.load(f)

# Load coherence-constrained selections for marking
with open(RESULTS_DIR / "coherence_constrained_selections.json") as f:
    selections = json.load(f)

# --- Plot 1: Trait + Coherence dual-axis for each persona ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

all_personas = ["creative_artist", "evil", "lazy", "stem_nerd", "uncensored"]
colors = {"creative_artist": "#e74c3c", "evil": "#2c3e50", "lazy": "#95a5a6",
          "stem_nerd": "#2980b9", "uncensored": "#e67e22"}
labels = {"creative_artist": "Creative Artist", "evil": "Evil",
          "lazy": "Lazy", "stem_nerd": "STEM Nerd", "uncensored": "Uncensored"}

for idx, persona in enumerate(all_personas):
    ax = axes[idx]
    d = data[persona]

    mults = [entry["multiplier"] for entry in d]
    trait_key = "mean_trait" if "mean_trait" in d[0] else "mean_score"
    std_key = "std_trait" if "std_trait" in d[0] else "std_score"
    traits = [entry[trait_key] for entry in d]
    trait_stds = [entry.get(std_key, 0) for entry in d]

    # Trait scores
    color = colors[persona]
    ax.errorbar(mults, traits, yerr=trait_stds, color=color, marker="o",
                linewidth=2, capsize=3, label="Trait score")
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Trait score (1-5)", color=color)
    ax.tick_params(axis="y", labelcolor=color)

    # Coherence on second axis (if available)
    if "coherence_rate" in d[0]:
        ax2 = ax.twinx()
        coh_rates = [entry["coherence_rate"] for entry in d]
        ax2.plot(mults, coh_rates, color="gray", marker="s", linewidth=1.5,
                 linestyle="--", alpha=0.7, label="Coherence rate")
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel("Coherence rate", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax2.axhline(y=0.9, color="gray", linestyle=":", alpha=0.3)

    # Mark selected multiplier
    sel_mult = selections[persona]["multiplier"]
    ax.axvline(x=sel_mult, color="green", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Multiplier")
    ax.set_title(labels[persona], fontweight="bold")
    ax.set_xlim(-0.01, max(mults) + 0.02)

# Hide unused subplot
axes[5].set_visible(False)

fig.suptitle("Dose-Response: Trait Score and Coherence vs. Steering Multiplier",
             fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out_path = ASSETS_DIR / "plot_022626_coherent_dose_response.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
