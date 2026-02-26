"""Generate preference steering comparison plot for v2 patch report."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/experiments/persona_vectors_v2")
ASSETS_DIR = Path("experiments/persona_vectors/follow_up/assets")

# Load v2 patch coherent results
with open(RESULTS_DIR / "preference_steering" / "coherent" / "preference_results.json") as f:
    coherent = json.load(f)

# Load original v2 results for comparison
with open(RESULTS_DIR / "preference_steering" / "preference_results.json") as f:
    original = json.load(f)

personas = ["creative_artist", "evil", "lazy", "stem_nerd", "uncensored"]
labels = {"creative_artist": "Creative\nArtist", "evil": "Evil",
          "lazy": "Lazy", "stem_nerd": "STEM\nNerd", "uncensored": "Uncensored"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left panel: Original v2 (no coherence filtering) ---
ax = axes[0]
x = np.arange(len(personas))
width = 0.35

base_rates = []
steer_rates = []
for p in personas:
    r = original[p]
    base_rates.append(r["baseline_rate"] if r["baseline_rate"] is not None else 0)
    steer_rates.append(r["steered_rate"] if r["steered_rate"] is not None else 0)

bars1 = ax.bar(x - width/2, base_rates, width, label="Baseline", color="#3498db", alpha=0.8)
bars2 = ax.bar(x + width/2, steer_rates, width, label="Steered", color="#e74c3c", alpha=0.8)

# Annotate with n
for i, p in enumerate(personas):
    r = original[p]
    ax.annotate(f"n={r['n_base']}", (x[i] - width/2, base_rates[i]),
                ha="center", va="bottom", fontsize=7)
    n_steer = r["n_steered"]
    n_unparse = r.get("n_unparseable", 0)
    label = f"n={n_steer}"
    if n_unparse > 0:
        label += f"\n({n_unparse} unparse)"
    ax.annotate(label, (x[i] + width/2, steer_rates[i]),
                ha="center", va="bottom", fontsize=7)

ax.set_ylabel("Positive preference rate")
ax.set_title("Original v2 (no coherence filter)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([labels[p] for p in personas])
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
ax.legend()

# --- Right panel: v2 patch (coherence-filtered) ---
ax = axes[1]

base_rates_coh = []
steer_rates_coh = []
for p in personas:
    r = coherent[p]
    coh = r["coherent_subset"]
    base_rates_coh.append(coh["baseline_rate"] if coh["baseline_rate"] is not None else 0)
    steer_rates_coh.append(coh["steered_rate"] if coh["steered_rate"] is not None else 0)

bars1 = ax.bar(x - width/2, base_rates_coh, width, label="Baseline (coherent)", color="#3498db", alpha=0.8)
bars2 = ax.bar(x + width/2, steer_rates_coh, width, label="Steered (coherent)", color="#e74c3c", alpha=0.8)

# Annotate with n
for i, p in enumerate(personas):
    r = coherent[p]
    coh = r["coherent_subset"]
    ax.annotate(f"n={coh['n_baseline']}", (x[i] - width/2, base_rates_coh[i]),
                ha="center", va="bottom", fontsize=7)
    ax.annotate(f"n={coh['n_steered']}", (x[i] + width/2, steer_rates_coh[i]),
                ha="center", va="bottom", fontsize=7)

ax.set_ylabel("Positive preference rate (coherent subset)")
ax.set_title("v2 Patch (coherence-filtered)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([labels[p] for p in personas])
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
ax.legend()

fig.suptitle("Preference Steering: Original vs. Coherence-Filtered",
             fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out_path = ASSETS_DIR / "plot_022626_coherent_preference_steering.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
