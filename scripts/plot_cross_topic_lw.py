"""Cross-topic generalisation plot for LW post.

Two-panel bar chart: Pearson r (left) and pairwise accuracy (right).
Three conditions: Gemma-3 IT, Gemma-3 PT, sentence-transformer baseline.
All numbers loaded from probe result JSONs — nothing hardcoded.

Colour scheme (shared with plot_per_topic_lw.py):
  IT: blue (#7ABAED light, #2978B5 dark)
  PT: slate (#B0BEC5 light, #607D8B dark)
  ST: amber (#F0D58C light, #C9A833 dark)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Data paths ---
GEMMA3_IT_HELDOUT = Path("results/probes/gemma3_10k_heldout_std_raw/manifest.json")
GEMMA3_PT_HELDOUT = Path("results/probes/gemma3_pt_10k_heldout_std_raw/manifest.json")
ST_HELDOUT = Path("results/probes/st_10k_heldout_std_raw/manifest.json")

GEMMA3_IT_HOO = Path("results/probes/gemma3_10k_hoo_topic/hoo_summary.json")
GEMMA3_PT_HOO = Path("results/probes/gemma3_pt_10k_hoo_topic/hoo_summary.json")
ST_HOO = Path("results/probes/st_10k_hoo_topic/hoo_summary.json")

LW_ASSETS = Path("docs/lw_post/assets")

BEST_LAYERS = {"gemma3_it": 31, "gemma3_pt": 31, "st": 0}

# Colour scheme: (light=held-out, dark=cross-topic)
COLORS = {
    "it": {"light": "#7ABAED", "dark": "#2978B5"},
    "pt": {"light": "#B0BEC5", "dark": "#607D8B"},
    "st": {"light": "#F0D58C", "dark": "#C9A833"},
}


def load_heldout_from_manifest(path: Path, layer: int):
    with open(path) as f:
        manifest = json.load(f)
    for probe in manifest["probes"]:
        if probe["layer"] == layer:
            return {"r": probe["final_r"], "acc": probe["final_acc"]}
    raise KeyError(f"Layer {layer} not found in {path}")


def load_hoo_summary(path: Path, layer: int):
    with open(path) as f:
        summary = json.load(f)
    key = f"ridge_L{layer}"
    hoo_rs, hoo_accs = [], []
    for fold in summary["folds"]:
        metrics = fold["layers"][key]
        hoo_rs.append(metrics["hoo_r"])
        if "hoo_acc" in metrics:
            hoo_accs.append(metrics["hoo_acc"])
    return {
        "hoo_r_mean": np.mean(hoo_rs),
        "hoo_r_std": np.std(hoo_rs),
        "hoo_acc_mean": np.mean(hoo_accs) if hoo_accs else None,
        "hoo_acc_std": np.std(hoo_accs) if hoo_accs else None,
    }


# --- Load data ---
g3_it_heldout = load_heldout_from_manifest(GEMMA3_IT_HELDOUT, BEST_LAYERS["gemma3_it"])
g3_pt_heldout = load_heldout_from_manifest(GEMMA3_PT_HELDOUT, BEST_LAYERS["gemma3_pt"])
st_heldout = load_heldout_from_manifest(ST_HELDOUT, BEST_LAYERS["st"])

g3_it_hoo = load_hoo_summary(GEMMA3_IT_HOO, BEST_LAYERS["gemma3_it"])
g3_pt_hoo = load_hoo_summary(GEMMA3_PT_HOO, BEST_LAYERS["gemma3_pt"])
st_hoo = load_hoo_summary(ST_HOO, BEST_LAYERS["st"])

# --- Conditions ---
conditions = [
    ("Gemma-3 IT\n(L31)", g3_it_heldout, g3_it_hoo, COLORS["it"]),
    ("Gemma-3 PT\n(L31)", g3_pt_heldout, g3_pt_hoo, COLORS["pt"]),
    ("Sentence-\ntransformer", st_heldout, st_hoo, COLORS["st"]),
]

models = [c[0] for c in conditions]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
x = np.arange(len(models))
width = 0.3


def draw_bar_pair(ax, i, heldout_val, hoo_mean, hoo_std, colors, first):
    ax.bar(i - width / 2, heldout_val, width, color=colors["light"],
           edgecolor="black", linewidth=0.5,
           label="Test set (2,000 utilities)" if first else "")
    ax.text(i - width / 2, heldout_val + 0.01, f"{heldout_val:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
    if hoo_mean is not None:
        ax.bar(i + width / 2, hoo_mean, width, yerr=hoo_std,
               color=colors["dark"], edgecolor="black", linewidth=0.5,
               capsize=3, error_kw={"linewidth": 0.8, "alpha": 0.5},
               label="Leave-one-topic-out" if first else "")
        ax.text(i + width / 2, hoo_mean + 0.01, f"{hoo_mean:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")


# --- Left: Pearson r ---
for i, (label, heldout, hoo, colors) in enumerate(conditions):
    draw_bar_pair(ax1, i, heldout["r"], hoo["hoo_r_mean"], hoo["hoo_r_std"], colors, first=(i == 0))

ax1.set_ylim(0, 1)
ax1.set_ylabel("Pearson r")
ax1.set_title("Pearson r")
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontweight="bold")
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(axis="y", alpha=0.3, linestyle="--")

# --- Right: Pairwise accuracy ---
for i, (label, heldout, hoo, colors) in enumerate(conditions):
    draw_bar_pair(ax2, i, heldout["acc"], hoo["hoo_acc_mean"], hoo["hoo_acc_std"], colors, first=(i == 0))

ax2.axhline(y=0.5, color="gray", linewidth=1, linestyle="--", label="Chance (0.50)")
ax2.set_ylim(0, 1)
ax2.set_ylabel("Pairwise accuracy")
ax2.set_title("Pairwise Accuracy")
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontweight="bold")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(axis="y", alpha=0.3, linestyle="--")

fig.suptitle("Probe Accuracy: Test Set vs Leave-One-Topic-Out", fontsize=13, y=1.02)
plt.tight_layout()
LW_ASSETS.mkdir(parents=True, exist_ok=True)
plot_path = LW_ASSETS / "plot_022626_cross_model_bar.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {plot_path}")
