"""Cross-topic generalisation plot for LW post.

Two-panel bar chart: Pearson r (left) and pairwise accuracy (right).
Three conditions: Gemma-3 IT, Gemma-2 base, content baseline (sentence transformer).
Uses 10k HOO results (hold-1-out-of-12 topics).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GEMMA3_HOO = Path("results/probes/gemma3_10k_hoo_topic/hoo_summary.json")
GEMMA2_HOO = Path("results/probes/gemma2_base_10k_hoo_topic/hoo_summary.json")
ST_HOO = Path("results/probes/st_10k_hoo_topic/hoo_summary.json")

# 10k heldout eval results (separate 4k test set, not CV)
GEMMA3_HELDOUT_R = 0.864    # from gemma3_10k_heldout_std_raw, L31
GEMMA3_HELDOUT_ACC = 0.768  # from gemma3_10k_heldout_std_raw, L31
GEMMA2_HELDOUT_R = 0.767    # from gemma2_10k_heldout_std_raw, L23
GEMMA2_HELDOUT_ACC = 0.708  # computed against real pairwise choices (4k eval set)
ST_HELDOUT_R = 0.614        # from st_10k_heldout_std_raw, L0 (all-MiniLM-L6-v2)
ST_HELDOUT_ACC = 0.651      # from st_10k_heldout_std_raw, L0

LW_ASSETS = Path("docs/lw_post/assets")

GEMMA3_BEST_LAYER = 31
GEMMA2_BEST_LAYER = 23  # ~50% depth in 46-layer model
ST_LAYER = 0  # sentence transformer has no layer dimension


def load_hoo_summary(path: Path, best_layer: int):
    with open(path) as f:
        summary = json.load(f)
    key = f"ridge_L{best_layer}"
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
        "n_folds": len(hoo_rs),
    }


def try_load(path, layer, label):
    if path.exists():
        data = load_hoo_summary(path, layer)
        print(f"{label} loaded: HOO r = {data['hoo_r_mean']:.3f}")
        return data
    print(f"{label} not found at {path}, will show TBD")
    return None


# --- Load data ---
g3 = load_hoo_summary(GEMMA3_HOO, GEMMA3_BEST_LAYER)
g2 = try_load(GEMMA2_HOO, GEMMA2_BEST_LAYER, "Gemma-2 base")
st = try_load(ST_HOO, ST_LAYER, "Content baseline (ST)")

# --- Conditions ---
conditions = [
    ("Gemma-3 27B IT\n(L31)", g3, GEMMA3_HELDOUT_R, GEMMA3_HELDOUT_ACC),
    ("Gemma-2 27B Base\n(L23)", g2, GEMMA2_HELDOUT_R, GEMMA2_HELDOUT_ACC),
    ("Content\nBaseline", st, ST_HELDOUT_R, ST_HELDOUT_ACC),
]

models = [c[0] for c in conditions]
colors_heldout = "#a8d8ea"
colors_hoo = "#3498db"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
x = np.arange(len(models))
width = 0.3


def draw_bar_pair(ax, i, heldout_val, hoo_mean, hoo_std, first):
    if heldout_val is not None:
        ax.bar(i - width / 2, heldout_val, width, color=colors_heldout,
               edgecolor="black", linewidth=0.5,
               label="Heldout (separate 4k)" if first else "")
        ax.text(i - width / 2, heldout_val + 0.01, f"{heldout_val:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    else:
        ax.text(i - width / 2, 0.5, "TBD", ha="center", va="bottom",
                fontsize=9, fontstyle="italic", color="gray")
    if hoo_mean is not None:
        ax.bar(i + width / 2, hoo_mean, width, yerr=hoo_std,
               color=colors_hoo, edgecolor="black", linewidth=0.5, capsize=3,
               label="Cross-topic HOO" if first else "")
        ax.text(i + width / 2, hoo_mean + 0.01, f"{hoo_mean:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    else:
        ax.text(i + width / 2, 0.5, "TBD", ha="center", va="bottom",
                fontsize=9, fontstyle="italic", color="gray")


# --- Left: Pearson r ---
for i, (label, data, heldout_r, heldout_acc) in enumerate(conditions):
    hoo_r = data["hoo_r_mean"] if data else None
    hoo_r_std = data["hoo_r_std"] if data else None
    draw_bar_pair(ax1, i, heldout_r, hoo_r, hoo_r_std, first=(i == 0))

ax1.set_ylim(0, 1)
ax1.set_ylabel("Pearson r")
ax1.set_title("Pearson r")
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(axis="y", alpha=0.3, linestyle="--")

# --- Right: Pairwise accuracy ---
for i, (label, data, heldout_r, heldout_acc) in enumerate(conditions):
    hoo_acc = data["hoo_acc_mean"] if data else None
    hoo_acc_std = data["hoo_acc_std"] if data else None
    draw_bar_pair(ax2, i, heldout_acc, hoo_acc, hoo_acc_std, first=(i == 0))

ax2.axhline(y=0.5, color="gray", linewidth=1, linestyle="--", label="Chance (0.50)")
ax2.set_ylim(0, 1)
ax2.set_ylabel("Pairwise accuracy")
ax2.set_title("Pairwise Accuracy")
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(axis="y", alpha=0.3, linestyle="--")

fig.suptitle("Cross-Topic Generalization at Best Layer (10k training)", fontsize=13, y=1.01)
plt.tight_layout()
LW_ASSETS.mkdir(parents=True, exist_ok=True)
plot_path = LW_ASSETS / "plot_021926_cross_model_bar.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {plot_path}")
