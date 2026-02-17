from dotenv import load_dotenv

load_dotenv()

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/probes")
ASSETS_DIR = Path("experiments/probe_generalization/gemma2_base/assets")

plt.style.use("seaborn-v0_8-whitegrid")


def load_summary(name: str) -> dict:
    path = RESULTS_DIR / name / "hoo_summary.json"
    with open(path) as f:
        return json.load(f)


# Load all summaries
gemma2_raw = load_summary("gemma2_base_hoo_raw")
gemma3_raw = load_summary("hoo_scaled_raw")
st_baseline = load_summary("hoo_scaled_st_baseline")
gemma2_demeaned = load_summary("gemma2_base_hoo_demeaned")

GEMMA2_TOTAL_LAYERS = 46
GEMMA3_TOTAL_LAYERS = 62


# ── Plot 1: Layer comparison ──

fig, ax = plt.subplots(figsize=(8, 5))

# Gemma-2 base
g2_layers = sorted(gemma2_raw["layer_summary"].keys(), key=int)
g2_depths = [int(l) / GEMMA2_TOTAL_LAYERS for l in g2_layers]
g2_means = [gemma2_raw["layer_summary"][l]["ridge"]["mean_hoo_r"] for l in g2_layers]
g2_stds = [gemma2_raw["layer_summary"][l]["ridge"]["std_hoo_r"] for l in g2_layers]

ax.errorbar(
    g2_depths, g2_means, yerr=g2_stds,
    fmt="o-", color="#d62728", capsize=4, label="Gemma-2 27B Base",
    markersize=6, linewidth=1.5,
)

# Gemma-3 IT
g3_layers = sorted(gemma3_raw["layer_summary"].keys(), key=int)
g3_depths = [int(l) / GEMMA3_TOTAL_LAYERS for l in g3_layers]
g3_means = [gemma3_raw["layer_summary"][l]["ridge"]["mean_hoo_r"] for l in g3_layers]
g3_stds = [gemma3_raw["layer_summary"][l]["ridge"]["std_hoo_r"] for l in g3_layers]

ax.errorbar(
    g3_depths, g3_means, yerr=g3_stds,
    fmt="s-", color="#1f77b4", capsize=4, label="Gemma-3 27B IT",
    markersize=6, linewidth=1.5,
)

# ST baseline (horizontal dashed line)
st_mean = st_baseline["layer_summary"]["0"]["ridge"]["mean_hoo_r"]
st_std = st_baseline["layer_summary"]["0"]["ridge"]["std_hoo_r"]
ax.axhline(st_mean, color="#2ca02c", linestyle="--", linewidth=1.5,
           label=f"ST Baseline (r={st_mean:.3f})")
ax.axhspan(st_mean - st_std, st_mean + st_std, color="#2ca02c", alpha=0.1)

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.set_xlabel("Relative Layer Depth (layer / total layers)")
ax.set_ylabel("HOO Pearson r")
ax.set_title("Hold-One-Out Generalization: Gemma-2 Base vs Gemma-3 IT")
ax.legend(loc="upper left")

fig.tight_layout()
fig.savefig(ASSETS_DIR / "plot_021726_layer_comparison.png", dpi=150)
plt.close(fig)
print(f"Saved: {ASSETS_DIR / 'plot_021726_layer_comparison.png'}")


# ── Plot 2: Per-fold comparison boxplot ──

# Extract per-fold hoo_r at best layer for each model
gemma2_best_layer_key = "ridge_L23"
gemma3_best_layer_key = "ridge_L31"

gemma2_fold_rs = [
    fold["layers"][gemma2_best_layer_key]["hoo_r"]
    for fold in gemma2_raw["folds"]
]
gemma3_fold_rs = [
    fold["layers"][gemma3_best_layer_key]["hoo_r"]
    for fold in gemma3_raw["folds"]
]

fig, ax = plt.subplots(figsize=(6, 5))

parts = ax.violinplot(
    [gemma2_fold_rs, gemma3_fold_rs],
    positions=[1, 2],
    showmeans=True,
    showmedians=True,
)

# Color the violins
colors = ["#d62728", "#1f77b4"]
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.6)
for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
    parts[key].set_color("black")

ax.set_xticks([1, 2])
ax.set_xticklabels(["Gemma-2 27B Base\n(Layer 23)", "Gemma-3 27B IT\n(Layer 31)"])
ax.set_ylim(0, 1.0)
ax.set_ylabel("HOO Pearson r (per fold)")
ax.set_title("Per-Fold HOO Generalization at Best Layer")

# Add individual data points with jitter
rng = np.random.default_rng(42)
for pos, data, color in [(1, gemma2_fold_rs, "#d62728"), (2, gemma3_fold_rs, "#1f77b4")]:
    jitter = rng.uniform(-0.05, 0.05, size=len(data))
    ax.scatter(
        [pos + j for j in jitter], data,
        color=color, alpha=0.3, s=12, zorder=3,
    )

fig.tight_layout()
fig.savefig(ASSETS_DIR / "plot_021726_fold_comparison.png", dpi=150)
plt.close(fig)
print(f"Saved: {ASSETS_DIR / 'plot_021726_fold_comparison.png'}")


# ── Plot 3: Raw vs Demeaned comparison ──

fig, ax = plt.subplots(figsize=(8, 5))

layers = sorted(gemma2_raw["layer_summary"].keys(), key=int)
x = np.arange(len(layers))
width = 0.35

raw_means = [gemma2_raw["layer_summary"][l]["ridge"]["mean_hoo_r"] for l in layers]
raw_stds = [gemma2_raw["layer_summary"][l]["ridge"]["std_hoo_r"] for l in layers]
dem_means = [gemma2_demeaned["layer_summary"][l]["ridge"]["mean_hoo_r"] for l in layers]
dem_stds = [gemma2_demeaned["layer_summary"][l]["ridge"]["std_hoo_r"] for l in layers]

ax.bar(
    x - width / 2, raw_means, width, yerr=raw_stds,
    label="Raw", color="#d62728", alpha=0.8, capsize=4,
)
ax.bar(
    x + width / 2, dem_means, width, yerr=dem_stds,
    label="Demeaned (topic)", color="#ff9896", alpha=0.8, capsize=4,
)

ax.set_xticks(x)
ax.set_xticklabels([f"L{l}" for l in layers])
ax.set_ylim(0, 1.0)
ax.set_xlabel("Layer")
ax.set_ylabel("HOO Pearson r")
ax.set_title("Gemma-2 27B Base: Raw vs Topic-Demeaned HOO Generalization")
ax.legend()

fig.tight_layout()
fig.savefig(ASSETS_DIR / "plot_021726_raw_vs_demeaned.png", dpi=150)
plt.close(fig)
print(f"Saved: {ASSETS_DIR / 'plot_021726_raw_vs_demeaned.png'}")
