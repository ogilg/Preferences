"""Plotting script for error prefill follow-up experiment results."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({"font.size": 11})

RESULTS_PATH = "experiments/truth_probes/error_prefill/error_prefill_followup_results.json"
ASSETS_DIR = "experiments/truth_probes/error_prefill/assets"

with open(RESULTS_PATH) as f:
    results = json.load(f)

LAYERS = ["25", "32", "39", "46", "53"]
LAYERS_INT = [25, 32, 39, 46, 53]
FOLLOWUPS = ["neutral", "presupposes", "challenge", "same_domain", "control"]
FOLLOWUP_COLORS = {
    "neutral": "#1f77b4",
    "presupposes": "#d62728",
    "challenge": "#ff7f0e",
    "same_domain": "#2ca02c",
    "control": "#7f7f7f",
}
PROBES = ["tb-2", "tb-5", "task_mean"]
PROBE_LABELS = {"tb-2": "tb-2", "tb-5": "tb-5", "task_mean": "task_mean"}
ASSISTANT_SELECTORS = ["assistant_mean", "assistant_tb:-1"]


def get_best_d(layer_dict):
    """Get the best Cohen's d across layers."""
    best = -np.inf
    for layer in LAYERS:
        if layer in layer_dict:
            d = layer_dict[layer]["cohens_d"]
            if d > best:
                best = d
    return best


# --- Plot 1: Assistant-turn selectors effect sizes (all probes) ---

fig, axes = plt.subplots(len(PROBES), len(ASSISTANT_SELECTORS), figsize=(12, 12), sharey=True, sharex=True)
fig.suptitle("Assistant-turn selectors: correct vs incorrect separation", fontsize=14)

for row_idx, probe in enumerate(PROBES):
    for col_idx, selector in enumerate(ASSISTANT_SELECTORS):
        ax = axes[row_idx, col_idx]
        sel_data = results["assistant_selectors_no_lying"][selector][probe]
        for followup in FOLLOWUPS:
            ds = [sel_data[followup][layer]["cohens_d"] for layer in LAYERS]
            ax.plot(LAYERS_INT, ds, marker="o", label=followup, color=FOLLOWUP_COLORS[followup])
        if row_idx == 0:
            ax.set_title(selector)
        if col_idx == 0:
            ax.set_ylabel(f"{PROBE_LABELS[probe]} probe\nCohen's d")
        if row_idx == len(PROBES) - 1:
            ax.set_xlabel("Layer")
        ax.set_xticks(LAYERS_INT)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

axes[0, 0].set_ylim(-0.5, None)
axes[0, -1].legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(f"{ASSETS_DIR}/plot_031226_assistant_selectors_effect_sizes.png", dpi=150)
plt.close(fig)
print("Saved plot 1: assistant_selectors_effect_sizes")


# --- Plot 2: Assistant selector comparison (presupposes, all probes) ---

selector_order = ["assistant_mean", "assistant_tb:-1"]
selector_labels = ["assistant_mean", "assistant_tb:-1"]

fig, axes = plt.subplots(1, len(PROBES), figsize=(14, 5), sharey=True)
fig.suptitle("Signal strength by selector position (presupposes follow-up, no lying)", fontsize=14)

for ax, probe in zip(axes, PROBES):
    probe_values = {}
    for sel in selector_order:
        sel_data = results["assistant_selectors_no_lying"][sel][probe]["presupposes"]
        probe_values[sel] = get_best_d(sel_data)

    bar_values = [probe_values[s] for s in selector_order]
    bar_colors = ["#dd8452", "#dd8452"]

    bars = ax.bar(range(len(selector_order)), bar_values, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(selector_order)))
    ax.set_xticklabels(selector_labels, fontsize=9)
    ax.set_ylabel("Best Cohen's d across layers")
    ax.set_title(f"{PROBE_LABELS[probe]} probe")
    ax.set_ylim(0, None)
    for bar, val in zip(bars, bar_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{val:.2f}",
                ha="center", va="bottom", fontsize=10)

fig.tight_layout()
fig.savefig(f"{ASSETS_DIR}/plot_031226_assistant_vs_followup_comparison.png", dpi=150)
plt.close(fig)
print("Saved plot 2: assistant_vs_followup_comparison")


# --- Plot 3: Lying effect on tb selectors (all probes) ---

lying_followups = ["neutral", "presupposes"]
lying_colors = {"neutral": "#1f77b4", "presupposes": "#d62728"}
lying_conditions = ["lie_direct", "lie_roleplay"]

fig, axes = plt.subplots(1, len(PROBES), figsize=(16, 5), sharey=True)
fig.suptitle("Effect of lying system prompts on turn-boundary signal (turn_boundary:-2)", fontsize=14)

for ax, probe in zip(axes, PROBES):
    lying_best_d = {}
    for condition in lying_conditions:
        lying_best_d[condition] = {}
        ly_data = results["lying_conversations"]["turn_boundary:-2"][probe][condition]
        for fu in lying_followups:
            lying_best_d[condition][fu] = get_best_d(ly_data[fu])

    x = np.arange(len(lying_conditions))
    width = 0.35

    for i, fu in enumerate(lying_followups):
        vals = [lying_best_d[cond][fu] for cond in lying_conditions]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=fu, color=lying_colors[fu],
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["lie_direct", "lie_roleplay"])
    ax.set_ylabel("Best Cohen's d across layers")
    ax.set_title(f"{PROBE_LABELS[probe]} probe")
    ax.set_ylim(0, None)
    if probe == PROBES[-1]:
        ax.legend()

fig.tight_layout()
fig.savefig(f"{ASSETS_DIR}/plot_031226_lying_effect_on_tb_selectors.png", dpi=150)
plt.close(fig)
print("Saved plot 3: lying_effect_on_tb_selectors")


# --- Plot 4: Lying assistant_tb heatmap (all probes) ---

heatmap_selector = "assistant_tb:-1"
heatmap_followup = "presupposes"
conditions_heatmap = ["no_lying", "lie_direct", "lie_roleplay"]

fig, axes = plt.subplots(1, len(PROBES), figsize=(18, 4))
fig.suptitle(f"Lying system prompt effect on assistant-turn signal\n({heatmap_selector}, {heatmap_followup})", fontsize=14)

for ax, probe in zip(axes, PROBES):
    matrix = np.zeros((3, len(LAYERS)))
    for row_idx, condition in enumerate(conditions_heatmap):
        if condition == "no_lying":
            layer_data = results["assistant_selectors_no_lying"][heatmap_selector][probe][heatmap_followup]
        else:
            layer_data = results["lying_conversations"][heatmap_selector][probe][condition][heatmap_followup]
        for col_idx, layer in enumerate(LAYERS):
            matrix[row_idx, col_idx] = layer_data[layer]["cohens_d"]

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=3.5, aspect="auto")

    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f"L{l}" for l in LAYERS])
    ax.set_yticks(range(len(conditions_heatmap)))
    ax.set_yticklabels(conditions_heatmap)
    ax.set_title(f"{PROBE_LABELS[probe]} probe")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11, color=text_color)

fig.colorbar(im, ax=axes.tolist(), label="Cohen's d", shrink=0.8)
fig.savefig(f"{ASSETS_DIR}/plot_031226_lying_assistant_tb_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved plot 4: lying_assistant_tb_heatmap")

print("\nAll plots saved to:", ASSETS_DIR)
