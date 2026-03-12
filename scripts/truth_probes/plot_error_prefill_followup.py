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
ASSISTANT_SELECTORS = ["assistant_mean", "assistant_tb:-1", "assistant_tb:0"]


def get_best_d(layer_dict):
    """Get the best Cohen's d across layers."""
    best = -np.inf
    for layer in LAYERS:
        if layer in layer_dict:
            d = layer_dict[layer]["cohens_d"]
            if d > best:
                best = d
    return best


# --- Plot 1: Assistant-turn selectors effect sizes ---

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("Assistant-turn selectors: correct vs incorrect separation (tb-2 probe)", fontsize=14)

for ax, selector in zip(axes, ASSISTANT_SELECTORS):
    sel_data = results["assistant_selectors_no_lying"][selector]["tb-2"]
    for followup in FOLLOWUPS:
        ds = [sel_data[followup][layer]["cohens_d"] for layer in LAYERS]
        ax.plot(LAYERS_INT, ds, marker="o", label=followup, color=FOLLOWUP_COLORS[followup])
    ax.set_title(selector)
    ax.set_xlabel("Layer")
    ax.set_xticks(LAYERS_INT)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

axes[0].set_ylabel("Cohen's d")
axes[0].set_ylim(-0.5, None)
axes[-1].legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(f"{ASSETS_DIR}/plot_031226_assistant_selectors_effect_sizes.png", dpi=150)
plt.close(fig)
print("Saved plot 1: assistant_selectors_effect_sizes")


# --- Plot 2: Assistant vs follow-up comparison (presupposes, tb-2 probe) ---

# Baseline values from original experiment
baseline_selectors = {
    "turn_boundary:-2": 2.58,
    "turn_boundary:-5": 2.33,
}

# Compute best d for assistant selectors from follow-up results
for sel in ASSISTANT_SELECTORS:
    sel_data = results["assistant_selectors_no_lying"][sel]["tb-2"]["presupposes"]
    baseline_selectors[sel] = get_best_d(sel_data)

selector_order = [
    "turn_boundary:-2",
    "turn_boundary:-5",
    "assistant_mean",
    "assistant_tb:-1",
    "assistant_tb:0",
]
selector_labels = [
    "turn_boundary:-2\n(original)",
    "turn_boundary:-5\n(original)",
    "assistant_mean",
    "assistant_tb:-1",
    "assistant_tb:0",
]
bar_values = [baseline_selectors[s] for s in selector_order]
bar_colors = ["#4c72b0", "#4c72b0", "#dd8452", "#dd8452", "#dd8452"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(selector_order)), bar_values, color=bar_colors, edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(selector_order)))
ax.set_xticklabels(selector_labels, fontsize=10)
ax.set_ylabel("Best Cohen's d across layers")
ax.set_title("Signal strength by selector position (presupposes, tb-2 probe)")
ax.set_ylim(0, max(bar_values) * 1.15)
for bar, val in zip(bars, bar_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{val:.2f}",
            ha="center", va="bottom", fontsize=10)
fig.tight_layout()
fig.savefig(f"{ASSETS_DIR}/plot_031226_assistant_vs_followup_comparison.png", dpi=150)
plt.close(fig)
print("Saved plot 2: assistant_vs_followup_comparison")


# --- Plot 3: Lying effect on tb selectors ---

# Baseline (no lying) values
no_lying_vals = {
    "neutral": 1.80,
    "presupposes": 2.58,
}

lying_conditions = ["no_lying", "lie_direct", "lie_roleplay"]
lying_followups = ["neutral", "presupposes"]
lying_colors = {"neutral": "#1f77b4", "presupposes": "#d62728"}

# Compute lying values from results
lying_best_d = {}
for condition in lying_conditions:
    lying_best_d[condition] = {}
    if condition == "no_lying":
        for fu in lying_followups:
            lying_best_d[condition][fu] = no_lying_vals[fu]
    else:
        ly_data = results["lying_conversations"]["turn_boundary:-2"]["tb-2"][condition]
        for fu in lying_followups:
            lying_best_d[condition][fu] = get_best_d(ly_data[fu])

fig, ax = plt.subplots(figsize=(9, 5))
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
ax.set_xticklabels(["no_lying\n(baseline)", "lie_direct", "lie_roleplay"])
ax.set_ylabel("Best Cohen's d across layers")
ax.set_title("Effect of lying system prompts on turn-boundary signal (tb-2 probe)")
ax.set_ylim(0, max(no_lying_vals.values()) * 1.2)
ax.legend()
fig.tight_layout()
fig.savefig(f"{ASSETS_DIR}/plot_031226_lying_effect_on_tb_selectors.png", dpi=150)
plt.close(fig)
print("Saved plot 3: lying_effect_on_tb_selectors")


# --- Plot 4: Lying assistant_tb heatmap ---

heatmap_selector = "assistant_tb:-1"
heatmap_probe = "tb-2"
heatmap_followup = "presupposes"
conditions_heatmap = ["no_lying", "lie_direct", "lie_roleplay"]

# Build the matrix
matrix = np.zeros((3, len(LAYERS)))
for row_idx, condition in enumerate(conditions_heatmap):
    if condition == "no_lying":
        layer_data = results["assistant_selectors_no_lying"][heatmap_selector][heatmap_probe][heatmap_followup]
    else:
        layer_data = results["lying_conversations"][heatmap_selector][heatmap_probe][condition][heatmap_followup]
    for col_idx, layer in enumerate(LAYERS):
        matrix[row_idx, col_idx] = layer_data[layer]["cohens_d"]

fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=3.5, aspect="auto")

ax.set_xticks(range(len(LAYERS)))
ax.set_xticklabels([f"L{l}" for l in LAYERS])
ax.set_yticks(range(len(conditions_heatmap)))
ax.set_yticklabels(conditions_heatmap)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        text_color = "white" if val < 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11, color=text_color)

ax.set_title("Lying system prompt effect on assistant-turn signal\n(assistant_tb:-1, tb-2 probe, presupposes)")
fig.colorbar(im, ax=ax, label="Cohen's d")
fig.tight_layout()
fig.savefig(f"{ASSETS_DIR}/plot_031226_lying_assistant_tb_heatmap.png", dpi=150)
plt.close(fig)
print("Saved plot 4: lying_assistant_tb_heatmap")

print("\nAll plots saved to:", ASSETS_DIR)
