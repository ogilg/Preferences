"""Generate plots for lying system prompt experiment results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE = Path("experiments/truth_probes/error_prefill/lying_prompts")
RESULTS_PATH = BASE / "lying_10prompt_results.json"
ASSETS = BASE / "assets"

with open(RESULTS_PATH) as f:
    data = json.load(f)

prompt_type_map = data["prompt_type_map"]
PROMPTS = list(prompt_type_map.keys())
LAYERS = ["25", "32", "39", "46", "53"]

DIRECT_COLOR = "#d62728"
ROLEPLAY_COLOR = "#1f77b4"
DIRECT_COLOR_LIGHT = "#ff9896"
ROLEPLAY_COLOR_LIGHT = "#aec7e8"

NO_SYSPROMPT_BASELINE = 3.29


def get_color(prompt_name: str) -> str:
    return DIRECT_COLOR if prompt_type_map[prompt_name] == "direct" else ROLEPLAY_COLOR


def best_d_across_layers(selector_data: dict, probe: str, prompt: str, followup: str) -> float:
    """Get the best (max absolute) Cohen's d across layers, preserving sign."""
    layers_data = selector_data[probe][prompt][followup]
    ds = [layers_data[l]["cohens_d"] for l in LAYERS if l in layers_data]
    if not ds:
        return 0.0
    # Return the d with the largest absolute value, preserving sign
    return max(ds, key=abs)


def best_d_for_layer(selector_data: dict, probe: str, prompt: str, followup: str, layer: str) -> float:
    return selector_data[probe][prompt][followup][layer]["cohens_d"]


def short_name(name: str) -> str:
    """Shorten prompt names for display."""
    return name.replace("direct_", "d:").replace("roleplay_", "rp:").replace("lie_", "")


# ── Plot 1: Cross-prompt comparison ──

def plot_cross_prompt_comparison():
    sel_data = data["assistant_selectors"]["assistant_tb:-1"]
    prompt_ds = []
    for p in PROMPTS:
        d = best_d_across_layers(sel_data, "tb-2", p, "minimal")
        prompt_ds.append((p, d))
    prompt_ds.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    names = [short_name(p) for p, _ in prompt_ds]
    ds = [d for _, d in prompt_ds]
    colors = [get_color(p) for p, _ in prompt_ds]

    bars = ax.bar(range(len(names)), ds, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=NO_SYSPROMPT_BASELINE, color="gray", linestyle="--", linewidth=1.5,
               label=f"No system prompt baseline (d={NO_SYSPROMPT_BASELINE})")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Best Cohen's d (across layers)")
    ax.set_title("Truth probe separation by lying system prompt\n(assistant_tb:-1, tb-2 probe)")
    ax.set_ylim(0, max(NO_SYSPROMPT_BASELINE * 1.15, max(ds) * 1.15))

    # Legend for prompt types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=DIRECT_COLOR, label="Direct"),
        Patch(facecolor=ROLEPLAY_COLOR, label="Roleplay"),
        ax.get_lines()[0],
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(ASSETS / "plot_031226_cross_prompt_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved plot_031226_cross_prompt_comparison.png")


# ── Plot 2: Layer × prompt heatmap ──

def plot_layer_prompt_heatmap():
    sel_data = data["assistant_selectors"]["assistant_tb:-1"]

    # Sort prompts by best d
    prompt_best = []
    for p in PROMPTS:
        d = best_d_across_layers(sel_data, "tb-2", p, "minimal")
        prompt_best.append((p, d))
    prompt_best.sort(key=lambda x: x[1], reverse=True)
    sorted_prompts = [p for p, _ in prompt_best]

    matrix = np.zeros((len(sorted_prompts), len(LAYERS)))
    for i, p in enumerate(sorted_prompts):
        for j, l in enumerate(LAYERS):
            matrix[i, j] = best_d_for_layer(sel_data, "tb-2", p, "minimal", l)

    vmax = max(abs(matrix.min()), abs(matrix.max()), NO_SYSPROMPT_BASELINE)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f"Layer {l}" for l in LAYERS])
    ax.set_yticks(range(len(sorted_prompts)))
    ylabels = [short_name(p) for p in sorted_prompts]
    ycolors = [get_color(p) for p in sorted_prompts]
    ax.set_yticklabels(ylabels)
    for tick_label, color in zip(ax.get_yticklabels(), ycolors):
        tick_label.set_color(color)

    ax.set_title("Cohen's d by layer and lying prompt\n(assistant_tb:-1, tb-2 probe)")
    fig.colorbar(im, ax=ax, label="Cohen's d", shrink=0.8)

    fig.tight_layout()
    fig.savefig(ASSETS / "plot_031226_layer_prompt_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved plot_031226_layer_prompt_heatmap.png")


# ── Plot 3: Selector × prompt heatmap ──

def plot_selector_prompt_heatmap():
    assistant_selectors = list(data["assistant_selectors"].keys())

    # Sort prompts by best d from assistant_tb:-1
    ref_data = data["assistant_selectors"]["assistant_tb:-1"]
    prompt_best = []
    for p in PROMPTS:
        d = best_d_across_layers(ref_data, "tb-2", p, "minimal")
        prompt_best.append((p, d))
    prompt_best.sort(key=lambda x: x[1], reverse=True)
    sorted_prompts = [p for p, _ in prompt_best]

    matrix = np.zeros((len(sorted_prompts), len(assistant_selectors)))
    for i, p in enumerate(sorted_prompts):
        for j, sel in enumerate(assistant_selectors):
            sel_data = data["assistant_selectors"][sel]
            matrix[i, j] = best_d_across_layers(sel_data, "tb-2", p, "minimal")

    vmax = max(abs(matrix.min()), abs(matrix.max()))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticks(range(len(assistant_selectors)))
    ax.set_xticklabels([s.replace("assistant_", "") for s in assistant_selectors], rotation=45, ha="right")
    ax.set_yticks(range(len(sorted_prompts)))
    ylabels = [short_name(p) for p in sorted_prompts]
    ycolors = [get_color(p) for p in sorted_prompts]
    ax.set_yticklabels(ylabels)
    for tick_label, color in zip(ax.get_yticklabels(), ycolors):
        tick_label.set_color(color)

    ax.set_title("Best Cohen's d by selector position and lying prompt\n(tb-2 probe, best across layers)")
    fig.colorbar(im, ax=ax, label="Cohen's d", shrink=0.8)

    fig.tight_layout()
    fig.savefig(ASSETS / "plot_031226_selector_prompt_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved plot_031226_selector_prompt_heatmap.png")


# ── Plot 4: User TB follow-up interaction ──

def plot_user_tb_followup_interaction():
    sel_data = data["user_tb_selectors"]["turn_boundary:-2"]
    followups = ["neutral", "presupposes", "challenge"]
    followup_colors = {"neutral": "#2ca02c", "presupposes": "#ff7f0e", "challenge": "#9467bd"}

    # Sort prompts by best d from assistant_tb:-1 for consistency
    ref_data = data["assistant_selectors"]["assistant_tb:-1"]
    prompt_best = []
    for p in PROMPTS:
        d = best_d_across_layers(ref_data, "tb-2", p, "minimal")
        prompt_best.append((p, d))
    prompt_best.sort(key=lambda x: x[1], reverse=True)
    sorted_prompts = [p for p, _ in prompt_best]

    fig, ax = plt.subplots(figsize=(14, 6))
    bar_width = 0.25
    x = np.arange(len(sorted_prompts))

    for k, fu in enumerate(followups):
        ds = []
        for p in sorted_prompts:
            d = best_d_across_layers(sel_data, "tb-2", p, fu)
            ds.append(d)
        ax.bar(x + k * bar_width, ds, bar_width, label=fu, color=followup_colors[fu],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([short_name(p) for p in sorted_prompts], rotation=45, ha="right", fontsize=9)
    # Color x-tick labels by type
    for tick_label, p in zip(ax.get_xticklabels(), sorted_prompts):
        tick_label.set_color(get_color(p))

    ax.set_ylabel("Best Cohen's d (across layers)")
    ax.set_title("User turn-boundary probe separation by follow-up type\n(turn_boundary:-2, tb-2 probe)")
    all_vals = []
    for fu in followups:
        for p in sorted_prompts:
            all_vals.append(best_d_across_layers(sel_data, "tb-2", p, fu))
    y_min = min(all_vals)
    y_max = max(all_vals)
    margin = max(abs(y_min), abs(y_max)) * 0.15
    ax.set_ylim(min(y_min - margin, -0.5), max(y_max + margin, 0.5))
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(title="Follow-up type")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(ASSETS / "plot_031226_user_tb_followup_interaction.png", dpi=150)
    plt.close(fig)
    print("Saved plot_031226_user_tb_followup_interaction.png")


if __name__ == "__main__":
    plot_cross_prompt_comparison()
    plot_layer_prompt_heatmap()
    plot_selector_prompt_heatmap()
    plot_user_tb_followup_interaction()
    print("All plots saved.")
