"""Regenerate all 4 plots for the lying prompts experiment."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "experiments/truth_probes/error_prefill/lying_prompts/lying_10prompt_results.json"
ASSETS_DIR = ROOT / "experiments/truth_probes/error_prefill/lying_prompts/assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

SHORT_NAMES = {
    "lie_direct": "d:direct",
    "direct_please_lie": "d:please_lie",
    "direct_opposite_day": "d:opposite_day",
    "direct_wrong": "d:wrong",
    "direct_mislead": "d:mislead",
    "lie_roleplay": "rp:generic",
    "roleplay_villain": "rp:villain",
    "roleplay_sadist": "rp:sadist",
    "roleplay_trickster": "rp:trickster",
    "roleplay_exam": "rp:exam",
}

CLUSTERS = {
    "Signal preserved": {"prompts": ["rp:villain", "rp:sadist"], "color": "#2ca02c"},
    "Moderate disruption": {"prompts": ["rp:generic", "d:opposite_day", "d:mislead"], "color": "#ff7f0e"},
    "Signal eliminated": {"prompts": ["d:direct", "d:please_lie", "d:wrong", "rp:trickster", "rp:exam"], "color": "#d62728"},
}

PROMPT_TO_CLUSTER_COLOR = {}
for cluster_info in CLUSTERS.values():
    for p in cluster_info["prompts"]:
        PROMPT_TO_CLUSTER_COLOR[p] = cluster_info["color"]

SORT_ORDER = [
    "rp:villain", "rp:sadist", "rp:generic", "d:opposite_day", "d:mislead",
    "rp:trickster", "rp:exam", "d:please_lie", "d:direct", "d:wrong",
]

LAYERS = ["25", "32", "39", "46", "53"]

SELECTOR_LABELS = {
    "assistant_mean": "mean",
    "assistant_tb:-1": "tb:-1 (\\n)",
    "assistant_tb:-2": "tb:-2 (model)",
    "assistant_tb:-3": "tb:-3 (<start_of_turn>)",
    "assistant_tb:-4": "tb:-4 (\\n)",
    "assistant_tb:-5": "tb:-5 (last asst token)",
}


def load_data():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def best_d_across_layers(layer_dict: dict) -> float:
    return max(layer_dict[l]["cohens_d"] for l in LAYERS)


def get_prompt_layer_data(data, selector, probe, followup):
    """Return {short_name: {layer: cohens_d}} for all prompts."""
    sel_data = data["assistant_selectors"][selector][probe]
    result = {}
    for prompt_key, short in SHORT_NAMES.items():
        if prompt_key in sel_data and followup in sel_data[prompt_key]:
            result[short] = {l: sel_data[prompt_key][followup][l]["cohens_d"] for l in LAYERS}
    return result


def plot1_cross_prompt_comparison(data):
    """Horizontal bar chart of best Cohen's d by prompt."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    prompt_data = get_prompt_layer_data(data, "assistant_tb:-1", "tb-2", "minimal")
    best_ds = {name: max(layers.values()) for name, layers in prompt_data.items()}

    # Reverse sort order for horizontal bars (highest at top)
    prompts = list(reversed(SORT_ORDER))
    values = [best_ds[p] for p in prompts]
    colors = [PROMPT_TO_CLUSTER_COLOR[p] for p in prompts]

    ax.barh(range(len(prompts)), values, color=colors)
    ax.set_yticks(range(len(prompts)))
    ax.set_yticklabels(prompts)
    ax.set_xlabel("Best Cohen's d (across layers)")
    ax.set_title("Cross-prompt comparison: truth probe separation under lying instructions")
    ax.axvline(x=3.29, color="gray", linestyle="--", linewidth=1.5, label="No-system-prompt baseline (d=3.29)")

    # Legend for clusters
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CLUSTERS[name]["color"], label=name)
        for name in CLUSTERS
    ]
    legend_elements.append(plt.Line2D([0], [0], color="gray", linestyle="--", label="No-system-prompt baseline"))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    out = ASSETS_DIR / "plot_031226_cross_prompt_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot2_layer_prompt_heatmap(data):
    """Heatmap: prompts x layers."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    prompt_data = get_prompt_layer_data(data, "assistant_tb:-1", "tb-2", "minimal")

    prompts = SORT_ORDER
    matrix = np.array([[prompt_data[p][l] for l in LAYERS] for p in prompts])

    vmax = max(abs(matrix.min()), abs(matrix.max()))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    # Annotate cells
    for i in range(len(prompts)):
        for j in range(len(LAYERS)):
            val = matrix[i, j]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=text_color)

    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f"Layer {l}" for l in LAYERS])
    ax.set_yticks(range(len(prompts)))
    ax.set_yticklabels(prompts)
    ax.set_title("Cohen's d by layer and lying prompt (assistant tb:-1, tb-2 probe, minimal followup)")
    fig.colorbar(im, ax=ax, label="Cohen's d")

    plt.tight_layout()
    out = ASSETS_DIR / "plot_031226_layer_prompt_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot3_selector_prompt_heatmap(data):
    """Heatmap: prompts x selector positions."""
    plt.style.use("seaborn-v0_8-whitegrid")

    selectors = ["assistant_mean", "assistant_tb:-1", "assistant_tb:-2", "assistant_tb:-3", "assistant_tb:-4", "assistant_tb:-5"]
    prompts = SORT_ORDER

    matrix = np.zeros((len(prompts), len(selectors)))
    for j, sel in enumerate(selectors):
        sel_data = data["assistant_selectors"][sel]["tb-2"]
        for i, short in enumerate(prompts):
            # Find the original prompt key
            prompt_key = [k for k, v in SHORT_NAMES.items() if v == short][0]
            if prompt_key in sel_data and "minimal" in sel_data[prompt_key]:
                layer_dict = sel_data[prompt_key]["minimal"]
                matrix[i, j] = max(layer_dict[l]["cohens_d"] for l in LAYERS)

    fig, ax = plt.subplots(figsize=(12, 6))
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    for i in range(len(prompts)):
        for j in range(len(selectors)):
            val = matrix[i, j]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=text_color)

    col_labels = [SELECTOR_LABELS[s] for s in selectors]
    ax.set_xticks(range(len(selectors)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(prompts)))
    ax.set_yticklabels(prompts)
    ax.set_title("Best Cohen's d by selector position and lying prompt (tb-2 probe, minimal followup)")
    fig.colorbar(im, ax=ax, label="Best Cohen's d")

    plt.tight_layout()
    out = ASSETS_DIR / "plot_031226_selector_prompt_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot4_user_tb_followup(data):
    """Grouped bar chart: followup interaction at user turn-boundary."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    followups = ["neutral", "presupposes", "challenge"]
    followup_colors = {"neutral": "#2ca02c", "presupposes": "#ff7f0e", "challenge": "#9467bd"}
    baselines = {"neutral": 1.80, "presupposes": 2.58, "challenge": 1.03}

    prompts = SORT_ORDER
    sel_data = data["user_tb_selectors"]["turn_boundary:-2"]["tb-2"]

    x = np.arange(len(prompts))
    width = 0.25

    for k, followup in enumerate(followups):
        values = []
        for short in prompts:
            prompt_key = [pk for pk, v in SHORT_NAMES.items() if v == short][0]
            layer_dict = sel_data[prompt_key][followup]
            values.append(max(layer_dict[l]["cohens_d"] for l in LAYERS))
        ax.bar(x + (k - 1) * width, values, width, label=followup, color=followup_colors[followup])

    # Baseline lines
    for followup in followups:
        ax.axhline(
            y=baselines[followup], color=followup_colors[followup], linestyle="--", linewidth=1.5, alpha=0.7,
            label=f"{followup} baseline (d={baselines[followup]:.2f})",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(prompts, rotation=45, ha="right")
    ax.set_ylabel("Best Cohen's d (across layers)")
    ax.set_title("Follow-up interaction at user turn-boundary (turn_boundary:-2, tb-2 probe)")
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    out = ASSETS_DIR / "plot_031226_user_tb_followup_interaction.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_user_tb_followup_fixed_layer(data, layer: str):
    """Grouped bar chart: followup interaction at user turn-boundary, single layer."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    followups = ["neutral", "presupposes", "challenge"]
    followup_colors = {"neutral": "#2ca02c", "presupposes": "#ff7f0e", "challenge": "#9467bd"}
    baselines = {"neutral": 1.80, "presupposes": 2.58, "challenge": 1.03}

    prompts = SORT_ORDER
    sel_data = data["user_tb_selectors"]["turn_boundary:-2"]["tb-2"]

    x = np.arange(len(prompts))
    width = 0.25

    for k, followup in enumerate(followups):
        values = []
        for short in prompts:
            prompt_key = [pk for pk, v in SHORT_NAMES.items() if v == short][0]
            values.append(sel_data[prompt_key][followup][layer]["cohens_d"])
        ax.bar(x + (k - 1) * width, values, width, label=followup, color=followup_colors[followup])

    for followup in followups:
        ax.axhline(
            y=baselines[followup], color=followup_colors[followup], linestyle="--", linewidth=1.5, alpha=0.7,
            label=f"{followup} baseline (d={baselines[followup]:.2f})",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(prompts, rotation=45, ha="right")
    ax.set_ylabel(f"Cohen's d (layer {layer})")
    ax.set_title(f"Follow-up interaction at user turn-boundary (turn_boundary:-2, tb-2 probe, layer {layer})")
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    out = ASSETS_DIR / f"plot_031226_user_tb_followup_L{layer}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    data = load_data()
    plot1_cross_prompt_comparison(data)
    plot2_layer_prompt_heatmap(data)
    plot3_selector_prompt_heatmap(data)
    plot4_user_tb_followup(data)
    plot_user_tb_followup_fixed_layer(data, "32")
    plot_user_tb_followup_fixed_layer(data, "53")
    print("All plots generated.")
