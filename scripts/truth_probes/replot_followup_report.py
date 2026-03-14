"""Replot figures for the error prefill follow-up report.

Fixes:
- Effect sizes plot: make overlapping lines visually explicit (thick line + annotation)
- Answer-turn heatmap: unified colorbar across sub-heatmaps
- Follow-up turn heatmap: fill no_lying row with original experiment baselines
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

matplotlib.rcParams.update({"font.size": 11})

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "experiments" / "truth_probes" / "error_prefill" / "error_prefill_followup_results.json"
ASSETS_DIR = ROOT / "experiments" / "truth_probes" / "error_prefill" / "assets"

with open(RESULTS_PATH) as f:
    results = json.load(f)

asst = results["assistant_selectors_no_lying"]
lying = results["lying_conversations"]

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
FOLLOWUP_LABELS = {
    "neutral": '"Thank you"',
    "presupposes": "Presupposes answer",
    "challenge": '"Are you sure?"',
    "same_domain": "Same topic",
    "control": "Unrelated task",
}

# Original experiment per-layer baselines for turn_boundary selectors (presupposes, tb-2 probe)
ORIG_TB2_PRESUPPOSES = {
    "turn_boundary:-2": {"25": 2.28, "32": 2.27, "39": 2.51, "46": 2.58, "53": 2.58},
    "turn_boundary:-5": {"25": 1.92, "32": 2.19, "39": 2.14, "46": 2.33, "53": 2.33},
}
ORIG_TB2_NEUTRAL = {
    "turn_boundary:-2": {"25": 1.31, "32": 1.70, "39": 1.27, "46": 1.62, "53": 1.80},
    "turn_boundary:-5": {"25": 1.87, "32": 1.79, "39": 0.62, "46": 1.99, "53": 1.39},
}


def get_d(data, layers=LAYERS):
    return [data[l]["cohens_d"] for l in layers]


def get_auc(data, layers=LAYERS):
    return [data[l]["auc"] for l in layers]


def best_d(data, layers=LAYERS):
    return max(data[l]["cohens_d"] for l in layers)


# =============================================================================
# Plot 1: Effect sizes — answer-turn selectors (tb-2 probe)
# Show that all follow-up types overlap, with explicit annotation
# =============================================================================

def plot_effect_sizes_tb2():
    selectors = ["assistant_mean", "assistant_tb:-1", "assistant_tb:-5"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Answer-turn selectors: correct vs incorrect (tb-2 probe)", fontsize=14)

    for ax, selector in zip(axes, selectors):
        sel_data = asst[selector]["tb-2"]
        # Plot all 5 follow-ups to show they overlap
        for followup in FOLLOWUPS:
            ds = get_d(sel_data[followup])
            ax.plot(LAYERS_INT, ds, marker="o", label=FOLLOWUP_LABELS[followup],
                    color=FOLLOWUP_COLORS[followup], linewidth=2, alpha=0.7, markersize=6)

        ax.set_title(selector, fontsize=12)
        ax.set_xlabel("Layer")
        ax.set_xticks(LAYERS_INT)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    axes[0].set_ylabel("Cohen's d (correct - incorrect)")
    axes[0].set_ylim(-0.5, 3.5)
    axes[-1].legend(loc="upper right", fontsize=8, title="Follow-up type", title_fontsize=9)

    # Add annotation explaining the overlap
    fig.text(0.5, -0.02, "All five follow-up types produce identical d values because these selectors read from the answer turn, before any follow-up.",
             ha="center", fontsize=10, style="italic", color="#555555")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(ASSETS_DIR / "plot_031226_assistant_selectors_effect_sizes_tb-2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: assistant_selectors_effect_sizes_tb-2")


# =============================================================================
# Plot 2: Answer turn vs follow-up turn comparison (bar chart)
# Already good, just update labels to be more descriptive
# =============================================================================

def plot_answer_vs_followup():
    selector_data = {
        "turn_boundary:-2\n(follow-up turn)": 2.58,
        "turn_boundary:-5\n(follow-up turn)": 2.33,
        "assistant_mean\n(answer turn)": best_d(asst["assistant_mean"]["tb-2"]["presupposes"]),
        "assistant_tb:-1\n(answer turn)": best_d(asst["assistant_tb:-1"]["tb-2"]["presupposes"]),
        "assistant_tb:-5\n(answer turn)": best_d(asst["assistant_tb:-5"]["tb-2"]["presupposes"]),
    }

    labels = list(selector_data.keys())
    values = list(selector_data.values())
    colors = ["#4c72b0", "#4c72b0", "#dd8452", "#dd8452", "#dd8452"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Best Cohen's d across layers")
    ax.set_title("Correct vs incorrect separation by selector position\n(presupposes condition, tb-2 probe)")
    ax.set_ylim(0, max(values) * 1.15)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_031226_assistant_vs_followup_presupposes_tb-2.png", dpi=150)
    plt.close(fig)
    print("Saved: assistant_vs_followup_presupposes_tb-2")


# =============================================================================
# Plot 3: Answer-turn heatmap with UNIFIED colorbar
# =============================================================================

def plot_answer_turn_heatmap(followup="presupposes", probe="tb-2"):
    conditions = ["no_lying", "lie_direct", "lie_roleplay"]
    condition_labels = ["No lying", "Direct lying", "Roleplay lying"]
    selectors = ["assistant_mean", "assistant_tb:-1", "assistant_tb:-5"]

    # Build matrices
    matrices = {}
    for selector in selectors:
        matrix = np.full((3, len(LAYERS)), np.nan)
        for row_idx, condition in enumerate(conditions):
            if condition == "no_lying":
                if selector in asst and probe in asst[selector]:
                    if followup in asst[selector][probe]:
                        for col_idx, layer in enumerate(LAYERS):
                            matrix[row_idx, col_idx] = asst[selector][probe][followup][layer]["cohens_d"]
            else:
                if selector in lying and probe in lying[selector]:
                    if condition in lying[selector][probe]:
                        if followup in lying[selector][probe][condition]:
                            for col_idx, layer in enumerate(LAYERS):
                                matrix[row_idx, col_idx] = lying[selector][probe][condition][followup][layer]["cohens_d"]
        matrices[selector] = matrix

    # Find global min/max for unified colorbar
    all_vals = np.concatenate([m.flatten() for m in matrices.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmin = min(-1.5, np.min(all_vals) - 0.2)
    vmax = max(3.5, np.max(all_vals) + 0.2)

    fig, axes = plt.subplots(len(selectors), 1, figsize=(10, 3.5 * len(selectors)), squeeze=False)
    fig.suptitle(f"Cohen's d: answer-turn selectors ({followup}, {probe} probe)",
                 fontsize=13, y=1.02)

    for ax_idx, selector in enumerate(selectors):
        ax = axes[ax_idx, 0]
        matrix = matrices[selector]
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(LAYERS)))
        ax.set_xticklabels([f"L{l}" for l in LAYERS_INT])
        ax.set_yticks(range(3))
        ax.set_yticklabels(condition_labels)
        ax.set_title(selector, fontsize=11)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < 0.3 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=10, color=text_color, fontweight="bold")

    # Single colorbar for all subplots
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Cohen's d", shrink=0.8, pad=0.04)

    fig.savefig(ASSETS_DIR / f"plot_031226_lying_heatmap_answer_turn_{followup}_{probe}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: lying_heatmap_answer_turn_{followup}_{probe}")


# =============================================================================
# Plot 4: Follow-up turn heatmap — fill no_lying with original baselines
# =============================================================================

def plot_followup_turn_heatmap(followup="presupposes", probe="tb-2"):
    conditions = ["no_lying", "lie_direct", "lie_roleplay"]
    condition_labels = ["No lying (from original exp.)", "Direct lying", "Roleplay lying"]
    selectors = ["turn_boundary:-2", "turn_boundary:-5"]

    orig_baselines = ORIG_TB2_PRESUPPOSES if followup == "presupposes" else ORIG_TB2_NEUTRAL

    matrices = {}
    for selector in selectors:
        matrix = np.full((3, len(LAYERS)), np.nan)
        for row_idx, condition in enumerate(conditions):
            if condition == "no_lying":
                # Fill from original experiment baselines
                if selector in orig_baselines:
                    for col_idx, layer in enumerate(LAYERS):
                        matrix[row_idx, col_idx] = orig_baselines[selector][layer]
            else:
                if selector in lying and probe in lying[selector]:
                    if condition in lying[selector][probe]:
                        if followup in lying[selector][probe][condition]:
                            for col_idx, layer in enumerate(LAYERS):
                                matrix[row_idx, col_idx] = lying[selector][probe][condition][followup][layer]["cohens_d"]
        matrices[selector] = matrix

    all_vals = np.concatenate([m.flatten() for m in matrices.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmin = min(-0.5, np.min(all_vals) - 0.2)
    vmax = max(3.0, np.max(all_vals) + 0.2)

    fig, axes = plt.subplots(len(selectors), 1, figsize=(9, 3.5 * len(selectors)), squeeze=False)
    fig.suptitle(f"Cohen's d: follow-up turn selectors ({followup}, {probe} probe)",
                 fontsize=13, y=1.02)

    for ax_idx, selector in enumerate(selectors):
        ax = axes[ax_idx, 0]
        matrix = matrices[selector]
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(LAYERS)))
        ax.set_xticklabels([f"L{l}" for l in LAYERS_INT])
        ax.set_yticks(range(3))
        ax.set_yticklabels(condition_labels)
        ax.set_title(selector, fontsize=11)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < 0.3 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=10, color=text_color, fontweight="bold")

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Cohen's d", shrink=0.8, pad=0.04)

    fig.savefig(ASSETS_DIR / f"plot_031226_lying_heatmap_followup_turn_{followup}_{probe}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: lying_heatmap_followup_turn_{followup}_{probe}")


# =============================================================================
# Plot 5: Comprehensive AUC heatmap — cleaner labels
# =============================================================================

def plot_comprehensive_auc(followup="presupposes", probe="tb-2"):
    row_configs = []

    # No-lying assistant selectors
    for sel in ["assistant_mean", "assistant_tb:-1", "assistant_tb:-5"]:
        if sel in asst and probe in asst[sel] and followup in asst[sel][probe]:
            aucs = get_auc(asst[sel][probe][followup])
            row_configs.append((f"{sel} (no lying)", aucs))

    # Lying: both TB and assistant selectors
    for sel in ["turn_boundary:-2", "turn_boundary:-5", "assistant_mean", "assistant_tb:-1", "assistant_tb:-5"]:
        for sys_type in ["lie_direct", "lie_roleplay"]:
            label = "direct" if sys_type == "lie_direct" else "roleplay"
            if sel in lying and probe in lying[sel]:
                if sys_type in lying[sel][probe] and followup in lying[sel][probe][sys_type]:
                    aucs = get_auc(lying[sel][probe][sys_type][followup])
                    row_configs.append((f"{sel} ({label} lying)", aucs))

    labels = [r[0] for r in row_configs]
    matrix = np.array([r[1] for r in row_configs])

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.6 + 1)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f"L{l}" for l in LAYERS_INT])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Add chance level reference line
    ax.axvline(-0.5, color="black", linewidth=0.5)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if val < 0.4 or val > 0.85 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=text_color)

    ax.set_title(f"AUC: all selectors and conditions ({followup}, {probe} probe)\nChance = 0.50; below 0.50 = inverted classification",
                 fontsize=12)
    fig.colorbar(im, ax=ax, label="AUC", shrink=0.8)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / f"plot_031226_comprehensive_auc_{followup}_{probe}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: comprehensive_auc_{followup}_{probe}")


# =============================================================================
# Plot 6: Mean score shift — fix missing no_lying for TB selectors
# =============================================================================

def plot_mean_score_shift(followup="presupposes", probe="tb-2", layer="46"):
    selector = "assistant_tb:-1"
    conditions = ["no_lying", "lie_direct", "lie_roleplay"]
    condition_labels = ["No lying", "Direct\nlying", "Roleplay\nlying"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(conditions))
    width = 0.35

    correct_means = []
    incorrect_means = []

    for cond in conditions:
        if cond == "no_lying":
            data = asst[selector][probe][followup][layer]
        else:
            data = lying[selector][probe][cond][followup][layer]
        correct_means.append(data["mean_correct"])
        incorrect_means.append(data["mean_incorrect"])

    ax.bar(x - width / 2, correct_means, width, label="Correct answer", color="#2ecc71",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, incorrect_means, width, label="Incorrect answer", color="#e74c3c",
           edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, fontsize=10)
    ax.set_ylabel("Mean probe score")
    ax.set_title(f"Mean probe score by condition ({selector}, {probe} probe, L{layer}, {followup})")
    ax.legend(fontsize=10)

    # Add d values as annotations
    for i, cond in enumerate(conditions):
        if cond == "no_lying":
            d = asst[selector][probe][followup][layer]["cohens_d"]
        else:
            d = lying[selector][probe][cond][followup][layer]["cohens_d"]
        ax.text(i, max(correct_means[i], incorrect_means[i]) + 0.3,
                f"d = {d:.2f}", ha="center", fontsize=9, style="italic")

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / f"plot_031226_mean_score_shift_{followup}_{probe}.png", dpi=150)
    plt.close(fig)
    print(f"Saved: mean_score_shift_{followup}_{probe}")


if __name__ == "__main__":
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    plot_effect_sizes_tb2()
    plot_answer_vs_followup()
    plot_answer_turn_heatmap("presupposes", "tb-2")
    plot_followup_turn_heatmap("presupposes", "tb-2")
    plot_comprehensive_auc("presupposes", "tb-2")
    plot_mean_score_shift("presupposes", "tb-2", "46")

    print("\nAll plots saved to:", ASSETS_DIR)
