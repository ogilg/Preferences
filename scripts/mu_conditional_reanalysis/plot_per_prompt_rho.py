"""Plot 3: Per-prompt Spearman rho(coefficient, response_length) as horizontal bar chart.

Bars colored by category. Sorted by rho. Significance markers added.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DATA_PATH = Path("experiments/steering/program/coefficient_calibration/generation_results.json")
OUT_PATH = Path("experiments/steering/program/open_ended_effects/mu_conditional_reanalysis/assets/plot_021426_per_prompt_length_rho.png")

COEF_RANGE = [-5000, -3000, -2000, -1000, -500, 0, 500, 1000, 2000, 3000, 5000]
CATEGORY_COLORS = {
    "B_rating": "#e66101",
    "C_completion": "#4dac26",
    "D_valence": "#2166ac",
    "E_neutral": "#888888",
    "F_affect": "#b2182b",
}
CATEGORY_SHORT = {
    "B_rating": "B",
    "C_completion": "C",
    "D_valence": "D",
    "E_neutral": "E",
    "F_affect": "F",
}
# A_pairwise excluded (choice output, not free generation)
INCLUDE_CATEGORIES = ["B_rating", "C_completion", "D_valence", "E_neutral", "F_affect"]


def sig_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    results = [r for r in data["results"] if r["coefficient"] in COEF_RANGE]
    prompts = data["prompts"]

    # Build prompt metadata lookup
    prompt_meta = {p["prompt_id"]: p for p in prompts}

    # Compute rho per prompt
    prompt_rhos = []
    for p in prompts:
        if p["category"] not in INCLUDE_CATEGORIES:
            continue
        pid = p["prompt_id"]
        subset = [r for r in results if r["prompt_id"] == pid]
        if len(subset) < 5:
            continue

        coeffs = [r["coefficient"] for r in subset]
        lengths = [r["response_length"] for r in subset]
        if len(set(lengths)) <= 1:
            continue
        rho, pval = stats.spearmanr(coeffs, lengths)

        # Build a readable label
        cat_short = CATEGORY_SHORT[p["category"]]
        meta = p["metadata"]
        if "prompt_text" in meta:
            label = f"[{cat_short}] {meta['prompt_text'][:45]}"
        elif "task_id" in meta:
            mu_str = f" (μ={meta['mu']:.1f})" if "mu" in meta else ""
            label = f"[{cat_short}] {meta['task_id']}{mu_str}"
        else:
            label = f"[{cat_short}] {pid}"

        prompt_rhos.append({
            "prompt_id": pid,
            "category": p["category"],
            "rho": rho,
            "pval": pval,
            "label": label,
            "sig": sig_marker(pval),
        })

    # Sort by rho
    prompt_rhos.sort(key=lambda x: x["rho"])

    n = len(prompt_rhos)
    fig_height = max(8, n * 0.32)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = np.arange(n)
    bar_colors = [CATEGORY_COLORS[pr["category"]] for pr in prompt_rhos]
    rho_vals = [pr["rho"] for pr in prompt_rhos]

    bars = ax.barh(y_positions, rho_vals, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.7)

    # Labels and significance markers
    for i, pr in enumerate(prompt_rhos):
        if pr["sig"]:
            x_offset = 0.01 if pr["rho"] >= 0 else -0.01
            ha = "left" if pr["rho"] >= 0 else "right"
            ax.text(
                pr["rho"] + x_offset, i, pr["sig"],
                ha=ha, va="center", fontsize=9, fontweight="bold", color="#333",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([pr["label"] for pr in prompt_rhos], fontsize=7.5)
    ax.axvline(0, color="#333", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("Spearman ρ (coefficient vs. response length)", fontsize=11)
    ax.set_title("Per-Prompt Sensitivity: Coefficient → Response Length", fontsize=13, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS[cat], label=f"{CATEGORY_SHORT[cat]}: {cat.split('_', 1)[1].replace('_', ' ').title()}")
        for cat in INCLUDE_CATEGORIES
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    ax.set_xlim(-1.05, 1.05)
    ax.tick_params(axis="y", which="both", length=0)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
