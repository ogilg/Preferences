"""Plot 1: Length dose-response by category (2x3 grid).

Each subplot shows mean response length vs coefficient for one category.
For B_rating and C_completion: lines split by LOW/MID/HIGH mu groups.
For D_valence, E_neutral, F_affect: thin per-prompt lines + thick category mean.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DATA_PATH = Path("experiments/steering/program/coefficient_calibration/generation_results.json")
OUT_PATH = Path("experiments/steering/program/open_ended_effects/mu_conditional_reanalysis/assets/plot_021426_length_dose_response_by_category.png")

COEF_RANGE = [-5000, -3000, -2000, -1000, -500, 0, 500, 1000, 2000, 3000, 5000]
CATEGORIES_ORDER = ["A_pairwise", "B_rating", "C_completion", "D_valence", "E_neutral", "F_affect"]
CATEGORY_LABELS = {
    "A_pairwise": "A: Pairwise Choice",
    "B_rating": "B: Post-Completion Rating",
    "C_completion": "C: Task Completion",
    "D_valence": "D: Valence Probes",
    "E_neutral": "E: Neutral / Factual",
    "F_affect": "F: Affect Elicitation",
}

MU_GROUP_COLORS = {"LOW": "#2166ac", "MID": "#4dac26", "HIGH": "#b2182b"}
MU_GROUP_LABELS = {"LOW": r"LOW ($\mu < -2$)", "MID": r"MID ($0 \leq \mu \leq 2$)", "HIGH": r"HIGH ($\mu > 4$)"}


def classify_mu(mu: float) -> str:
    if mu < -2:
        return "LOW"
    elif 0 <= mu <= 2:
        return "MID"
    elif mu > 4:
        return "HIGH"
    return "OTHER"


def compute_per_prompt_rho(results: list[dict], prompt_id: str) -> tuple[float, float]:
    """Spearman rho(coefficient, response_length) for a single prompt."""
    subset = [r for r in results if r["prompt_id"] == prompt_id and r["coefficient"] in COEF_RANGE]
    if len(subset) < 5:
        return 0.0, 1.0
    coeffs = [r["coefficient"] for r in subset]
    lengths = [r["response_length"] for r in subset]
    rho, p = stats.spearmanr(coeffs, lengths)
    return rho, p


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    prompts = data["prompts"]
    results = [r for r in data["results"] if r["coefficient"] in COEF_RANGE]

    prompt_meta = {p["prompt_id"]: p for p in prompts}

    # Build per-category, per-prompt, per-coefficient mean lengths
    # key: (category, prompt_id, coefficient) -> list of lengths
    length_map: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for r in results:
        length_map[(r["category"], r["prompt_id"], r["coefficient"])].append(r["response_length"])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, cat in enumerate(CATEGORIES_ORDER):
        ax = axes[idx]
        cat_prompts = [p for p in prompts if p["category"] == cat]
        prompt_ids = [p["prompt_id"] for p in cat_prompts]

        if cat in ("B_rating", "C_completion"):
            # Group by mu
            groups: dict[str, list[str]] = defaultdict(list)
            for p in cat_prompts:
                mu = p["metadata"]["mu"]
                g = classify_mu(mu)
                if g != "OTHER":
                    groups[g].append(p["prompt_id"])

            for g_name in ["LOW", "MID", "HIGH"]:
                g_pids = groups[g_name]
                if not g_pids:
                    continue
                means = []
                sems = []
                for coef in COEF_RANGE:
                    all_lengths = []
                    for pid in g_pids:
                        all_lengths.extend(length_map[(cat, pid, coef)])
                    means.append(np.mean(all_lengths) if all_lengths else 0)
                    sems.append(stats.sem(all_lengths) if len(all_lengths) > 1 else 0)
                ax.errorbar(
                    COEF_RANGE, means, yerr=sems,
                    color=MU_GROUP_COLORS[g_name], label=MU_GROUP_LABELS[g_name],
                    linewidth=2, marker="o", markersize=4, capsize=3,
                )

            # Compute and annotate per-group rho
            rho_texts = []
            for g_name in ["LOW", "MID", "HIGH"]:
                g_pids = groups[g_name]
                if not g_pids:
                    continue
                # Pool all data for the group
                coeffs_all, lengths_all = [], []
                for pid in g_pids:
                    for coef in COEF_RANGE:
                        for length in length_map[(cat, pid, coef)]:
                            coeffs_all.append(coef)
                            lengths_all.append(length)
                if len(coeffs_all) > 5 and len(set(lengths_all)) > 1:
                    rho, p = stats.spearmanr(coeffs_all, lengths_all)
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    rho_texts.append(f"{g_name}: ρ={rho:.2f}{sig}")
            if rho_texts:
                ax.text(
                    0.02, 0.98, "\n".join(rho_texts),
                    transform=ax.transAxes, fontsize=8, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )
            ax.legend(fontsize=8, loc="lower right")

        elif cat == "A_pairwise":
            # Just show category mean
            means = []
            sems = []
            for coef in COEF_RANGE:
                all_lengths = []
                for pid in prompt_ids:
                    all_lengths.extend(length_map[(cat, pid, coef)])
                means.append(np.mean(all_lengths) if all_lengths else 0)
                sems.append(stats.sem(all_lengths) if len(all_lengths) > 1 else 0)
            ax.errorbar(
                COEF_RANGE, means, yerr=sems,
                color="#333333", linewidth=2, marker="o", markersize=4, capsize=3,
                label="Category mean",
            )
            ax.legend(fontsize=8, loc="lower right")

        else:
            # D, E, F: thin per-prompt lines + thick mean
            cat_color = {"D_valence": "#2166ac", "E_neutral": "#888888", "F_affect": "#b2182b"}[cat]
            all_means_per_coef = []

            for pid in prompt_ids:
                per_coef_means = []
                for coef in COEF_RANGE:
                    lens = length_map[(cat, pid, coef)]
                    per_coef_means.append(np.mean(lens) if lens else 0)
                ax.plot(COEF_RANGE, per_coef_means, color=cat_color, alpha=0.25, linewidth=0.8)
                all_means_per_coef.append(per_coef_means)

                # Annotate significant per-prompt rho
                rho, p = compute_per_prompt_rho(results, pid)
                if p < 0.05:
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                    # Annotate at rightmost point
                    ax.annotate(
                        f"{pid.split('_')[1]}{sig}",
                        (COEF_RANGE[-1], per_coef_means[-1]),
                        fontsize=5, alpha=0.6, textcoords="offset points", xytext=(4, 0),
                    )

            # Category mean (thick line)
            mean_curve = np.mean(all_means_per_coef, axis=0)
            sem_curve = stats.sem(all_means_per_coef, axis=0)
            ax.errorbar(
                COEF_RANGE, mean_curve, yerr=sem_curve,
                color=cat_color, linewidth=2.5, marker="o", markersize=4, capsize=3,
                label="Category mean", zorder=10,
            )
            ax.legend(fontsize=8, loc="lower right")

        ax.set_title(CATEGORY_LABELS[cat], fontsize=11, fontweight="bold")
        ax.set_xlabel("Coefficient", fontsize=9)
        ax.set_ylabel("Mean response length (chars)", fontsize=9)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.tick_params(labelsize=8)
        ax.set_xlim(-5500, 5500)

    fig.suptitle("Response Length Dose-Response by Category", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
