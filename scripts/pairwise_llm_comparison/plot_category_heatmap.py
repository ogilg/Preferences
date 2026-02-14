import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CATEGORIES = ["A_pairwise", "B_rating", "C_completion", "D_valence", "E_neutral", "F_affect"]
DIMENSIONS = ["emotional_engagement", "hedging", "elaboration", "confidence"]
SCORE_FIELDS = ["emotional_engagement_score", "hedging_score", "elaboration_score", "confidence_score"]

ORIGINAL_PATH = Path(__file__).parent / "judge_results.json"
SWAPPED_PATH = Path(__file__).parent / "judge_results_swapped.json"
OUTPUT_PATH = Path(
    "/workspace/repo/experiments/steering/program/open_ended_effects/"
    "pairwise_llm_comparison/assets/plot_021426_category_direction_heatmap.png"
)


def load_and_filter(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return [e for e in data if "error" not in e]


def compute_direction_asymmetry(entries: list[dict]) -> dict[str, float]:
    """For a list of entries for one prompt, compute score_neg - score_pos per dimension."""
    scores_by_coef: dict[int, list[dict]] = defaultdict(list)
    for e in entries:
        scores_by_coef[e["steered_coefficient"]].append(e)

    result = {}
    for dim_score in SCORE_FIELDS:
        neg_scores = [e[dim_score] for e in scores_by_coef[-3000]]
        pos_scores = [e[dim_score] for e in scores_by_coef[3000]]
        neg_mean = np.mean(neg_scores) if neg_scores else 0.0
        pos_mean = np.mean(pos_scores) if pos_scores else 0.0
        result[dim_score] = neg_mean - pos_mean
    return result


def main():
    original = load_and_filter(ORIGINAL_PATH)
    swapped = load_and_filter(SWAPPED_PATH)

    # Group by (category, prompt_id) across both datasets
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in original + swapped:
        grouped[(entry["category"], entry["prompt_id"])].append(entry)

    # Compute per-category, per-dimension mean direction asymmetry
    cat_dim_values: dict[str, dict[str, list[float]]] = {
        cat: {sf: [] for sf in SCORE_FIELDS} for cat in CATEGORIES
    }
    prompts_per_cat: dict[str, set[str]] = defaultdict(set)

    for (cat, prompt_id), entries in grouped.items():
        asymmetry = compute_direction_asymmetry(entries)
        for sf in SCORE_FIELDS:
            cat_dim_values[cat][sf].append(asymmetry[sf])
        prompts_per_cat[cat].add(prompt_id)

    # Build matrix
    matrix = np.zeros((len(CATEGORIES), len(DIMENSIONS)))
    for i, cat in enumerate(CATEGORIES):
        for j, sf in enumerate(SCORE_FIELDS):
            values = cat_dim_values[cat][sf]
            matrix[i, j] = np.mean(values) if values else 0.0

    n_prompts = [len(prompts_per_cat[cat]) for cat in CATEGORIES]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    vmax = max(abs(matrix.min()), abs(matrix.max()))
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    # Annotate cells
    for i in range(len(CATEGORIES)):
        for j in range(len(DIMENSIONS)):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.6 * vmax else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=11)

    # Axis labels
    row_labels = [f"{cat} (n={n_prompts[i]})" for i, cat in enumerate(CATEGORIES)]
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xticks(range(len(DIMENSIONS)))
    ax.set_xticklabels([d.replace("_", " ").title() for d in DIMENSIONS], fontsize=11)

    ax.set_title(
        "Direction Asymmetry by Category and Dimension (combined original + swapped)",
        fontsize=13,
        pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Direction Asymmetry (neg - pos)", fontsize=10)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
