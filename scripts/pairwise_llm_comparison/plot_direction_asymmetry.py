import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

RESULTS_DIR = Path("/workspace/repo/scripts/pairwise_llm_comparison")
OUTPUT_PATH = Path(
    "/workspace/repo/experiments/steering/program/open_ended_effects"
    "/pairwise_llm_comparison/assets/plot_021426_direction_asymmetry.png"
)

DIMENSIONS = ["emotional_engagement", "hedging", "elaboration", "confidence"]

CATEGORY_COLORS = {
    "A_pairwise": "#1f77b4",
    "B_rating": "#ff7f0e",
    "C_completion": "#2ca02c",
    "D_valence": "#d62728",
    "E_neutral": "#9467bd",
    "F_affect": "#8c564b",
}


def load_and_filter(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return [entry for entry in data if "error" not in entry]


def build_lookup(entries: list[dict]) -> dict[tuple[str, int], dict]:
    """Map (prompt_id, coefficient) -> entry."""
    lookup: dict[tuple[str, int], dict] = {}
    for entry in entries:
        key = (entry["prompt_id"], entry["steered_coefficient"])
        lookup[key] = entry
    return lookup


def compute_combined_diffs(
    original: list[dict], swapped: list[dict]
) -> dict[str, dict]:
    """Returns {prompt_id: {dim_score: avg_diff, category: str}}."""
    orig_lookup = build_lookup(original)
    swap_lookup = build_lookup(swapped)

    prompt_ids = sorted({e["prompt_id"] for e in original + swapped})
    results: dict[str, dict] = {}

    for pid in prompt_ids:
        orig_neg = orig_lookup.get((pid, -3000))
        orig_pos = orig_lookup.get((pid, 3000))
        swap_neg = swap_lookup.get((pid, -3000))
        swap_pos = swap_lookup.get((pid, 3000))

        if not all([orig_neg, orig_pos, swap_neg, swap_pos]):
            continue

        category = orig_neg["category"]
        row: dict[str, object] = {"category": category}

        for dim in DIMENSIONS:
            score_key = f"{dim}_score"
            orig_diff = orig_neg[score_key] - orig_pos[score_key]
            swap_diff = swap_neg[score_key] - swap_pos[score_key]
            row[score_key] = (orig_diff + swap_diff) / 2.0

        results[pid] = row

    return results


def sign_test_pvalue(diffs: list[float]) -> float:
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    n_nonzero = n_pos + n_neg
    if n_nonzero == 0:
        return 1.0
    return binomtest(n_pos, n_nonzero, 0.5).pvalue


def main() -> None:
    original = load_and_filter(RESULTS_DIR / "judge_results.json")
    swapped = load_and_filter(RESULTS_DIR / "judge_results_swapped.json")

    combined = compute_combined_diffs(original, swapped)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Pairwise Judge: Direction Asymmetry (score at \u22123000 minus score at +3000)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for ax, dim in zip(axes.flat, DIMENSIONS):
        score_key = f"{dim}_score"

        prompt_ids = sorted(combined.keys(), key=lambda pid: combined[pid][score_key])
        values = [combined[pid][score_key] for pid in prompt_ids]
        categories = [combined[pid]["category"] for pid in prompt_ids]
        colors = [CATEGORY_COLORS[cat] for cat in categories]

        pval = sign_test_pvalue(values)

        y_pos = np.arange(len(prompt_ids))
        ax.barh(y_pos, values, color=colors, edgecolor="none", height=0.8)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(prompt_ids, fontsize=6)
        ax.set_ylabel("Prompt ID", fontsize=9)
        ax.set_xlabel("Score difference (neg \u2212 pos)", fontsize=9)
        ax.set_title(f"{dim}  (sign test p = {pval:.3f})", fontsize=11)
        ax.tick_params(axis="x", labelsize=8)

    # Build shared legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor="none")
        for color in CATEGORY_COLORS.values()
    ]
    labels = list(CATEGORY_COLORS.keys())
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
