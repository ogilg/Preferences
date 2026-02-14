"""Plot 2: Embedding cosine distance vs |coefficient| by category.

Shows how much steered generations diverge from unsteered (coef=0) generations
in embedding space, as a function of steering magnitude.

Categories C, D, E, F only (A has no meaningful text variation, B ratings are short).
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer

DATA_PATH = Path("experiments/steering/program/coefficient_calibration/generation_results.json")
OUT_PATH = Path("experiments/steering/program/open_ended_effects/mu_conditional_reanalysis/assets/plot_021426_embedding_distance_by_category.png")

COEF_RANGE = [-5000, -3000, -2000, -1000, -500, 0, 500, 1000, 2000, 3000, 5000]
PLOT_CATEGORIES = ["C_completion", "D_valence", "E_neutral", "F_affect"]
CATEGORY_COLORS = {
    "C_completion": "#4dac26",
    "D_valence": "#2166ac",
    "E_neutral": "#888888",
    "F_affect": "#b2182b",
}
CATEGORY_LABELS = {
    "C_completion": "C: Task Completion",
    "D_valence": "D: Valence Probes",
    "E_neutral": "E: Neutral / Factual",
    "F_affect": "F: Affect Elicitation",
}


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    results = [r for r in data["results"] if r["coefficient"] in COEF_RANGE]
    prompts = data["prompts"]

    # Collect all responses we need to embed
    print("Loading sentence transformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Organize responses: (category, prompt_id, coefficient, seed) -> response text
    response_map: dict[tuple[str, str, int, int], str] = {}
    for r in results:
        if r["category"] in PLOT_CATEGORIES:
            response_map[(r["category"], r["prompt_id"], r["coefficient"], r["seed"])] = r["response"]

    # Get unique texts and embed them
    unique_texts = list(set(response_map.values()))
    print(f"Embedding {len(unique_texts)} unique responses...")
    embeddings_array = model.encode(unique_texts, show_progress_bar=True, batch_size=64)
    text_to_embedding = {t: embeddings_array[i] for i, t in enumerate(unique_texts)}

    # For each category, prompt, compute mean cosine distance from coef=0 responses
    abs_coefs = sorted(set(abs(c) for c in COEF_RANGE if c != 0))

    fig, ax = plt.subplots(figsize=(10, 6))

    for cat in PLOT_CATEGORIES:
        cat_prompts = [p["prompt_id"] for p in prompts if p["category"] == cat]
        seeds = sorted(set(r["seed"] for r in results))

        # Per |coef|, collect distances across all prompts and seeds
        distances_by_abs_coef: dict[int, list[float]] = defaultdict(list)

        for pid in cat_prompts:
            # Get baseline (coef=0) embeddings
            baseline_embs = []
            for s in seeds:
                key = (cat, pid, 0, s)
                if key in response_map:
                    baseline_embs.append(text_to_embedding[response_map[key]])
            if not baseline_embs:
                continue
            baseline_mean = np.mean(baseline_embs, axis=0)

            for coef in COEF_RANGE:
                if coef == 0:
                    continue
                for s in seeds:
                    key = (cat, pid, coef, s)
                    if key in response_map:
                        emb = text_to_embedding[response_map[key]]
                        dist = cosine_distance(emb, baseline_mean)
                        distances_by_abs_coef[abs(coef)].append(dist)

        means = []
        sems = []
        for ac in abs_coefs:
            dists = distances_by_abs_coef[ac]
            means.append(np.mean(dists) if dists else 0)
            sems.append(stats.sem(dists) if len(dists) > 1 else 0)

        ax.errorbar(
            abs_coefs, means, yerr=sems,
            color=CATEGORY_COLORS[cat], label=CATEGORY_LABELS[cat],
            linewidth=2, marker="o", markersize=5, capsize=3,
        )

    ax.set_xlabel("|Coefficient|", fontsize=11)
    ax.set_ylabel("Mean cosine distance from unsteered", fontsize=11)
    ax.set_title("Embedding Distance from Unsteered Generations", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
