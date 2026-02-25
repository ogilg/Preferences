import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PERSONAS = ["evil", "stem_nerd", "creative_artist", "uncensored", "lazy"]
PRETTY_LABELS = {
    "evil": "Evil",
    "stem_nerd": "STEM Nerd",
    "creative_artist": "Creative Artist",
    "uncensored": "Uncensored",
    "lazy": "Lazy",
}

RESULTS_ROOT = pathlib.Path("results/experiments/persona_vectors")
OUTPUT_PATH = pathlib.Path("experiments/persona_vectors/assets/plot_022526_trait_dose_response.png")


def load_results(persona: str) -> list[dict]:
    path = RESULTS_ROOT / persona / "steering" / "trait_steering_results.json"
    with open(path) as f:
        data = json.load(f)
    # Filter out judge errors (trait_score == -1)
    return [r for r in data if r["trait_score"] != -1]


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    na, nb = len(group_a), len(group_b)
    mean_diff = group_b.mean() - group_a.mean()
    pooled_std = np.sqrt(
        ((na - 1) * group_a.std(ddof=1) ** 2 + (nb - 1) * group_b.std(ddof=1) ** 2)
        / (na + nb - 2)
    )
    return mean_diff / pooled_std


def main():
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)
    fig.suptitle(
        "Persona Vector Steering: Trait Expression Dose-Response",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    for ax, persona in zip(axes, PERSONAS):
        results = load_results(persona)
        multipliers = np.array([r["multiplier"] for r in results])
        scores = np.array([r["trait_score"] for r in results])

        unique_mults = np.sort(np.unique(multipliers))

        # Compute means and 95% CIs per multiplier
        means = []
        ci_lows = []
        ci_highs = []
        for m in unique_mults:
            mask = multipliers == m
            group = scores[mask]
            mean = group.mean()
            sem = group.std(ddof=1) / np.sqrt(len(group))
            means.append(mean)
            ci_lows.append(mean - 1.96 * sem)
            ci_highs.append(mean + 1.96 * sem)

        means = np.array(means)
        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)

        # Individual points with jitter
        jitter = np.random.default_rng(42).normal(
            0, (unique_mults[1] - unique_mults[0]) * 0.06, size=len(multipliers)
        )
        ax.scatter(
            multipliers + jitter, scores, alpha=0.2, s=12, color="steelblue", zorder=1
        )

        # Mean + 95% CI error bars
        yerr = np.array([means - ci_lows, ci_highs - means])
        ax.errorbar(
            unique_mults,
            means,
            yerr=yerr,
            fmt="o-",
            color="darkblue",
            capsize=4,
            linewidth=1.5,
            markersize=5,
            zorder=2,
        )

        # Cohen's d: most negative vs most positive multiplier
        neg_scores = scores[multipliers == unique_mults[0]]
        pos_scores = scores[multipliers == unique_mults[-1]]
        d = cohens_d(neg_scores, pos_scores)

        ax.set_title(f"{PRETTY_LABELS[persona]}\nCohen's d = {d:.2f}", fontsize=11)
        ax.set_xlabel("Multiplier")
        ax.set_ylim(0.5, 5.5)
        ax.set_xticks(unique_mults)
        ax.set_xticklabels([f"{m:.2f}" for m in unique_mults], fontsize=7)

    axes[0].set_ylabel("Trait Score")

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
