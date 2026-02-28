"""Plot coherence-filtered preference steering results with clear labelling."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("results/experiments/persona_vectors_v2/preference_steering/coherent/preference_results.json")

# What the pairs actually are, and what "positive rate" means
PAIR_DESCRIPTIONS = {
    "creative_artist": {
        "label": "Creative Artist",
        "pair": "creative (WildChat) vs math (MATH)",
        "y_label": "P(choose creative task)",
        "expected": "Steering should increase preference for creative tasks",
    },
    "evil": {
        "label": "Evil",
        "pair": "harmful (BailBench/StressTest) vs benign (Alpaca)",
        "y_label": "P(choose harmful task)",
        "expected": "Steering should increase preference for harmful tasks",
    },
    "lazy": {
        "label": "Lazy",
        "pair": "hard (MATH) vs easy (Alpaca)",
        "y_label": "P(choose hard task)",
        "expected": "Steering should decrease preference for hard tasks",
    },
    "stem_nerd": {
        "label": "STEM Nerd",
        "pair": "math (MATH) vs creative (WildChat)",
        "y_label": "P(choose math task)",
        "expected": "Steering should increase preference for math tasks",
    },
    "uncensored": {
        "label": "Uncensored",
        "pair": "harmful (BailBench/StressTest) vs benign (Alpaca)",
        "y_label": "P(choose harmful task)",
        "expected": "Steering should increase preference for harmful tasks",
    },
}

PERSONAS = ["creative_artist", "evil", "lazy", "stem_nerd", "uncensored"]


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
    fig.suptitle("Preference Steering: Does persona vector shift task choice?", fontsize=14, y=1.02)

    bar_width = 0.35
    colors_base = "#5B9BD5"
    colors_steer = "#E07B54"

    for i, persona in enumerate(PERSONAS):
        ax = axes[i]
        r = results[persona]
        desc = PAIR_DESCRIPTIONS[persona]
        coh = r["coherent_subset"]

        base_rate = coh["baseline_rate"]
        steer_rate = coh["steered_rate"]
        n_base = coh["n_baseline"]
        n_steer = coh["n_steered"]
        steer_coh_rate = coh["steered_coherence_rate"]
        delta = coh.get("delta", 0)

        x = np.array([0, 1])
        bars = ax.bar(x, [base_rate, steer_rate], width=0.6,
                      color=[colors_base, colors_steer], edgecolor="white", linewidth=0.5)

        # Annotate bars with n
        ax.text(0, base_rate + 0.02, f"n={n_base}", ha="center", va="bottom", fontsize=8)
        steer_label = f"n={n_steer}"
        if steer_coh_rate < 0.9:
            steer_label += f"\n({steer_coh_rate:.0%} coh)"
        ax.text(1, steer_rate + 0.02, steer_label, ha="center", va="bottom", fontsize=8, color="red" if steer_coh_rate < 0.9 else "black")

        # Delta annotation
        delta_color = "green" if delta > 0 else "red" if delta < -0.05 else "gray"
        ax.text(0.5, max(base_rate, steer_rate) + 0.10, f"\u0394 = {delta:+.1%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=delta_color)

        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", f"Steered\n({r['multiplier']:.2f}\u00d7)"], fontsize=9)
        ax.set_title(f"{desc['label']}\n{desc['pair']}", fontsize=9, fontweight="bold")
        ax.set_ylabel(desc["y_label"], fontsize=9)

        ax.set_ylim(0, 1.15)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    out_path = Path("experiments/persona_vectors/follow_up/assets/plot_022626_preference_steering_labelled.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
