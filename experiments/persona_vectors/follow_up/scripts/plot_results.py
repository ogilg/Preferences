"""
Plotting script for persona vectors v2 experiment results.

Produces three plots:
1. Cosine similarity heatmap between persona vectors.
2. Topic projections box plot (mean +/- std by origin).
3. Triage screen results heatmap (mean trait score by selector x layer).
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

RESULTS_DIR = pathlib.Path("results/experiments/persona_vectors_v2")
ASSETS_DIR = pathlib.Path("experiments/persona_vectors/follow_up/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

PERSONAS = ["creative_artist", "evil", "lazy", "stem_nerd", "uncensored"]
PERSONA_DISPLAY = {
    "creative_artist": "Creative Artist",
    "evil": "Evil",
    "lazy": "Lazy",
    "stem_nerd": "STEM Nerd",
    "uncensored": "Uncensored",
}
ORIGIN_ORDER = ["alpaca", "bailbench", "math", "stress_test", "wildchat"]
ORIGIN_DISPLAY = {
    "alpaca": "Alpaca",
    "bailbench": "BailBench",
    "math": "Math",
    "stress_test": "Stress Test",
    "wildchat": "WildChat",
}


# ---------------------------------------------------------------------------
# Plot 1: Cosine similarity heatmap
# ---------------------------------------------------------------------------
def plot_cosine_heatmap():
    matrix = np.load(RESULTS_DIR / "geometry" / "persona_cosine_matrix.npy")
    with open(RESULTS_DIR / "geometry" / "persona_cosine_labels.json") as f:
        labels = json.load(f)

    display_labels = [PERSONA_DISPLAY.get(l, l) for l in labels]

    fig, ax = plt.subplots(figsize=(7, 6))
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="equal")

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_xticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(display_labels)))
    ax.set_yticklabels(display_labels, fontsize=10)
    ax.set_title("Persona Vector Cosine Similarity", fontsize=14, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=10)

    fig.tight_layout()
    out = ASSETS_DIR / "plot_022526_persona_cosine_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: Topic projections box plot
# ---------------------------------------------------------------------------
def plot_topic_projections():
    with open(RESULTS_DIR / "geometry" / "topic_projections.json") as f:
        data = json.load(f)

    personas_with_data = [p for p in PERSONAS if p in data]
    n_panels = len(personas_with_data)

    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3.2 * n_panels),
                             squeeze=False)

    colors = plt.cm.Set2(np.linspace(0, 1, len(ORIGIN_ORDER)))

    for row_idx, persona in enumerate(personas_with_data):
        ax = axes[row_idx, 0]
        pdata = data[persona]
        layer = pdata["layer"]
        stats = pdata["topic_stats"]

        origins = [o for o in ORIGIN_ORDER if o in stats]
        means = [stats[o]["mean"] for o in origins]
        stds = [stats[o]["std"] for o in origins]
        ns = [stats[o]["n"] for o in origins]

        x = np.arange(len(origins))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(origins)],
                      edgecolor="gray", linewidth=0.5, alpha=0.85, width=0.6)

        # Annotate with sample sizes
        for xi, n in zip(x, ns):
            ax.text(xi, ax.get_ylim()[0], f"n={n}", ha="center", va="bottom",
                    fontsize=7, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels([ORIGIN_DISPLAY.get(o, o) for o in origins], fontsize=9)
        ax.set_ylabel("Projection", fontsize=10)
        ax.set_title(f"{PERSONA_DISPLAY[persona]}  (layer {layer})",
                     fontsize=11, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

    fig.suptitle("Topic Projections onto Persona Vectors", fontsize=14,
                 y=1.01)
    fig.tight_layout()
    out = ASSETS_DIR / "plot_022526_topic_projections.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: Triage screen results
# ---------------------------------------------------------------------------
def plot_triage_screen():
    # Load data for all personas
    all_data = {}
    for persona in PERSONAS:
        path = RESULTS_DIR / persona / "triage" / "screen_results.json"
        if path.exists():
            with open(path) as f:
                all_data[persona] = json.load(f)

    personas_with_data = [p for p in PERSONAS if p in all_data]
    n_panels = len(personas_with_data)

    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.5 * n_panels),
                             squeeze=False)

    for row_idx, persona in enumerate(personas_with_data):
        ax = axes[row_idx, 0]
        rows = all_data[persona]

        # Filter out baseline (multiplier == 0)
        steered = [r for r in rows if r["multiplier"] != 0.0]

        # Get unique selectors and layers (sorted)
        selectors = sorted(set(r["selector"] for r in steered))
        layers = sorted(set(r["layer"] for r in steered))

        # Build mean-score matrix: rows = selectors, cols = layers
        # For each (selector, layer) pair, we average across multipliers and questions
        matrix = np.full((len(selectors), len(layers)), np.nan)
        for si, sel in enumerate(selectors):
            for li, lay in enumerate(layers):
                scores = [r["trait_score"] for r in steered
                          if r["selector"] == sel and r["layer"] == lay]
                if scores:
                    matrix[si, li] = np.mean(scores)

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                       vmin=1, vmax=5)

        # Annotate cells
        for si in range(matrix.shape[0]):
            for li in range(matrix.shape[1]):
                val = matrix[si, li]
                if not np.isnan(val):
                    color = "white" if val > 3.5 else "black"
                    ax.text(li, si, f"{val:.1f}", ha="center", va="center",
                            fontsize=10, fontweight="bold", color=color)

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([str(l) for l in layers], fontsize=9)
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_yticks(range(len(selectors)))
        sel_display = {"prompt_last": "prompt_last", "prompt_mean": "prompt_mean"}
        ax.set_yticklabels([sel_display.get(s, s) for s in selectors], fontsize=9)
        ax.set_title(f"{PERSONA_DISPLAY[persona]} -- Mean Trait Score (steered only)",
                     fontsize=11, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Trait Score", fontsize=9)

    fig.suptitle("Triage Screen: Trait Scores by Selector and Layer",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    out = ASSETS_DIR / "plot_022526_triage_screen.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating persona vectors v2 plots...")
    plot_cosine_heatmap()
    plot_topic_projections()
    plot_triage_screen()
    print("All plots saved.")
