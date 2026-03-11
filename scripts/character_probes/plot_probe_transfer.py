"""Plot probe transfer results for character persona experiment.

Produces:
1. Selector comparison: mean transfer r across personas (best layer per selector)
2. Layer comparison within TB-2: mean transfer r across personas
3. Final bar chart: baseline r vs probe r at fixed (TB-2, best layer)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("results/probes/character_probes/cross_eval/cross_eval_results.json")
ASSETS_DIR = Path("experiments/character_probes/probes/assets")

PERSONAS = [
    "goodness", "humor", "impulsiveness", "loving", "mathematical",
    "nonchalance", "poeticism", "remorse", "sarcasm", "sycophancy",
]

PERSONA_DISPLAY = {
    "goodness": "Goodness",
    "humor": "Humor",
    "impulsiveness": "Impulsive",
    "loving": "Loving",
    "mathematical": "Mathematical",
    "nonchalance": "Nonchalant",
    "poeticism": "Poetic",
    "remorse": "Remorseful",
    "sarcasm": "Sarcastic",
    "sycophancy": "Sycophantic",
}

SELECTOR_DISPLAY = {
    "task_mean": "Task mean",
    "turn_boundary:-1": "<start_of_turn> -1",
    "turn_boundary:-2": "<start_of_turn> -2",
    "turn_boundary:-3": "<start_of_turn> -3",
    "turn_boundary:-4": "<start_of_turn> -4",
    "turn_boundary:-5": "<start_of_turn> -5",
}

SELECTORS = list(SELECTOR_DISPLAY.keys())
LAYERS = [8, 12, 16, 20, 24]


def load_results() -> dict:
    with open(RESULTS_PATH) as f:
        return json.load(f)


def get_transfer_r(results: dict, selector: str, layer: int, persona: str) -> float:
    return results["probe_transfer"][selector][str(layer)][persona]["r"]


def plot_selector_comparison(results: dict):
    """(a) For each selector, compute mean transfer r across personas (at best layer per selector)."""
    selector_means = {}
    selector_per_persona = {}
    for selector in SELECTORS:
        # For each persona, pick the best layer for this selector
        best_per_persona = []
        per_persona = {}
        for persona in PERSONAS:
            best_r = max(get_transfer_r(results, selector, layer, persona) for layer in LAYERS)
            best_per_persona.append(best_r)
            per_persona[persona] = best_r
        selector_means[selector] = np.mean(best_per_persona)
        selector_per_persona[selector] = per_persona

    # Sort selectors by mean r
    sorted_sels = sorted(SELECTORS, key=lambda s: selector_means[s], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sorted_sels))

    means = [selector_means[s] for s in sorted_sels]
    colors = ["#1976d2" if s == "turn_boundary:-2" else "#90caf9" for s in sorted_sels]
    ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.5)

    # Overlay individual persona dots
    for i, sel in enumerate(sorted_sels):
        vals = [selector_per_persona[sel][p] for p in PERSONAS]
        jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(vals))
        ax.scatter(x[i] + jitter, vals, color="black", s=12, alpha=0.4, zorder=3)

    ax.set_ylabel("Pearson r (best layer per selector)", fontsize=11)
    ax.set_title("Which activation selector transfers best?", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SELECTOR_DISPLAY[s] for s in sorted_sels], fontsize=10, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, m in enumerate(means):
        ax.text(i, m + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    path = ASSETS_DIR / "plot_031126_selector_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    return "turn_boundary:-2"


def plot_layer_comparison(results: dict, selector: str):
    """(b) For the winning selector, compare layers."""
    layer_means = {}
    layer_per_persona = {}
    for layer in LAYERS:
        vals = [get_transfer_r(results, selector, layer, p) for p in PERSONAS]
        layer_means[layer] = np.mean(vals)
        layer_per_persona[layer] = vals

    sorted_layers = sorted(LAYERS, key=lambda l: layer_means[l], reverse=True)
    best_layer = sorted_layers[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(LAYERS))

    means = [layer_means[l] for l in LAYERS]
    colors = ["#1976d2" if l == best_layer else "#90caf9" for l in LAYERS]
    ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.5)

    # Overlay individual persona dots
    for i, layer in enumerate(LAYERS):
        vals = layer_per_persona[layer]
        jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(vals))
        ax.scatter(x[i] + jitter, vals, color="black", s=12, alpha=0.4, zorder=3)

    ax.set_ylabel("Pearson r", fontsize=11)
    sel_name = SELECTOR_DISPLAY[selector]
    ax.set_title(f"Which layer transfers best? ({sel_name} selector)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in LAYERS], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, m in enumerate(means):
        ax.text(i, m + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    path = ASSETS_DIR / "plot_031126_layer_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    return best_layer


def plot_final_transfer_bar(results: dict, selector: str, layer: int):
    """(c) Bar chart: baseline r vs fixed layer vs best layer, all at fixed selector."""
    fixed_vals = {p: get_transfer_r(results, selector, layer, p) for p in PERSONAS}

    best_layer_vals = {}
    best_layer_choice = {}
    for p in PERSONAS:
        best_r = -1.0
        best_l = 0
        for l in LAYERS:
            r = get_transfer_r(results, selector, l, p)
            if r > best_r:
                best_r = r
                best_l = l
        best_layer_vals[p] = best_r
        best_layer_choice[p] = best_l

    # Sort by best probe r
    sorted_personas = sorted(PERSONAS, key=lambda p: best_layer_vals[p], reverse=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(sorted_personas))
    width = 0.25

    baseline = [results["baseline_r"][p] for p in sorted_personas]
    fixed = [fixed_vals[p] for p in sorted_personas]
    best = [best_layer_vals[p] for p in sorted_personas]

    sel_name = SELECTOR_DISPLAY[selector]
    ax.bar(x - width, baseline, width, label="Utility correlation (default vs character)",
           color="#9e9e9e", edgecolor="white", linewidth=0.5)
    ax.bar(x, fixed, width, label=f"Probe (<start_of_turn>, layer {layer})",
           color="#1976d2", edgecolor="white", linewidth=0.5)
    ax.bar(x + width, best, width, label="Probe (<start_of_turn>, best layer)",
           color="#0d47a1", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_title("Base model probes predict character persona preferences",
                 fontsize=12, fontweight="bold", pad=18)
    ax.text(0.5, 1.01,
            "Ridge probe trained on default persona activations & utilities, applied to predicting character utilities from character activations",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels([PERSONA_DISPLAY[p] for p in sorted_personas], rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate best-layer bar with layer choice
    for i, p in enumerate(sorted_personas):
        ax.text(x[i] + width, best[i] + 0.012, f"L{best_layer_choice[p]}",
                ha="center", va="bottom", fontsize=8, color="#555")

    fig.tight_layout()
    path = ASSETS_DIR / "plot_031126_probe_transfer_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()

    print("=== (a) Selector comparison ===")
    best_selector = plot_selector_comparison(results)
    print(f"Best selector: {best_selector}")

    print(f"\n=== (b) Layer comparison ({SELECTOR_DISPLAY[best_selector]}) ===")
    best_layer = plot_layer_comparison(results, best_selector)
    print(f"Best layer: {best_layer}")

    print(f"\n=== (c) Final bar chart ({SELECTOR_DISPLAY[best_selector]}, L{best_layer}) ===")
    plot_final_transfer_bar(results, best_selector, best_layer)

    # Print summary table
    print(f"\nPer-persona results at {SELECTOR_DISPLAY[best_selector]}, layer {best_layer}:")
    print(f"{'Persona':<14} {'Baseline':>10} {'Probe':>10} {'Gain':>10}")
    print("-" * 46)
    for p in PERSONAS:
        bl = results["baseline_r"][p]
        pr = get_transfer_r(results, best_selector, best_layer, p)
        print(f"{PERSONA_DISPLAY[p]:<14} {bl:>10.3f} {pr:>10.3f} {pr - bl:>+10.3f}")


if __name__ == "__main__":
    main()
