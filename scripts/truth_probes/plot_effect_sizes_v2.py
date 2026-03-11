import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "experiments" / "truth_probes" / "truth_probes_results.json"
OUT_PATH = ROOT / "experiments" / "truth_probes" / "assets" / "plot_031126_truth_effect_size_by_layer.png"

LAYERS = [25, 32, 39, 46, 53]
FRAMINGS = ["raw", "repeat"]
PROBES = ["tb-2", "tb-5"]

PROBE_LABELS = {
    "tb-2": "tb-2 (`model` token)",
    "tb-5": "tb-5 (`<end_of_turn>` token)",
}
PANEL_TITLES = {
    "raw": "Raw: claim as user message",
    "repeat": "Repeat: 'Please say the statement'",
}


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150, sharey=True)
    # Compute global y-max across all data to set shared limits
    all_ds = [
        results[f][p][str(l)]["cohens_d"]
        for f in FRAMINGS for p in PROBES for l in LAYERS
    ]
    y_max = max(all_ds) * 1.1

    for i, framing in enumerate(FRAMINGS):
        ax = axes[i]
        for probe in PROBES:
            ds = [results[framing][probe][str(layer)]["cohens_d"] for layer in LAYERS]
            ax.plot(LAYERS, ds, marker="o", label=PROBE_LABELS[probe])

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8)

        if i == 0:
            ax.annotate(
                "large effect (d=0.8)",
                xy=(LAYERS[-1], 0.8),
                xytext=(5, 2),
                textcoords="offset points",
                fontsize=7,
                color="gray",
                ha="left",
                va="bottom",
            )
            ax.set_ylabel("Cohen's d")

        ax.set_xlabel("Layer")
        ax.set_title(PANEL_TITLES[framing])
        ax.set_xticks(LAYERS)
        ax.set_ylim(bottom=0, top=y_max)
        ax.legend(fontsize=8)

    fig.suptitle("Where in the network does the truth signal peak?", fontsize=13)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
