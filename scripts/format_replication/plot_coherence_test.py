"""Regenerate coherence bar chart from updated coherence_by_coefficient.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "experiments" / "steering" / "stated_steering" / "coherence_test" / "results"
)
ASSETS_DIR = RESULTS_DIR.parent / "assets"
MEAN_NORM = 52820


def main():
    with open(RESULTS_DIR / "coherence_by_coefficient.json") as f:
        data = json.load(f)

    coefs = sorted(data.keys(), key=lambda x: float(x))
    pcts = [float(c) / MEAN_NORM * 100 for c in coefs]
    coherent_pcts = [data[c]["coherent_pct"] * 100 for c in coefs]
    passes = [data[c]["coherent"] for c in coefs]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#4CAF50" if p else "#F44336" for p in passes]
    bars = ax.bar([f"{p:+.0f}%" for p in pcts], coherent_pcts, color=colors, edgecolor="white")

    ax.axhline(90, color="black", linestyle="--", linewidth=0.8, label="90% threshold")
    ax.set_ylabel("Coherent responses (%)")
    ax.set_xlabel("Steering coefficient (% of mean L31 norm)")
    ax.set_ylim(0, 105)
    ax.set_title("Coherence by coefficient — last-token steering")
    ax.legend()

    for bar, pct, ok in zip(bars, coherent_pcts, passes):
        label = f"{pct:.0f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                label, ha="center", va="bottom", fontsize=8,
                color="#F44336" if not ok else "black")

    plt.tight_layout()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out = ASSETS_DIR / "plot_022526_coherence_by_coefficient.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
