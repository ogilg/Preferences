"""Horizontal bar chart visualization for narrow persona results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_narrow_personas():
    """Create a horizontal bar chart of narrow persona on-target deltas."""

    # Load data
    analysis_path = Path(
        "/Users/oscargilg/Dev/MATS/Preferences/experiments/probe_generalization/persona_ood/v2_analysis.json"
    )
    with open(analysis_path) as f:
        data = json.load(f)

    narrow = data["narrow"]

    # Sort by on_target_delta descending
    narrow_sorted = sorted(narrow, key=lambda x: x["on_target_delta"], reverse=True)

    # Extract data
    names = [p["name"] for p in narrow_sorted]
    deltas = [p["on_target_delta"] for p in narrow_sorted]
    passes = [p["passes"] for p in narrow_sorted]
    specificity = [p["specificity_ratio"] for p in narrow_sorted]
    ranks = [p["rank"] for p in narrow_sorted]
    n_tasks = [p["n_tasks"] for p in narrow_sorted]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Colors based on pass/fail
    colors = ["#2ecc71" if p else "#e74c3c" for p in passes]

    # Create bar chart
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, deltas, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Add vertical line at delta=0.1 threshold
    ax.axvline(x=0.1, color="black", linestyle="--", linewidth=2, alpha=0.6, label="Δ = 0.1 threshold")

    # Add text labels on bars
    for i, (bar, spec, rank, n) in enumerate(zip(bars, specificity, ranks, n_tasks)):
        width = bar.get_width()
        label_x = width + 0.02

        # Format the label with specificity ratio and rank
        label_text = f"spec={spec:.2f}, rank={rank}/{n}"

        ax.text(
            label_x,
            bar.get_y() + bar.get_height()/2,
            label_text,
            va="center",
            ha="left",
            fontsize=8
        )

    # Customize axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("On-target Δ(p_choose)", fontsize=12, fontweight="bold")
    ax.set_title("Narrow personas: On-target Δ(p_choose) and specificity", fontsize=14, fontweight="bold")
    ax.set_xlim(left=0)

    # Add legend
    ax.legend(loc="lower right", fontsize=10)

    # Invert y-axis so highest delta is at top
    ax.invert_yaxis()

    # Add grid for readability
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    plt.tight_layout()

    # Save figure
    output_path = Path(
        "/Users/oscargilg/Dev/MATS/Preferences/experiments/probe_generalization/persona_ood/assets"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    fig_path = output_path / "plot_021726_narrow_persona_bars.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {fig_path}")

    plt.close()


if __name__ == "__main__":
    plot_narrow_personas()
