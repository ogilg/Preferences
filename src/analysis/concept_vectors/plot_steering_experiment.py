"""Plot concept vector steering experiment results as violin plots."""

import argparse
import re
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np


def discover_conditions(results_dir: Path) -> tuple[list[str], list[int], list[float]]:
    """Discover sources, layers, and coefficients from directory names."""
    pattern = re.compile(r"completion_(\w+)_layer(\d+)_coef(-?\d+\.?\d*)")
    
    sources = set()
    layers = set()
    coefficients = set()
    
    for d in results_dir.iterdir():
        if d.is_dir():
            match = pattern.match(d.name)
            if match:
                sources.add(match.group(1))
                layers.add(int(match.group(2)))
                coefficients.add(float(match.group(3)))
    
    # Sort for consistent ordering
    sources_order = ["positive", "neutral", "negative"]
    sources_list = [s for s in sources_order if s in sources] + sorted(sources - set(sources_order))
    
    return sources_list, sorted(layers), sorted(coefficients)


def load_scores(results_dir: Path, source: str, layer: int, coef: float) -> list[float]:
    """Load scores for a specific condition."""
    condition = f"completion_{source}_layer{layer}_coef{coef}"
    path = results_dir / condition / "measurements.yaml"
    if not path.exists():
        return []
    with open(path) as f:
        data = yaml.safe_load(f)
    return [item["score"] for item in data if item["score"] is not None and 1 <= item["score"] <= 5]


def main(experiment_id: str):
    results_dir = Path(f"results/experiments/{experiment_id}/post_task_stated")
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    sources, layers, coefficients = discover_conditions(results_dir)
    print(f"Discovered: sources={sources}, layers={layers}, coefficients={coefficients}")
    
    fig, axes = plt.subplots(len(sources), len(layers), figsize=(4 * len(layers), 3.5 * len(sources)), 
                              sharey=True, sharex=True, squeeze=False)

    # Generate colors for coefficients (blue for negative, red for positive)
    coef_colors = {}
    for coef in coefficients:
        if coef < 0:
            coef_colors[coef] = "#2196F3"  # blue
        else:
            coef_colors[coef] = "#F44336"  # red

    for row_idx, source in enumerate(sources):
        for col_idx, layer in enumerate(layers):
            ax = axes[row_idx, col_idx]

            data_to_plot = []
            positions = []
            colors_to_use = []

            for i, coef in enumerate(coefficients):
                scores = load_scores(results_dir, source, layer, coef)
                if scores:
                    data_to_plot.append(scores)
                    positions.append(i + 1)
                    colors_to_use.append(coef_colors[coef])

            if data_to_plot:
                parts = ax.violinplot(
                    data_to_plot,
                    positions=positions,
                    showmeans=True,
                    showmedians=False,
                    widths=0.7,
                )

                for pc, color in zip(parts["bodies"], colors_to_use):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                for partname in ["cbars", "cmins", "cmaxes", "cmeans"]:
                    if partname in parts:
                        parts[partname].set_color("black")

            # Add n counts and means
            for i, coef in enumerate(coefficients):
                scores = load_scores(results_dir, source, layer, coef)
                if scores:
                    n = len(scores)
                    mean = np.mean(scores)
                    ax.text(i + 1, 5.3, f"n={n}", ha="center", fontsize=8, color="gray")
                    ax.text(i + 1, 0.6, f"μ={mean:.2f}", ha="center", fontsize=8, color="gray")

            # Neutral line
            ax.axhline(y=3, color="gray", linestyle="--", alpha=0.5, linewidth=1)

            ax.set_xticks(range(1, len(coefficients) + 1))
            ax.set_xticklabels([str(c) for c in coefficients])
            ax.set_ylim(0.3, 5.7)
            ax.set_yticks([1, 2, 3, 4, 5])

            # Column titles (layer) - only on top row
            if row_idx == 0:
                ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")

            # Row titles (source) - only on left column
            if col_idx == 0:
                ax.set_ylabel(f"Source: {source}\n\nScore (1-5)", fontsize=10)

            # X-axis label - only on bottom row
            if row_idx == len(sources) - 1:
                ax.set_xlabel("Steering coefficient", fontsize=10)

    fig.suptitle(
        f"Effect of Concept Vector Steering on Stated Preferences\n({experiment_id})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=coef_colors[c], alpha=0.7, label=f"coef = {c}")
        for c in coefficients
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.92))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = f"src/analysis/concept_vectors/plot_{experiment_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot concept vector steering results")
    parser.add_argument("experiment_id", help="Experiment ID (e.g. concept_vector_steering_math_001)")
    args = parser.parse_args()
    main(args.experiment_id)
