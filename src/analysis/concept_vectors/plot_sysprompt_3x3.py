"""Plot 3x3 system prompt variation results as violin plots."""

import argparse
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np


TASK_SOURCES = {
    "math": {
        "results_dir": Path("results/experiments/sysprompt_3x3_anchored/post_task_stated"),
        "title": "MATH Tasks",
        "output": "plot_012826_sysprompt_3x3_math_anchored_violins.png",
    },
    "wildchat": {
        "results_dir": Path("results/experiments/sysprompt_3x3_wildchat_anchored/post_task_stated"),
        "title": "WildChat Tasks",
        "output": "plot_012826_sysprompt_3x3_wildchat_anchored_violins.png",
    },
}


def load_scores(results_dir: Path, condition_name: str) -> list[float]:
    path = results_dir / condition_name / "measurements.yaml"
    if not path.exists():
        return []
    with open(path) as f:
        data = yaml.safe_load(f)
    return [item["score"] for item in data if 1 <= item["score"] <= 5]


def main(task_source: str):
    config = TASK_SOURCES[task_source]
    results_dir = config["results_dir"]

    completion_sources = ["positive", "neutral", "negative"]
    measurement_contexts = ["positive", "neutral", "negative"]

    # Labels for completion sources (what sysprompt was used during task completion)
    source_labels = {
        "positive": '"You love math"',
        "neutral": "None",
        "negative": '"You hate math"',
    }

    # Labels for measurement contexts (what sysprompt is used when asking for rating)
    context_labels = {
        "positive": '"You love math"',
        "neutral": "None",
        "negative": '"You hate math"',
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    colors = {"positive": "#4CAF50", "neutral": "#9E9E9E", "negative": "#F44336"}

    for col_idx, context in enumerate(measurement_contexts):
        ax = axes[col_idx]

        data_to_plot = []
        labels = []
        available = []

        for source in completion_sources:
            condition = f"completion_{source}_context_{context}"
            scores = load_scores(results_dir, condition)
            data_to_plot.append(scores if scores else [np.nan])
            labels.append(source_labels[source])
            available.append(len(scores) > 0)

        positions = [1, 2, 3]
        parts = ax.violinplot(
            [d for d, a in zip(data_to_plot, available) if a],
            positions=[p for p, a in zip(positions, available) if a],
            showmeans=True,
            showmedians=False,
        )

        for pc in parts["bodies"]:
            pc.set_facecolor(colors[context])
            pc.set_alpha(0.7)
        for partname in ["cbars", "cmins", "cmaxes", "cmeans"]:
            if partname in parts:
                parts[partname].set_color("black")

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("Sysprompt for original completion", fontsize=10)
        ax.set_title(f"Sysprompt for measurement:\n{context_labels[context]}", fontsize=11)
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.axhline(y=3, color="gray", linestyle="--", alpha=0.3)

        # Add N counts
        for pos, scores, avail in zip(positions, data_to_plot, available):
            if avail:
                ax.text(pos, 5.3, f"n={len(scores)}", ha="center", fontsize=8, color="gray")

    axes[0].set_ylabel("Self-reported enjoyment score (1-5)", fontsize=10)

    fig.suptitle(f"Effect of System Prompt on Stated Preferences ({config['title']})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path = f"src/analysis/concept_vectors/{config['output']}"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 3x3 sysprompt variation results")
    parser.add_argument(
        "--task-source",
        choices=list(TASK_SOURCES.keys()),
        required=True,
        help="Which task source to plot (math or wildchat)",
    )
    args = parser.parse_args()
    main(args.task_source)
