"""Plot average scores grouped by task category for each dataset.

Creates a 2x2 grid with one plot per dataset origin, showing mean normalized
scores by category/topic/type.

Usage:
    python -m src.analysis.topics.plot_by_category --experiment-id multi_model_discrimination_v1
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.task_data import OriginDataset
from src.task_data.loader import _load_origin
from src.measurement.storage import EXPERIMENTS_DIR

OUTPUT_DIR = Path("src/analysis/topics/plots")

SCALE_RANGES = {
    "bipolar_neg5_pos5": (-5, 5),
    "percentile_1_100": (1, 100),
    "ban_four_1_5": (1, 5),
    "compressed_anchors_1_5": (1, 5),
    "random_scale_27_32": (27, 32),
    "fruit_rating": (0, 4),
    "fruit_qualitative": (0, 4),
}


def extract_template(dir_name: str) -> str | None:
    parts = dir_name.split("_")
    for i, part in enumerate(parts):
        if part == "regex":
            for j in range(1, i):
                candidate = "_".join(parts[j:i])
                if any(candidate.startswith(m) for m in ["qwen3", "llama", "gemma", "claude", "gpt"]):
                    return "_".join(parts[:j])
    return None


def normalize_score(score: float, template: str) -> float | None:
    if template not in SCALE_RANGES:
        return None
    min_val, max_val = SCALE_RANGES[template]
    tolerance = (max_val - min_val) * 0.5
    if score < min_val - tolerance or score > max_val + tolerance:
        return None
    return (score - min_val) / (max_val - min_val)


def get_task_category(task) -> str | None:
    """Extract the primary category from task metadata."""
    meta = task.metadata

    if task.origin == OriginDataset.WILDCHAT:
        return meta.get("type") or meta.get("topic")

    elif task.origin == OriginDataset.ALPACA:
        nemo = meta.get("nemo_analysis", {})
        return nemo.get("task_type_1")

    elif task.origin == OriginDataset.MATH:
        q_meta = meta.get("q_metadata", {}).get("competition_math", {})
        return q_meta.get("type")

    elif task.origin == OriginDataset.BAILBENCH:
        return meta.get("category")

    return None


def load_task_categories() -> dict[str, str]:
    """Load category for each task ID."""
    task_categories = {}
    for origin in OriginDataset:
        try:
            tasks = _load_origin(origin)
            for task in tasks:
                cat = get_task_category(task)
                if cat:
                    task_categories[task.id] = cat
        except Exception:
            pass
    return task_categories


def load_scores_by_task(experiment_dir: Path) -> dict[str, list[float]]:
    """Load normalized scores for each task ID."""
    results_dir = experiment_dir / "post_task_stated"
    if not results_dir.exists():
        return {}

    task_scores: dict[str, list[float]] = defaultdict(list)

    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue

        measurements_path = run_dir / "measurements.yaml"
        if not measurements_path.exists():
            continue

        template = extract_template(run_dir.name)
        if not template:
            continue

        with open(measurements_path) as f:
            data = yaml.safe_load(f)

        if not data:
            continue

        for m in data:
            if "score" not in m or not isinstance(m["score"], (int, float)):
                continue

            task_id = m["task_id"]
            score = float(m["score"])
            normalized = normalize_score(score, template)

            if normalized is not None:
                task_scores[task_id].append(normalized)

    return dict(task_scores)


def aggregate_by_category(
    task_scores: dict[str, list[float]],
    task_categories: dict[str, str],
    origin_filter: str,
) -> dict[str, list[float]]:
    """Aggregate scores by category for a specific origin."""
    category_scores: dict[str, list[float]] = defaultdict(list)

    for task_id, scores in task_scores.items():
        if not task_id.startswith(origin_filter):
            continue

        category = task_categories.get(task_id)
        if category:
            category_scores[category].extend(scores)

    return dict(category_scores)


def plot_category_panel(
    ax: plt.Axes,
    category_scores: dict[str, list[float]],
    title: str,
):
    """Plot a single panel showing scores by category."""
    if not category_scores:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    # Compute stats and sort by mean
    stats = []
    for cat, scores in category_scores.items():
        arr = np.array(scores)
        stats.append({
            "category": cat,
            "mean": arr.mean(),
            "std": arr.std(),
            "n": len(arr),
        })

    stats.sort(key=lambda x: x["mean"])

    categories = [s["category"] for s in stats]
    means = [s["mean"] for s in stats]
    stds = [s["std"] for s in stats]
    ns = [s["n"] for s in stats]

    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, means, xerr=stds, capsize=3, color="steelblue", alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean Normalized Score")
    ax.set_title(title)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    # Add n labels
    for bar, n in zip(bars, ns):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"n={n}", va="center", fontsize=7,
        )


def plot_by_category_grid(
    experiment_dir: Path,
    output_path: Path,
    experiment_id: str,
):
    """Create 2x2 grid of category plots."""
    print("Loading task categories...")
    task_categories = load_task_categories()
    print(f"  Loaded categories for {len(task_categories)} tasks")

    print("Loading scores...")
    task_scores = load_scores_by_task(experiment_dir)
    print(f"  Loaded scores for {len(task_scores)} tasks")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    origins = [
        ("wildchat_", "WILDCHAT", axes[0, 0]),
        ("alpaca_", "ALPACA", axes[0, 1]),
        ("competition_math_", "MATH", axes[1, 0]),
        ("bailbench_", "BAILBENCH", axes[1, 1]),
    ]

    for prefix, name, ax in origins:
        category_scores = aggregate_by_category(task_scores, task_categories, prefix)
        plot_category_panel(ax, category_scores, name)

    fig.suptitle(f"Mean Scores by Task Category: {experiment_id}", fontsize=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot scores by task category")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    args = parser.parse_args()

    experiment_dir = EXPERIMENTS_DIR / args.experiment_id
    if not experiment_dir.exists():
        print(f"Experiment not found: {experiment_dir}")
        return

    output_dir = args.output_dir or OUTPUT_DIR
    date_str = datetime.now().strftime("%m%d%y")
    safe_exp_id = args.experiment_id.replace("/", "_")
    output_path = output_dir / f"plot_{date_str}_{safe_exp_id}_by_category.png"

    plot_by_category_grid(experiment_dir, output_path, args.experiment_id)


if __name__ == "__main__":
    main()
