"""Plot score distribution for a specific template folder.

Usage:
    python -m src.analysis.sensitivity.plot_template_distribution results/experiments/multi_model_discrimination_v1/post_task_stated/bipolar_neg5_pos5_claude-haiku-4.5_regex_cseed0_rseed0
    python -m src.analysis.sensitivity.plot_template_distribution bipolar_neg5_pos5_claude-haiku-4.5_regex_cseed0_rseed0 --experiment-id multi_model_discrimination_v1
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.measurement.storage import EXPERIMENTS_DIR

OUTPUT_DIR = Path("src/analysis/sensitivity/plots")


def load_scores(run_dir: Path) -> tuple[list[float], list[str]]:
    """Load scores and origins from measurements.yaml."""
    measurements_path = run_dir / "measurements.yaml"
    if not measurements_path.exists():
        raise FileNotFoundError(f"No measurements.yaml in {run_dir}")

    with open(measurements_path) as f:
        data = yaml.safe_load(f)

    if not data:
        return [], []

    scores = []
    origins = []
    for m in data:
        if "score" in m and isinstance(m["score"], (int, float)):
            scores.append(float(m["score"]))
            origins.append(m.get("origin", "UNKNOWN"))

    return scores, origins


def plot_distribution(
    scores: list[float],
    origins: list[str],
    output_path: Path,
    title: str,
) -> None:
    """Plot score distribution as histogram with origin breakdown."""
    if not scores:
        print("No scores to plot")
        return

    arr = np.array(scores)
    unique_scores = sorted(set(int(round(s)) for s in scores))

    # Determine if we should use discrete bars or histogram
    use_discrete = len(unique_scores) <= 15

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Overall distribution
    ax1 = axes[0]
    if use_discrete:
        counts = Counter(int(round(s)) for s in scores)
        x = sorted(counts.keys())
        y = [counts[v] for v in x]
        bars = ax1.bar(x, y, color="steelblue", alpha=0.7, edgecolor="black")
        ax1.set_xticks(x)

        # Add percentage labels
        total = len(scores)
        for bar, val in zip(bars, x):
            pct = 100 * counts[val] / total
            if pct >= 3:
                ax1.annotate(
                    f"{pct:.0f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8,
                )
    else:
        ax1.hist(arr, bins=20, color="steelblue", alpha=0.7, edgecolor="black")

    ax1.set_xlabel("Score")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Overall Distribution\nn={len(scores)}, μ={arr.mean():.2f}, σ={arr.std():.2f}")
    ax1.grid(axis="y", alpha=0.3)

    # Right: By origin
    ax2 = axes[1]
    origin_counts = Counter(origins)
    origin_colors = {
        "WILDCHAT": "#4ECDC4",
        "ALPACA": "#FF6B6B",
        "MATH": "#45B7D1",
        "BAILBENCH": "#96CEB4",
        "UNKNOWN": "#888888",
    }

    if use_discrete:
        x_positions = np.array(sorted(set(int(round(s)) for s in scores)))
        width = 0.8 / len(origin_counts)

        for i, (origin, _) in enumerate(sorted(origin_counts.items())):
            origin_scores = [int(round(s)) for s, o in zip(scores, origins) if o == origin]
            counts = Counter(origin_scores)
            y = [counts.get(v, 0) for v in x_positions]
            offset = (i - len(origin_counts) / 2 + 0.5) * width
            ax2.bar(
                x_positions + offset, y, width,
                label=f"{origin} (n={len(origin_scores)})",
                color=origin_colors.get(origin, "#888888"),
                alpha=0.7, edgecolor="black", linewidth=0.5,
            )
        ax2.set_xticks(x_positions)
    else:
        for origin, _ in sorted(origin_counts.items()):
            origin_scores = [s for s, o in zip(scores, origins) if o == origin]
            ax2.hist(
                origin_scores, bins=20, alpha=0.5,
                label=f"{origin} (n={len(origin_scores)})",
                color=origin_colors.get(origin, "#888888"),
            )

    ax2.set_xlabel("Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution by Origin")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot score distribution for a template folder")
    parser.add_argument("folder", type=str, help="Run folder path or name")
    parser.add_argument("--experiment-id", type=str, default=None, help="Experiment ID (if folder is just the name)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path for plot")
    args = parser.parse_args()

    # Resolve folder path
    folder_path = Path(args.folder)
    if not folder_path.exists():
        if args.experiment_id:
            folder_path = EXPERIMENTS_DIR / args.experiment_id / "post_task_stated" / args.folder
        else:
            # Try to find it in recent experiments
            for exp_dir in sorted(EXPERIMENTS_DIR.iterdir(), reverse=True):
                candidate = exp_dir / "post_task_stated" / args.folder
                if candidate.exists():
                    folder_path = candidate
                    break

    if not folder_path.exists():
        print(f"Folder not found: {args.folder}")
        return

    print(f"Loading from {folder_path}")
    scores, origins = load_scores(folder_path)

    if not scores:
        print("No scores found")
        return

    print(f"Loaded {len(scores)} scores")

    # Output path
    if args.output:
        output_path = args.output
    else:
        date_str = datetime.now().strftime("%m%d%y")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"plot_{date_str}_dist_{folder_path.name}.png"

    plot_distribution(scores, origins, output_path, folder_path.name)


if __name__ == "__main__":
    main()
