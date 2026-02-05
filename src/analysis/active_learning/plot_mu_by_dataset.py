"""Plot average mu and std for each dataset from active learning results.

Usage:
    python -m src.analysis.active_learning.plot_mu_by_dataset --experiment-id gemma3_al_500
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.measurement.storage import EXPERIMENTS_DIR

OUTPUT_DIR = Path(__file__).parent / "plots"


def find_thurstonian_csv(experiment_dir: Path, run_name: str | None = None) -> Path | None:
    """Find the thurstonian CSV file in the experiment directory."""
    al_dir = experiment_dir / "post_task_active_learning"
    if not al_dir.exists():
        return None

    for run_dir in al_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if run_name and not run_dir.name.startswith(run_name):
            continue
        for f in run_dir.iterdir():
            if f.name.startswith("thurstonian_") and f.suffix == ".csv":
                return f
    return None


def load_mu_by_dataset(csv_path: Path) -> dict[str, list[float]]:
    """Load mu values grouped by dataset origin."""
    dataset_mus: dict[str, list[float]] = defaultdict(list)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row["task_id"]
            mu = float(row["mu"])

            if task_id.startswith("wildchat_"):
                ds = "wildchat"
            elif task_id.startswith("alpaca_"):
                ds = "alpaca"
            elif task_id.startswith("competition_math_"):
                ds = "math"
            elif task_id.startswith("bailbench_"):
                ds = "bailbench"
            elif task_id.startswith("stresstest_"):
                ds = "stress_test"
            else:
                ds = "other"

            dataset_mus[ds].append(mu)

    return dict(dataset_mus)


def plot_mu_by_dataset(
    dataset_mus: dict[str, list[float]],
    output_path: Path,
    experiment_id: str,
) -> None:
    """Create bar plot of mean mu by dataset with error bars."""
    # Compute stats and sort by mean
    stats = []
    for ds, mus in dataset_mus.items():
        arr = np.array(mus)
        stats.append({
            "dataset": ds,
            "mean": arr.mean(),
            "std": arr.std(),
            "sem": arr.std() / np.sqrt(len(arr)),
            "n": len(arr),
        })

    stats.sort(key=lambda x: x["mean"], reverse=True)

    datasets = [s["dataset"] for s in stats]
    means = [s["mean"] for s in stats]
    sems = [s["sem"] for s in stats]
    ns = [s["n"] for s in stats]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["coral" if m > 0 else "steelblue" for m in means]
    bars = ax.bar(datasets, means, yerr=sems, capsize=5, color=colors, edgecolor="black", alpha=0.8)

    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_ylabel("Mean Utility (μ)")
    ax.set_xlabel("Dataset")
    ax.set_title(f"Mean Preference Utility by Dataset\n{experiment_id}")

    # Add n labels on bars
    for bar, n, mean in zip(bars, ns, means):
        y_offset = 0.1 if mean >= 0 else -0.15
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + y_offset,
            f"n={n}",
            ha="center",
            va="bottom" if mean >= 0 else "top",
            fontsize=9,
        )

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {output_path}")


def print_stats(dataset_mus: dict[str, list[float]]) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("MEAN UTILITY BY DATASET")
    print("=" * 60)

    stats = []
    for ds, mus in dataset_mus.items():
        arr = np.array(mus)
        stats.append({
            "dataset": ds,
            "n": len(arr),
            "mean": arr.mean(),
            "std": arr.std(),
            "min": arr.min(),
            "max": arr.max(),
        })

    stats.sort(key=lambda x: x["mean"], reverse=True)

    print(f"{'Dataset':<12} {'n':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 60)
    for s in stats:
        print(f"{s['dataset']:<12} {s['n']:>6} {s['mean']:>+8.3f} {s['std']:>8.3f} {s['min']:>8.3f} {s['max']:>8.3f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plot mean mu by dataset")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None, help="Filter to run starting with this name (e.g., 'enjoy_most')")
    args = parser.parse_args()

    experiment_dir = EXPERIMENTS_DIR / args.experiment_id
    if not experiment_dir.exists():
        print(f"Experiment not found: {experiment_dir}")
        return

    csv_path = find_thurstonian_csv(experiment_dir, args.run_name)
    if csv_path is None:
        print(f"No thurstonian CSV found in {experiment_dir}")
        return

    print(f"Loading from {csv_path}")
    dataset_mus = load_mu_by_dataset(csv_path)

    print_stats(dataset_mus)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%m%d%y")
    suffix = f"_{args.run_name}" if args.run_name else ""
    output_path = OUTPUT_DIR / f"plot_{date_str}_mu_by_dataset_{args.experiment_id}{suffix}.png"

    plot_mu_by_dataset(dataset_mus, output_path, f"{args.experiment_id} ({args.run_name})" if args.run_name else args.experiment_id)


if __name__ == "__main__":
    main()
