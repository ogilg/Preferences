"""Usage: python -m src.experiments.sensitivity_experiments.plot results/measurements/"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.experiments.correlation import compute_pairwise_correlations
from src.experiments.sensitivity_experiments.sensitivity import compute_sensitivities
from src.preferences.ranking import PairwiseData, fit_thurstonian
from src.preferences.storage import MEASUREMENTS_DIR, load_yaml
from src.task_data import OriginDataset, Task


@dataclass
class RunConfig:
    template_name: str
    template_tags: dict
    model_short: str
    run_dir: Path


def list_runs(results_dir: Path) -> list[RunConfig]:
    runs = []
    if not results_dir.exists():
        return runs

    for run_dir in sorted(results_dir.iterdir()):
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            config = load_yaml(config_path)
            runs.append(RunConfig(
                template_name=config["template_name"],
                template_tags=config["template_tags"],
                model_short=config["model_short"],
                run_dir=run_dir,
            ))
    return runs


def load_run_utilities(run_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load measurements and fit thurstonian model, returning (mu, task_ids)."""
    measurements_path = run_dir / "measurements.yaml"
    if not measurements_path.exists():
        raise FileNotFoundError(f"No measurements.yaml in {run_dir}")

    raw = load_yaml(measurements_path)

    task_ids = sorted({m["task_a"] for m in raw} | {m["task_b"] for m in raw})
    tasks = [Task(prompt="", origin=OriginDataset.WILDCHAT, id=tid, metadata={}) for tid in task_ids]
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    n = len(tasks)
    wins = np.zeros((n, n), dtype=np.int32)
    for m in raw:
        i, j = id_to_idx[m["task_a"]], id_to_idx[m["task_b"]]
        if m["choice"] == "a":
            wins[i, j] += 1
        else:
            wins[j, i] += 1

    data = PairwiseData(tasks=tasks, wins=wins)
    result = fit_thurstonian(data)

    return result.mu, task_ids


def load_all_runs(results_dir: Path) -> list[tuple[RunConfig, np.ndarray, list[str]]]:
    """Returns list of (config, mu, task_ids)."""
    runs = list_runs(results_dir)
    loaded = []
    for config in runs:
        try:
            mu, task_ids = load_run_utilities(config.run_dir)
            loaded.append((config, mu, task_ids))
        except FileNotFoundError:
            print(f"Warning: Could not load data for {config.template_name}")
    return loaded


def compute_all_field_sensitivities(
    runs: list[tuple[RunConfig, np.ndarray, list[str]]],
) -> tuple[list[dict], list[dict]]:
    """Returns (sensitivities, correlations)."""
    results = {
        config.template_name: (mu, task_ids)
        for config, mu, task_ids in runs
    }
    tags = {
        config.template_name: config.template_tags
        for config, _, _ in runs
    }

    correlations = compute_pairwise_correlations(results, tags=tags)
    sensitivities = compute_sensitivities(correlations, correlation_key="correlation")

    sensitivities_list = [
        {
            "field": field,
            "mean": stats["mean"],
            "std": stats["std"],
            "n_pairs": stats["n_pairs"],
        }
        for field, stats in sensitivities.items()
    ]

    return sensitivities_list, correlations


def save_sensitivity_report(
    sensitivities: list[dict],
    correlations: list[dict],
    n_runs: int,
    output_path: Path,
) -> None:
    """Save sensitivity analysis results to YAML."""
    valid = [s for s in sensitivities if not np.isnan(s["mean"])]

    report = {
        "n_runs": n_runs,
        "by_field": {
            s["field"]: {
                "mean_correlation": float(s["mean"]),
                "std": float(s["std"]),
                "n_pairs": s["n_pairs"],
            }
            for s in valid
        },
        "pairwise_correlations": correlations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)


def plot_sensitivity_bars(
    sensitivities: list[dict],
    output_path: Path,
) -> None:
    if not sensitivities:
        return

    fields = [s["field"] for s in sensitivities]
    means = [s["mean"] for s in sensitivities]
    stds = [s["std"] for s in sensitivities]

    _, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(fields))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)

    ax.set_xlabel("Field Varied")
    ax.set_ylabel("Mean Utility Correlation (r)")
    ax.set_title("Preference Sensitivity by Template Field\n(higher = more robust)")
    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.9, color="green", linestyle="--", alpha=0.5, label="High robustness")
    ax.axhline(y=0.7, color="orange", linestyle="--", alpha=0.5, label="Moderate")
    ax.legend()

    for bar, mean in zip(bars, means):
        if not np.isnan(mean):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze preference sensitivity to template variations")
    parser.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=MEASUREMENTS_DIR,
        help="Directory containing measurement runs (default: results/measurements/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <results_dir>)",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return

    print(f"Loading runs from {args.results_dir}...")
    runs = load_all_runs(args.results_dir)

    if not runs:
        print("No measurement runs found.")
        return

    print(f"Loaded {len(runs)} runs, computing correlations...")
    sensitivities, correlations = compute_all_field_sensitivities(runs)

    output_dir = args.output or args.results_dir

    report_path = output_dir / "sensitivity.yaml"
    save_sensitivity_report(sensitivities, correlations, len(runs), report_path)
    print(f"Saved report to {report_path}")

    if sensitivities:
        plot_path = output_dir / "sensitivity.png"
        plot_sensitivity_bars(sensitivities, plot_path)
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
