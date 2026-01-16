"""Sensitivity analysis for preference measurements.

Usage:
    python -m src.experiments.sensitivity_experiments.plot results/measurements/
    python -m src.experiments.sensitivity_experiments.plot results/stated/ --templates src/preferences/templates/data/stated_v1.yaml
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.experiments.correlation import compute_pairwise_correlations
from src.experiments.sensitivity_experiments.sensitivity import (
    compute_sensitivities,
    compute_sensitivity_regression,
)
from src.preferences.storage import MEASUREMENTS_DIR, load_yaml
from src.preferences.templates.template import load_templates_from_yaml


@dataclass
class RunConfig:
    template_name: str
    template_tags: dict
    model_short: str
    run_dir: Path


def _parse_stated_dir_name(dir_name: str) -> tuple[str, str] | None:
    """Parse 'stated_{template_name}_{model_short}' -> (template_name, model_short)."""
    match = re.match(r"stated_([^_]+_\d+)_(.+)$", dir_name)
    if match:
        return match.group(1), match.group(2)
    return None


def list_runs(results_dir: Path, template_yaml: Path | None = None) -> list[RunConfig]:
    runs = []
    if not results_dir.exists():
        return runs

    template_tags_map: dict[str, dict] | None = None

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.yaml"
        if config_path.exists():
            config = load_yaml(config_path)
            runs.append(RunConfig(
                template_name=config["template_name"],
                template_tags=config["template_tags"],
                model_short=config["model_short"],
                run_dir=run_dir,
            ))
        elif (run_dir / "measurements.yaml").exists():
            # Stated format: parse directory name
            parsed = _parse_stated_dir_name(run_dir.name)
            if parsed is None:
                continue
            template_name, model_short = parsed

            if template_tags_map is None:
                if template_yaml is None:
                    continue
                templates = load_templates_from_yaml(template_yaml)
                template_tags_map = {t.name: t.tags_dict for t in templates}

            if template_name not in template_tags_map:
                continue

            runs.append(RunConfig(
                template_name=template_name,
                template_tags=template_tags_map[template_name],
                model_short=model_short,
                run_dir=run_dir,
            ))
    return runs


def find_thurstonian_csv(run_dir: Path) -> Path | None:
    """Find pre-computed thurstonian CSV file (active learning only)."""
    # Try hash-based filename first
    matches = list(run_dir.glob("thurstonian_active_learning_*.csv"))
    if matches:
        return matches[0]

    # Fallback to old naming
    csv_path = run_dir / "thurstonian_active_learning.csv"
    if csv_path.exists():
        return csv_path

    return None


def _aggregate_scores(measurements: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Aggregate multiple samples per task into mean scores."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for m in measurements:
        by_task[m["task_id"]].append(m["score"])
    task_ids = sorted(by_task.keys())
    scores = np.array([np.mean(by_task[tid]) for tid in task_ids])
    return scores, task_ids


def load_run_utilities(run_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load utilities from thurstonian CSV, scores.yaml, or measurements.yaml."""
    # Try binary format first (thurstonian CSV)
    csv_path = find_thurstonian_csv(run_dir)
    if csv_path is not None:
        task_ids = []
        mus = []
        with open(csv_path) as f:
            next(f)  # Skip header
            for line in f:
                task_id, mu, _ = line.strip().split(",")
                task_ids.append(task_id)
                mus.append(float(mu))
        return np.array(mus), task_ids

    # Try rating format (scores.yaml)
    scores_path = run_dir / "scores.yaml"
    if scores_path.exists():
        scores = load_yaml(scores_path)
        task_ids = [s["task_id"] for s in scores]
        utilities = np.array([s["score"] for s in scores])
        return utilities, task_ids

    # Try stated format (measurements.yaml with raw samples)
    measurements_path = run_dir / "measurements.yaml"
    if measurements_path.exists():
        measurements = load_yaml(measurements_path)
        # Check if this is stated format (task_id, score) vs pairwise comparison (task_a, task_b, choice)
        if measurements and "task_id" in measurements[0]:
            return _aggregate_scores(measurements)
        # Pairwise comparison format without thurstonian CSV - skip
        raise FileNotFoundError(f"Pairwise comparison data without thurstonian CSV in {run_dir}")

    raise FileNotFoundError(f"No thurstonian CSV, scores.yaml, or measurements.yaml found in {run_dir}")


def load_all_runs(
    results_dir: Path,
    template_yaml: Path | None = None,
) -> list[tuple[RunConfig, np.ndarray, list[str]]]:
    """Returns list of (config, mu, task_ids)."""
    runs = list_runs(results_dir, template_yaml)
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
) -> tuple[list[dict], list[dict], dict]:
    """Returns (sensitivities_list, correlations, regression_results)."""
    # Use run_dir name as unique key (includes template, format, order, seed)
    results = {
        config.run_dir.name: (mu, task_ids)
        for config, mu, task_ids in runs
    }
    tags = {
        config.run_dir.name: config.template_tags
        for config, _, _ in runs
    }

    correlations = compute_pairwise_correlations(results, tags=tags)
    sensitivities = compute_sensitivities(correlations, correlation_key="correlation")
    regression = compute_sensitivity_regression(correlations, correlation_key="correlation")

    sensitivities_list = [
        {
            "field": field,
            "mean_when_same": stats["mean_when_same"],
            "mean_when_diff": stats["mean_when_diff"],
            "sensitivity": stats["sensitivity"],
            "std_when_diff": stats["std_when_diff"],
            "n_same": stats["n_same"],
            "n_diff": stats["n_diff"],
        }
        for field, stats in sensitivities.items()
    ]
    # Sort by sensitivity (highest impact first)
    sensitivities_list.sort(key=lambda x: -x["sensitivity"] if not np.isnan(x["sensitivity"]) else -999)

    return sensitivities_list, correlations, regression


def save_sensitivity_report(
    sensitivities: list[dict],
    correlations: list[dict],
    regression: dict,
    n_runs: int,
    output_path: Path,
) -> None:
    """Save sensitivity analysis results to YAML."""
    valid = [s for s in sensitivities if not np.isnan(s["sensitivity"])]

    # Format regression results
    regression_summary = {}
    if "_meta" in regression:
        regression_summary["intercept"] = regression["_meta"]["intercept"]
        regression_summary["r_squared"] = regression["_meta"]["r_squared"]
        regression_summary["n_pairs"] = regression["_meta"]["n_pairs"]
        regression_summary["coefficients"] = {
            field: data["coefficient"]
            for field, data in regression.items()
            if field != "_meta"
        }

    report = {
        "n_runs": n_runs,
        "regression": regression_summary,
        "by_field_averaging": {
            s["field"]: {
                "sensitivity": float(s["sensitivity"]),
                "mean_when_same": float(s["mean_when_same"]),
                "mean_when_diff": float(s["mean_when_diff"]),
                "std_when_diff": float(s["std_when_diff"]),
                "n_same": s["n_same"],
                "n_diff": s["n_diff"],
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
    title: str,
) -> None:
    if not sensitivities:
        return

    # Filter out NaN sensitivities and sort by sensitivity
    valid = [s for s in sensitivities if not np.isnan(s["sensitivity"])]
    if not valid:
        return

    fields = [s["field"] for s in valid]
    sens_values = [s["sensitivity"] for s in valid]
    stds = [s["std_when_diff"] for s in valid]

    _, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(fields))
    bars = ax.bar(x, sens_values, yerr=stds, capsize=5, color="steelblue", alpha=0.8)

    ax.set_xlabel("Field")
    ax.set_ylabel("Δ Correlation")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=45, ha="right")
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)

    for bar, val in zip(bars, sens_values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_regression_coefficients(
    regression: dict,
    output_path: Path,
    title: str,
) -> None:
    if "_meta" not in regression:
        return

    # Extract coefficients and std errors, excluding _meta
    items = [
        (field, data["coefficient"], data.get("std_err", 0))
        for field, data in regression.items()
        if field != "_meta"
    ]
    if not items:
        return

    # Sort by coefficient value (descending)
    items.sort(key=lambda x: -x[1])
    fields = [f for f, _, _ in items]
    coefs = [c for _, c, _ in items]
    std_errs = [se for _, _, se in items]

    _, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(fields))
    colors = ["steelblue" if c >= 0 else "coral" for c in coefs]
    ax.bar(x, coefs, yerr=std_errs, capsize=4, color=colors, alpha=0.8)

    ax.set_xlabel("Field")
    ax.set_ylabel("β Coefficient")
    r2 = regression["_meta"]["r_squared"]
    ax.set_title(f"{title} (R²={r2:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=45, ha="right")
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_sensitivities_for_runs(
    runs: list[tuple[RunConfig, np.ndarray, list[str]]],
) -> dict[str, float]:
    """Compute sensitivities for a subset of runs, returns field -> sensitivity."""
    if len(runs) < 2:
        return {}
    sens_list, _, _ = compute_all_field_sensitivities(runs)
    return {s["field"]: s["sensitivity"] for s in sens_list if not np.isnan(s["sensitivity"])}


def plot_sensitivity_by_model(
    runs: list[tuple[RunConfig, np.ndarray, list[str]]],
    output_path: Path,
    title: str,
) -> None:
    """Plot sensitivities grouped by model, one bar per model for each field."""
    # Group runs by model
    by_model: dict[str, list] = defaultdict(list)
    for config, mu, task_ids in runs:
        by_model[config.model_short].append((config, mu, task_ids))

    # Compute sensitivities per model
    model_sensitivities: dict[str, dict[str, float]] = {}
    for model, model_runs in sorted(by_model.items()):
        sens = compute_sensitivities_for_runs(model_runs)
        if sens:
            model_sensitivities[model] = sens

    if not model_sensitivities:
        return

    # Collect all fields across models
    all_fields: set[str] = set()
    for sens in model_sensitivities.values():
        all_fields.update(sens.keys())

    # Sort fields by average sensitivity across models
    field_avg = {}
    for field in all_fields:
        vals = [sens.get(field, 0) for sens in model_sensitivities.values()]
        field_avg[field] = np.mean([v for v in vals if v != 0])
    fields = sorted(all_fields, key=lambda f: -field_avg.get(f, 0))

    models = sorted(model_sensitivities.keys())
    n_models = len(models)
    n_fields = len(fields)

    _, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(n_fields)
    width = 0.8 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for i, model in enumerate(models):
        sens = model_sensitivities[model]
        values = [sens.get(field, 0) for field in fields]
        n_model_runs = len(by_model[model])
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=f"{model} (n={n_model_runs})", color=colors[i], alpha=0.8)

    ax.set_xlabel("Field")
    ax.set_ylabel("Δ Correlation (Sensitivity)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=45, ha="right")
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)

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
        help="Output directory (default: results/sensitivity_experiments/)",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=None,
        help="Template YAML file (required for stated results without config.yaml)",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return

    print(f"Loading runs from {args.results_dir}...")
    runs = load_all_runs(args.results_dir, args.templates)

    if not runs:
        print("No measurement runs found.")
        return

    print(f"Loaded {len(runs)} runs, computing correlations...")
    sensitivities, correlations, regression = compute_all_field_sensitivities(runs)

    output_dir = args.output or Path("results/sensitivity_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine measurement type and model for titles
    source_name = args.results_dir.name
    models = sorted(set(config.model_short for config, _, _ in runs))
    model_str = models[0] if len(models) == 1 else f"{len(models)} models"
    pref_type = "Revealed" if source_name == "measurements" else "Stated"
    date_str = datetime.now().strftime("%m%d%y")

    report_path = output_dir / f"sensitivity_{source_name}.yaml"
    save_sensitivity_report(sensitivities, correlations, regression, len(runs), report_path)
    print(f"Saved report to {report_path}")

    n_runs = len(runs)
    if sensitivities:
        plot_path = output_dir / f"plot_{date_str}_{pref_type.lower()}_sensitivity_averaging.png"
        title = f"{model_str} {pref_type} Pref Sensitivity (Averaging, n={n_runs})"
        plot_sensitivity_bars(sensitivities, plot_path, title)
        print(f"Saved plot to {plot_path}")

    if regression:
        plot_path = output_dir / f"plot_{date_str}_{pref_type.lower()}_sensitivity_regression.png"
        title = f"{model_str} {pref_type} Pref Sensitivity (Regression, n={n_runs})"
        plot_regression_coefficients(regression, plot_path, title)
        print(f"Saved regression plot to {plot_path}")

    # Per-model breakdown plots
    if len(models) > 1:
        plot_path = output_dir / f"plot_{date_str}_{pref_type.lower()}_sensitivity_by_model.png"
        plot_sensitivity_by_model(runs, plot_path, f"{pref_type} Pref Sensitivity by Model")
        print(f"Saved per-model plot to {plot_path}")


if __name__ == "__main__":
    main()
