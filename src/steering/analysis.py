"""Analysis and plotting for steering validation experiments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_steering_results(experiment_dir: Path) -> dict:
    """Load steering experiment results from JSON."""
    results_path = experiment_dir / "steering_results.json"
    with open(results_path) as f:
        return json.load(f)


def aggregate_by_coefficient(results: dict) -> dict[float, list[float]]:
    """Aggregate parsed values by steering coefficient.

    Only includes conditions where parsing succeeded (numeric values).
    """
    by_coef: dict[float, list[float]] = {}
    for task_result in results["results"]:
        for cond in task_result["conditions"]:
            coef = cond["steering_coefficient"]
            parsed = cond["parsed_value"]
            # Skip non-numeric values (e.g., "refusal")
            if not isinstance(parsed, (int, float)):
                continue
            if coef not in by_coef:
                by_coef[coef] = []
            by_coef[coef].append(parsed)
    return by_coef


def compute_statistics(by_coef: dict[float, list[float]]) -> dict:
    """Compute summary statistics for steering effects."""
    coefficients = sorted(by_coef.keys())
    means = [np.mean(by_coef[c]) for c in coefficients]
    stds = [np.std(by_coef[c]) for c in coefficients]
    sems = [np.std(by_coef[c]) / np.sqrt(len(by_coef[c])) for c in coefficients]

    # Compute effect sizes
    if len(coefficients) >= 2:
        max_coef = max(coefficients)
        min_coef = min(coefficients)
        pos_scores = by_coef[max_coef]
        neg_scores = by_coef[min_coef]

        # Cohen's d between most positive and most negative
        pooled_std = np.sqrt((np.var(pos_scores) + np.var(neg_scores)) / 2)
        cohens_d = (np.mean(pos_scores) - np.mean(neg_scores)) / pooled_std if pooled_std > 0 else 0

        # Paired t-test (if same samples across conditions)
        if len(pos_scores) == len(neg_scores):
            t_stat, p_value = stats.ttest_rel(pos_scores, neg_scores)
        else:
            t_stat, p_value = stats.ttest_ind(pos_scores, neg_scores)
    else:
        cohens_d = 0.0
        t_stat, p_value = 0.0, 1.0

    # Linear regression: valence ~ steering_coefficient
    all_coefs = []
    all_scores = []
    for c, scores in by_coef.items():
        all_coefs.extend([c] * len(scores))
        all_scores.extend(scores)

    slope, intercept, r_value, p_slope, std_err = stats.linregress(all_coefs, all_scores)

    return {
        "coefficients": coefficients,
        "means": means,
        "stds": stds,
        "sems": sems,
        "cohens_d": cohens_d,
        "t_statistic": t_stat,
        "p_value_ttest": p_value,
        "regression_slope": slope,
        "regression_intercept": intercept,
        "regression_r2": r_value ** 2,
        "regression_p_value": p_slope,
        "n_per_condition": [len(by_coef[c]) for c in coefficients],
    }


def plot_dose_response(
    by_coef: dict[float, list[float]],
    stats_dict: dict,
    output_path: Path,
    title: str = "Steering Effect on Self-Reported Valence",
) -> None:
    """Plot dose-response curve: valence vs steering coefficient."""
    fig, ax = plt.subplots(figsize=(8, 6))

    coefficients = stats_dict["coefficients"]
    means = stats_dict["means"]
    sems = stats_dict["sems"]

    # Scatter individual points with jitter
    for coef in coefficients:
        scores = by_coef[coef]
        jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(scores))
        ax.scatter(
            [coef + j for j in jitter],
            scores,
            alpha=0.3,
            s=20,
            color="steelblue",
        )

    # Plot means with error bars
    ax.errorbar(
        coefficients,
        means,
        yerr=[1.96 * s for s in sems],
        fmt="o-",
        color="darkblue",
        markersize=10,
        linewidth=2,
        capsize=5,
        label="Mean ± 95% CI",
    )

    # Add regression line
    x_range = np.linspace(min(coefficients), max(coefficients), 100)
    y_pred = stats_dict["regression_slope"] * x_range + stats_dict["regression_intercept"]
    ax.plot(x_range, y_pred, "--", color="red", alpha=0.7, label=f"Linear fit (R²={stats_dict['regression_r2']:.3f})")

    # Reference line at zero
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Steering Coefficient", fontsize=12)
    ax.set_ylabel("Parsed Value", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc="upper left")

    # Add statistics text box
    stats_text = (
        f"Cohen's d: {stats_dict['cohens_d']:.2f}\n"
        f"Slope: {stats_dict['regression_slope']:.3f}\n"
        f"p (regression): {stats_dict['regression_p_value']:.4f}"
    )
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def analyze_steering_experiment(experiment_dir: Path, output_dir: Path | None = None) -> dict:
    """Run full analysis on steering experiment.

    Args:
        experiment_dir: Directory containing steering_results.json
        output_dir: Where to save plots (defaults to experiment_dir)
    """
    if output_dir is None:
        output_dir = experiment_dir

    # Load results
    results = load_steering_results(experiment_dir)
    config = results["config"]

    print(f"Analyzing steering experiment: {config['experiment_id']}")
    print(f"  Model: {config['model']}")
    print(f"  Probe: {config['probe_id']}")
    print(f"  Tasks: {len(results['results'])}")

    # Aggregate and compute statistics
    by_coef = aggregate_by_coefficient(results)
    stats_dict = compute_statistics(by_coef)

    # Print summary
    print("\nResults by steering coefficient:")
    for coef, mean, std, n in zip(
        stats_dict["coefficients"],
        stats_dict["means"],
        stats_dict["stds"],
        stats_dict["n_per_condition"],
    ):
        print(f"  {coef:+.1f}: mean={mean:.3f}, std={std:.3f}, n={n}")

    print(f"\nEffect size (Cohen's d): {stats_dict['cohens_d']:.3f}")
    print(f"Regression slope: {stats_dict['regression_slope']:.4f} (p={stats_dict['regression_p_value']:.4f})")
    print(f"R² (variance explained): {stats_dict['regression_r2']:.4f}")

    # Generate plots
    date_str = datetime.now().strftime("%m%d%y")
    plot_path = output_dir / f"plot_{date_str}_steering_dose_response.png"
    plot_dose_response(by_coef, stats_dict, plot_path)

    # Save statistics
    stats_path = output_dir / "steering_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    return stats_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze steering experiment results")
    parser.add_argument("experiment_dir", type=Path, help="Directory containing steering_results.json")
    parser.add_argument("--output-dir", type=Path, help="Output directory for plots (default: experiment_dir)")
    args = parser.parse_args()

    analyze_steering_experiment(args.experiment_dir, args.output_dir)


if __name__ == "__main__":
    main()
