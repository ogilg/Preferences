"""Analyze preference ratings by grouping (dataset, response format, template, etc.)."""

from __future__ import annotations

import argparse
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from src.measurement.storage.loading import (
    load_activation_metadata,
    load_raw_scores,
    list_runs,
)


def discover_templates(results_dir: Path, pattern: str | None = None) -> list[str]:
    """Discover all template names in results_dir, optionally filtered by pattern."""
    runs = list_runs(results_dir)
    template_names = {r.template_name for r in runs}

    if pattern:
        regex = re.compile(pattern)
        template_names = {t for t in template_names if regex.search(t)}

    return sorted(template_names)


def compute_grouped_stats(
    results_dir: Path,
    template_name: str,
    group_by: Literal["dataset", "template", "response_format"] = "dataset",
    seeds: list[int] | None = None,
) -> dict[str, dict]:
    """Compute average preference ratings grouped by specified dimension."""
    activation_metadata = load_activation_metadata()
    task_to_origin = {m.task_id: m.origin.lower() if m.origin else "unknown" for m in activation_metadata}

    raw_measurements = load_raw_scores(results_dir, [template_name], seeds)

    groups: dict[str, list[float]] = {}

    for task_id, score in raw_measurements:
        if score is None:
            continue

        if group_by == "dataset":
            group_val = task_to_origin.get(task_id, "unknown")
        elif group_by == "template":
            group_val = template_name
        elif group_by == "response_format":
            group_val = "unknown"
        else:
            group_val = "unknown"

        if group_val not in groups:
            groups[group_val] = []
        groups[group_val].append(score)

    results = {}
    for group_val, values in sorted(groups.items()):
        if not values:
            continue

        results[str(group_val)] = {
            "n": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "values": values,
        }

    return results


def compute_category_stats(
    results_dir: Path,
    categories: dict[str, list[str]],
    seeds: list[int] | None = None,
) -> dict[str, dict]:
    """Compute stats aggregated by category (e.g., qualitative vs stated)."""
    category_values: dict[str, list[float]] = {cat: [] for cat in categories}

    for category, template_names in categories.items():
        for template_name in template_names:
            raw_measurements = load_raw_scores(results_dir, [template_name], seeds)
            for _, score in raw_measurements:
                if score is not None:
                    category_values[category].append(score)

    results = {}
    for category, values in category_values.items():
        if not values:
            continue
        results[category] = {
            "n": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "sem": statistics.stdev(values) / (len(values) ** 0.5) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "values": values,
        }
    return results


def compute_category_by_dataset_stats(
    results_dir: Path,
    categories: dict[str, list[str]],
    seeds: list[int] | None = None,
) -> dict[str, dict[str, dict]]:
    """Compute stats by category AND dataset.

    Returns:
        dict[category][dataset] -> statistics
    """
    activation_metadata = load_activation_metadata()
    task_to_origin = {m.task_id: m.origin.lower() if m.origin else "unknown" for m in activation_metadata}

    # category -> dataset -> list of scores
    nested_values: dict[str, dict[str, list[float]]] = {cat: {} for cat in categories}

    for category, template_names in categories.items():
        for template_name in template_names:
            raw_measurements = load_raw_scores(results_dir, [template_name], seeds)
            for task_id, score in raw_measurements:
                if score is None:
                    continue
                dataset = task_to_origin.get(task_id, "unknown")
                if dataset not in nested_values[category]:
                    nested_values[category][dataset] = []
                nested_values[category][dataset].append(score)

    results: dict[str, dict[str, dict]] = {}
    for category, datasets in nested_values.items():
        results[category] = {}
        for dataset, values in datasets.items():
            if not values:
                continue
            results[category][dataset] = {
                "n": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "sem": statistics.stdev(values) / (len(values) ** 0.5) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
    return results


def print_results(
    results: dict[str, dict],
    group_by: str,
    template_name: str,
) -> None:
    """Print formatted results table."""
    if not results:
        print("No results")
        return

    # Header
    print("\n" + "=" * 100)
    print(f"Preference Ratings | Template: {template_name} | Grouped by: {group_by.upper()}")
    print("=" * 100)
    print(f"{'Group':<20} {'N':<8} {'Mean':<12} {'Median':<12} {'Stdev':<12} {'Min':<12} {'Max':<12}")
    print("-" * 100)

    # Rows
    for group_val in sorted(results.keys()):
        stats = results[group_val]
        print(
            f"{str(group_val):<20} {stats['n']:<8} {stats['mean']:<12.4f} "
            f"{stats['median']:<12.4f} {stats['stdev']:<12.4f} {stats['min']:<12.4f} {stats['max']:<12.4f}"
        )

    print("=" * 100 + "\n")


def plot_results(
    results: dict[str, dict],
    group_by: str,
    template_name: str,
    output_dir: Path | None = None,
) -> None:
    """Create bar chart with error bars."""
    if not results:
        print("No results to plot")
        return

    groups = sorted(results.keys())
    means = [results[g]["mean"] for g in groups]
    stdevs = [results[g]["stdev"] for g in groups]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(groups))
    bars = ax.bar(x, means, yerr=stdevs, capsize=5, alpha=0.7, color="steelblue", edgecolor="black")

    ax.set_xlabel(group_by.capitalize(), fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Preference Rating", fontsize=12, fontweight="bold")
    ax.set_title(f"Preference Ratings: {template_name}\nGrouped by {group_by.upper()}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    for bar, mean, stdev in zip(bars, means, stdevs):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + stdev + 0.05,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    if output_dir is None:
        output_dir = Path("src/analysis/general/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%m%d%y")
    template_short = template_name.replace("post_task_", "").replace("pre_task_", "")[:15]
    plot_path = output_dir / f"plot_{date_str}_ratings_{group_by}_{template_short}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_category_comparison(
    results: dict[str, dict],
    title: str = "Average Rating by Category",
    output_dir: Path | None = None,
    use_sem: bool = True,
) -> Path:
    """Create bar chart comparing categories (e.g., qualitative vs stated)."""
    if not results:
        print("No results to plot")
        return

    categories = list(results.keys())
    means = [results[c]["mean"] for c in categories]
    errors = [results[c]["sem" if use_sem else "stdev"] for c in categories]
    ns = [results[c]["n"] for c in categories]

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"][:len(categories)]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(categories))
    bars = ax.bar(x, means, yerr=errors, capsize=8, alpha=0.8, color=colors, edgecolor="black", linewidth=1.2)

    ax.set_ylabel("Average Rating", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, mean, err, n in zip(bars, means, errors, ns):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + err + 0.02,
            f"{mean:.3f}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    y_min = min(means) - max(errors) - 0.3
    y_max = max(means) + max(errors) + 0.3
    ax.set_ylim(max(0, y_min), y_max)

    plt.tight_layout()

    if output_dir is None:
        output_dir = Path("src/analysis/general/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%m%d%y")
    plot_path = output_dir / f"plot_{date_str}_category_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")
    return plot_path


def plot_category_by_dataset(
    results: dict[str, dict[str, dict]],
    title: str = "Average Rating by Category and Dataset",
    output_dir: Path | None = None,
    use_sem: bool = True,
    normalize: bool = False,
) -> Path:
    """Create grouped bar chart with datasets on x-axis and categories as colors."""
    if not results:
        print("No results to plot")
        return

    categories = list(results.keys())
    # Get all datasets across all categories
    all_datasets = set()
    for cat_data in results.values():
        all_datasets.update(cat_data.keys())
    datasets = sorted(all_datasets)

    # Category colors
    cat_colors = {"Qualitative": "#4C72B0", "Stated": "#55A868"}
    default_colors = ["#C44E52", "#8172B3", "#CCB974", "#64B5CD"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.35
    offsets = np.linspace(-width/2, width/2, len(categories))

    for i, category in enumerate(categories):
        means = []
        errors = []
        for dataset in datasets:
            if dataset in results[category]:
                stats = results[category][dataset]
                mean_val = stats["mean"]
                err_val = stats["sem" if use_sem else "stdev"]
                if normalize:
                    # Normalize: qualitative 0-2 -> 0-1, stated 0-5 -> 0-1
                    if category == "Qualitative":
                        mean_val = mean_val / 2.0
                        err_val = err_val / 2.0
                    elif category == "Stated":
                        mean_val = mean_val / 5.0
                        err_val = err_val / 5.0
                means.append(mean_val)
                errors.append(err_val)
            else:
                means.append(0)
                errors.append(0)

        color = cat_colors.get(category, default_colors[i % len(default_colors)])
        ax.bar(
            x + offsets[i],
            means,
            width * 0.9,
            yerr=errors,
            capsize=4,
            alpha=0.8,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            label=category,
        )

    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ylabel = "Normalized Rating (0-1)" if normalize else "Average Rating"
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    if output_dir is None:
        output_dir = Path("src/analysis/general/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%m%d%y")
    suffix = "_normalized" if normalize else ""
    plot_path = output_dir / f"plot_{date_str}_category_by_dataset{suffix}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")
    return plot_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze preference ratings grouped by dataset, template, or response format"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/experiments/probe_4_all_datasets"),
        help="Results directory (default: results/experiments/probe_4_all_datasets)",
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Single template name (e.g., post_task_qualitative_013)",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=["dataset", "template", "response_format"],
        default="dataset",
        help="Dimension to group by (default: dataset)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Specific seeds to include (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save plot (default: src/analysis/general/plots)",
    )
    parser.add_argument(
        "--compare-categories",
        action="store_true",
        help="Compare categories (qualitative vs stated) across all templates",
    )
    parser.add_argument(
        "--by-dataset",
        action="store_true",
        help="When using --compare-categories, also break down by dataset",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize scores to 0-1 scale (qualitative/2, stated/5)",
    )
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List all available templates in the results directory",
    )

    args = parser.parse_args()

    if args.list_templates:
        templates = discover_templates(args.results_dir)
        print(f"Found {len(templates)} templates in {args.results_dir}:")
        for t in templates:
            print(f"  {t}")
        return

    if args.compare_categories:
        # Discover all templates and categorize them
        all_templates = discover_templates(args.results_dir)
        categories = {
            "Qualitative": [t for t in all_templates if "qualitative" in t],
            "Stated": [t for t in all_templates if "stated" in t and "qualitative" not in t],
        }

        print(f"\nCategories discovered:")
        for cat, templates in categories.items():
            print(f"  {cat}: {len(templates)} templates")
            for t in templates:
                print(f"    - {t}")

        if args.by_dataset:
            # Breakdown by category AND dataset
            results = compute_category_by_dataset_stats(args.results_dir, categories, args.seeds)

            print("\n" + "=" * 100)
            print("Category x Dataset Breakdown")
            print("=" * 100)
            for cat, datasets in results.items():
                print(f"\n{cat}:")
                print(f"  {'Dataset':<15} {'N':<10} {'Mean':<12} {'SEM':<12} {'Stdev':<12}")
                print("  " + "-" * 70)
                for dataset, stats in sorted(datasets.items()):
                    print(f"  {dataset:<15} {stats['n']:<10} {stats['mean']:<12.4f} {stats['sem']:<12.4f} {stats['stdev']:<12.4f}")
            print("=" * 100)

            plot_category_by_dataset(
                results,
                "Average Rating by Category and Dataset",
                args.output_dir,
                normalize=args.normalize,
            )
        else:
            results = compute_category_stats(args.results_dir, categories, args.seeds)

            print("\n" + "=" * 80)
            print("Category Comparison")
            print("=" * 80)
            print(f"{'Category':<20} {'N':<10} {'Mean':<12} {'SEM':<12} {'Stdev':<12}")
            print("-" * 80)
            for cat, stats in results.items():
                print(f"{cat:<20} {stats['n']:<10} {stats['mean']:<12.4f} {stats['sem']:<12.4f} {stats['stdev']:<12.4f}")
            print("=" * 80)

            plot_category_comparison(results, "Average Rating: Qualitative vs Stated", args.output_dir)
        return

    if args.template is None:
        parser.error("--template is required unless using --compare-categories or --list-templates")

    results = compute_grouped_stats(
        results_dir=args.results_dir,
        template_name=args.template,
        group_by=args.group_by,
        seeds=args.seeds,
    )

    print_results(results, args.group_by, args.template)
    plot_results(results, args.group_by, args.template, args.output_dir)


if __name__ == "__main__":
    main()
