"""Plot relationship between refusal and preference (mu).

Usage:
    python -m src.analysis.active_learning.plot_refusal_preference --experiment-id gemma3_al_500

Requires running export_ranked_tasks.py first to generate the ranked_tasks JSON with refusal data.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pointbiserialr, mannwhitneyu

OUTPUT_DIR = Path(__file__).parent / "output"
PLOTS_DIR = Path(__file__).parent / "plots"


def load_ranked_tasks(experiment_id: str, run_name: str | None = None) -> list[dict]:
    """Load the most recent ranked tasks JSON for this experiment."""
    suffix = f"_{run_name}" if run_name else ""
    pattern = f"ranked_tasks_{experiment_id}{suffix}_*.json"
    candidates = list(OUTPUT_DIR.glob(pattern))
    if not candidates:
        raise ValueError(f"No ranked_tasks file found for pattern {pattern}. Run export_ranked_tasks.py first.")

    # Get most recent
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    path = candidates[0]
    print(f"Loading {path}")

    with open(path) as f:
        return json.load(f)


def compute_stats(tasks: list[dict]) -> dict:
    """Compute correlation statistics."""
    # Filter to tasks with refusal data
    tasks_with_refusal = [t for t in tasks if t["is_refusal"] is not None]

    refusals = np.array([t["is_refusal"] for t in tasks_with_refusal], dtype=int)
    mus = np.array([t["mu"] for t in tasks_with_refusal])

    n_refusals = int(refusals.sum())
    n_total = len(tasks_with_refusal)

    stats = {
        "n_total": n_total,
        "n_refusals": n_refusals,
        "refusal_rate": n_refusals / n_total if n_total > 0 else 0,
    }

    refused_mus = mus[refusals == 1]
    non_refused_mus = mus[refusals == 0]

    # Always compute means when data exists
    if len(refused_mus) > 0:
        stats["mean_mu_refused"] = float(refused_mus.mean())
        stats["std_mu_refused"] = float(refused_mus.std())
    if len(non_refused_mus) > 0:
        stats["mean_mu_non_refused"] = float(non_refused_mus.mean())
        stats["std_mu_non_refused"] = float(non_refused_mus.std())

    # Correlation only when both groups have data
    if n_refusals > 0 and n_refusals < n_total:
        corr, p_val = pointbiserialr(refusals, mus)
        stats["correlation"] = float(corr)
        stats["p_value"] = float(p_val)

        if len(refused_mus) >= 5 and len(non_refused_mus) >= 5:
            u_stat, u_p = mannwhitneyu(refused_mus, non_refused_mus, alternative="two-sided")
            stats["mann_whitney_u"] = float(u_stat)
            stats["mann_whitney_p"] = float(u_p)

    return stats


def compute_stats_by_dataset(tasks: list[dict]) -> dict[str, dict]:
    """Compute stats separately for each dataset."""
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_dataset[t["dataset"]].append(t)

    return {ds: compute_stats(ts) for ds, ts in by_dataset.items()}


def plot_refusal_vs_mu(tasks: list[dict], output_path: Path, experiment_id: str) -> None:
    """Create multi-panel plot showing refusal-preference relationship."""
    tasks_with_refusal = [t for t in tasks if t["is_refusal"] is not None]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Violin plot - mu distribution by refusal status (overall)
    ax1 = axes[0, 0]
    refused_mus = [t["mu"] for t in tasks_with_refusal if t["is_refusal"]]
    non_refused_mus = [t["mu"] for t in tasks_with_refusal if not t["is_refusal"]]

    violin_data = []
    positions = []
    labels = []
    if non_refused_mus:
        violin_data.append(non_refused_mus)
        positions.append(0)
        labels.append(f"Non-refused\n(n={len(non_refused_mus)})")
    if refused_mus:
        violin_data.append(refused_mus)
        positions.append(1)
        labels.append(f"Refused\n(n={len(refused_mus)})")

    if violin_data:
        parts = ax1.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)

    overall_stats = compute_stats(tasks)
    title = "Preference by Refusal Status (Overall)"
    if "correlation" in overall_stats:
        title += f"\nr={overall_stats['correlation']:.3f}, p={overall_stats['p_value']:.3g}"
    ax1.set_title(title)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Utility (μ)")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Panel 2: Bar chart - mean mu by refusal status per dataset
    ax2 = axes[0, 1]
    ds_stats = compute_stats_by_dataset(tasks)

    datasets = sorted(ds_stats.keys())
    x = np.arange(len(datasets))
    width = 0.35

    refused_means = []
    non_refused_means = []
    for ds in datasets:
        s = ds_stats[ds]
        refused_means.append(s.get("mean_mu_refused"))
        non_refused_means.append(s.get("mean_mu_non_refused"))

    # Plot bars, skipping None values
    non_ref_labeled = False
    ref_labeled = False
    for i, (ds, non_ref, ref) in enumerate(zip(datasets, non_refused_means, refused_means)):
        if non_ref is not None:
            label = "Non-refused" if not non_ref_labeled else ""
            ax2.bar(i - width / 2, non_ref, width, color="steelblue", alpha=0.8, label=label)
            non_ref_labeled = True
        if ref is not None:
            label = "Refused" if not ref_labeled else ""
            ax2.bar(i + width / 2, ref, width, color="coral", alpha=0.8, label=label)
            ref_labeled = True

    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45, ha="right")
    ax2.set_ylabel("Mean Utility (μ)")
    ax2.set_title("Mean Preference by Refusal Status per Dataset")
    ax2.legend()
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Panel 3: Scatter plot - mu vs rank colored by refusal
    ax3 = axes[1, 0]
    for t in tasks_with_refusal:
        color = "coral" if t["is_refusal"] else "steelblue"
        alpha = 0.6 if t["is_refusal"] else 0.3
        ax3.scatter(t["rank"], t["mu"], c=color, alpha=alpha, s=20)

    ax3.set_xlabel("Rank (1 = most preferred)")
    ax3.set_ylabel("Utility (μ)")
    ax3.set_title("Preference Ranking (coral = refusal)")
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Panel 4: Refusal rate by mu quartile
    ax4 = axes[1, 1]
    mus = np.array([t["mu"] for t in tasks_with_refusal])
    quartiles = np.percentile(mus, [25, 50, 75])

    def get_quartile(mu):
        if mu <= quartiles[0]:
            return "Q1 (lowest)"
        elif mu <= quartiles[1]:
            return "Q2"
        elif mu <= quartiles[2]:
            return "Q3"
        return "Q4 (highest)"

    quartile_refusals = defaultdict(lambda: {"total": 0, "refused": 0})
    for t in tasks_with_refusal:
        q = get_quartile(t["mu"])
        quartile_refusals[q]["total"] += 1
        if t["is_refusal"]:
            quartile_refusals[q]["refused"] += 1

    q_labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    refusal_rates = [
        quartile_refusals[q]["refused"] / quartile_refusals[q]["total"]
        if quartile_refusals[q]["total"] > 0 else 0
        for q in q_labels
    ]
    ns = [quartile_refusals[q]["total"] for q in q_labels]

    bars = ax4.bar(q_labels, [r * 100 for r in refusal_rates], color="coral", alpha=0.8)
    for bar, n in zip(bars, ns):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax4.set_ylabel("Refusal Rate (%)")
    ax4.set_xlabel("Preference Quartile")
    ax4.set_title("Refusal Rate by Preference Quartile")

    fig.suptitle(f"Refusal-Preference Analysis: {experiment_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {output_path}")


def print_summary(tasks: list[dict]) -> None:
    """Print summary statistics."""
    overall = compute_stats(tasks)
    ds_stats = compute_stats_by_dataset(tasks)

    print("\n" + "=" * 80)
    print("REFUSAL-PREFERENCE CORRELATION SUMMARY")
    print("=" * 80)

    print(f"\nOverall: {overall['n_refusals']}/{overall['n_total']} refusals ({overall['refusal_rate']:.1%})")
    if "correlation" in overall:
        print(f"Point-biserial correlation: r={overall['correlation']:.3f}, p={overall['p_value']:.3g}")
        print(f"Mean μ (refused): {overall['mean_mu_refused']:+.3f} ± {overall['std_mu_refused']:.3f}")
        print(f"Mean μ (non-refused): {overall['mean_mu_non_refused']:+.3f} ± {overall['std_mu_non_refused']:.3f}")
        if "mann_whitney_p" in overall:
            print(f"Mann-Whitney U p-value: {overall['mann_whitney_p']:.3g}")

    print("\n" + "-" * 80)
    print("BY DATASET:")
    print("-" * 80)
    print(f"{'Dataset':<12} {'N':>6} {'Refusals':>8} {'Rate':>8} {'μ(ref)':>10} {'μ(non)':>10} {'r':>8}")

    for ds in sorted(ds_stats.keys()):
        s = ds_stats[ds]
        ref_mu = f"{s['mean_mu_refused']:+.2f}" if "mean_mu_refused" in s else "N/A"
        non_mu = f"{s['mean_mu_non_refused']:+.2f}" if "mean_mu_non_refused" in s else "N/A"
        corr = f"{s['correlation']:.3f}" if "correlation" in s else "N/A"
        print(f"{ds:<12} {s['n_total']:>6} {s['n_refusals']:>8} {s['refusal_rate']:>7.1%} {ref_mu:>10} {non_mu:>10} {corr:>8}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Plot refusal-preference relationship")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None, help="Filter to run starting with this name (e.g., 'enjoy_most')")
    args = parser.parse_args()

    tasks = load_ranked_tasks(args.experiment_id, args.run_name)
    print(f"Loaded {len(tasks)} tasks")

    print_summary(tasks)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%m%d%y")
    suffix = f"_{args.run_name}" if args.run_name else ""
    output_path = PLOTS_DIR / f"plot_{date_str}_refusal_preference_{args.experiment_id}{suffix}.png"

    display_name = f"{args.experiment_id} ({args.run_name})" if args.run_name else args.experiment_id
    plot_refusal_vs_mu(tasks, output_path, display_name)


if __name__ == "__main__":
    main()
