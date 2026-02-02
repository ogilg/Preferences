"""Refusal-preference correlation analysis.

Analyzes correlation between task completion refusals and stated preference scores.

Usage:
    python -m src.analysis.correlation.refusal_preference_correlation --experiment-id refusal_pref_correlation
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from scipy.stats import pointbiserialr, mannwhitneyu

from src.measurement.storage import EXPERIMENTS_DIR
from src.measurement.storage.completions import extract_completion_text
from src.measurement.elicitation.refusal_judge import RefusalResult, judge_refusal_async

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "plots"


@dataclass
class CompletionWithScores:
    task_id: str
    task_prompt: str
    completion: str
    origin: str
    refusal: RefusalResult | None
    preference_scores: list[float]

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.preference_scores))

    @property
    def is_refusal(self) -> bool:
        return self.refusal is not None and self.refusal.is_refusal


@dataclass
class OriginStats:
    origin: str
    n_total: int
    n_refusals: int
    refusal_rate: float
    mean_score_refused: float | None
    mean_score_non_refused: float | None
    correlation: float | None
    p_value: float | None
    mann_whitney_u: float | None
    mann_whitney_p: float | None


def load_completions(activations_dir: Path) -> list[dict]:
    """Load completions from activations directory."""
    path = activations_dir / "completions_with_activations.json"
    with open(path) as f:
        return json.load(f)


def load_preference_scores_from_experiment(experiment_id: str) -> dict[str, list[float]]:
    """Load preference scores from an experiment's measurements.yaml files."""
    exp_dir = EXPERIMENTS_DIR / experiment_id / "post_task_stated"
    task_scores: dict[str, list[float]] = defaultdict(list)

    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir():
            continue
        measurements_path = run_dir / "measurements.yaml"
        if not measurements_path.exists():
            continue

        with open(measurements_path) as f:
            measurements = yaml.safe_load(f)

        for m in measurements:
            task_scores[m["task_id"]].append(m["score"])

    return dict(task_scores)


def load_refusal_cache(activations_dir: Path) -> dict[str, dict]:
    """Load cached refusal results."""
    cache_path = activations_dir / "refusal_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_refusal_cache(cache: dict[str, dict], activations_dir: Path) -> None:
    """Save refusal cache to disk."""
    cache_path = activations_dir / "refusal_cache.json"
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


async def detect_refusals_batch(
    completions: list[dict],
    activations_dir: Path,
    max_concurrent: int = 20,
) -> list[RefusalResult]:
    """Run refusal detection on completions, using cache for previously detected."""
    cache = load_refusal_cache(activations_dir)

    # Split into cached and uncached
    uncached_completions = []
    uncached_indices = []
    for i, c in enumerate(completions):
        if c["task_id"] not in cache:
            uncached_completions.append(c)
            uncached_indices.append(i)

    print(f"  Cache hit: {len(completions) - len(uncached_completions)}/{len(completions)}")
    print(f"  Need to detect: {len(uncached_completions)}")

    if uncached_completions:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def detect_one(c: dict) -> tuple[str, RefusalResult]:
            async with semaphore:
                completion = extract_completion_text(c["completion"])
                result = await judge_refusal_async(c["task_prompt"], completion)
                return c["task_id"], result

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Detecting refusals", total=len(uncached_completions))

            for coro in asyncio.as_completed([detect_one(c) for c in uncached_completions]):
                task_id, result = await coro
                cache[task_id] = result.model_dump()
                progress.update(task, advance=1)

        save_refusal_cache(cache, activations_dir)

    # Build results in original order
    return [RefusalResult.model_validate(cache[c["task_id"]]) for c in completions]


def build_dataset(
    completions: list[dict],
    refusals: list[RefusalResult],
    scores: dict[str, list[float]],
) -> list[CompletionWithScores]:
    """Join completions with refusal results and preference scores."""
    dataset = []
    for c, refusal in zip(completions, refusals):
        task_id = c["task_id"]
        if task_id not in scores:
            continue

        dataset.append(CompletionWithScores(
            task_id=task_id,
            task_prompt=c["task_prompt"],
            completion=extract_completion_text(c["completion"]),
            origin=c["origin"],
            refusal=refusal,
            preference_scores=scores[task_id],
        ))

    return dataset


def compute_correlation_stats(data: list[CompletionWithScores]) -> OriginStats | None:
    """Compute point-biserial correlation between refusal and preference score."""
    if len(data) < 10:
        return None

    refusals = np.array([d.is_refusal for d in data], dtype=int)
    scores = np.array([d.mean_score for d in data])

    n_refusals = int(refusals.sum())
    if n_refusals == 0 or n_refusals == len(data):
        # Can't compute correlation with no variance
        return OriginStats(
            origin=data[0].origin,
            n_total=len(data),
            n_refusals=n_refusals,
            refusal_rate=n_refusals / len(data),
            mean_score_refused=float(np.mean(scores[refusals == 1])) if n_refusals > 0 else None,
            mean_score_non_refused=float(np.mean(scores[refusals == 0])) if n_refusals < len(data) else None,
            correlation=None,
            p_value=None,
            mann_whitney_u=None,
            mann_whitney_p=None,
        )

    corr, p_val = pointbiserialr(refusals, scores)
    refused_scores = scores[refusals == 1]
    non_refused_scores = scores[refusals == 0]

    # Mann-Whitney U test for difference in distributions
    if len(refused_scores) >= 5 and len(non_refused_scores) >= 5:
        u_stat, u_p = mannwhitneyu(refused_scores, non_refused_scores, alternative="two-sided")
    else:
        u_stat, u_p = None, None

    return OriginStats(
        origin=data[0].origin,
        n_total=len(data),
        n_refusals=n_refusals,
        refusal_rate=n_refusals / len(data),
        mean_score_refused=float(np.mean(refused_scores)),
        mean_score_non_refused=float(np.mean(non_refused_scores)),
        correlation=float(corr),
        p_value=float(p_val),
        mann_whitney_u=float(u_stat) if u_stat is not None else None,
        mann_whitney_p=float(u_p) if u_p is not None else None,
    )


def analyze_by_origin(dataset: list[CompletionWithScores]) -> dict[str, OriginStats]:
    """Compute stats for each origin dataset."""
    by_origin: dict[str, list[CompletionWithScores]] = defaultdict(list)
    for d in dataset:
        by_origin[d.origin].append(d)

    stats = {}
    for origin, data in by_origin.items():
        result = compute_correlation_stats(data)
        if result:
            stats[origin] = result

    return stats


def plot_preference_distribution(
    dataset: list[CompletionWithScores],
    stats: dict[str, OriginStats],
    output_path: Path,
) -> None:
    """Plot preference score distributions for refused vs non-refused, faceted by origin."""
    origins = sorted(stats.keys())
    n_cols = 2
    n_rows = (len(origins) + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    by_origin: dict[str, list[CompletionWithScores]] = defaultdict(list)
    for d in dataset:
        by_origin[d.origin].append(d)

    for ax, origin in zip(axes, origins):
        data = by_origin[origin]
        refused_scores = [d.mean_score for d in data if d.is_refusal]
        non_refused_scores = [d.mean_score for d in data if not d.is_refusal]

        # Build violin data and positions based on what's available
        violin_data = []
        positions = []
        labels = []
        if non_refused_scores:
            violin_data.append(non_refused_scores)
            positions.append(0)
            labels.append("Non-refused")
        if refused_scores:
            violin_data.append(refused_scores)
            positions.append(1)
            labels.append("Refused")

        if violin_data:
            ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)

        origin_stats = stats[origin]
        ax.set_title(
            f"{origin}\n"
            f"n={origin_stats.n_total}, refusal rate={origin_stats.refusal_rate:.1%}\n"
            f"r={origin_stats.correlation:.3f}, p={origin_stats.p_value:.3g}" if origin_stats.correlation else
            f"{origin}\n"
            f"n={origin_stats.n_total}, refusal rate={origin_stats.refusal_rate:.1%}\n"
            f"(insufficient variance)"
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Preference Score")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Hide unused axes
    for ax in axes[len(origins):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_refusal_rate_by_origin(
    stats: dict[str, OriginStats],
    output_path: Path,
) -> None:
    """Bar chart of refusal rates by origin."""
    origins = sorted(stats.keys(), key=lambda o: stats[o].refusal_rate, reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    rates = [stats[o].refusal_rate * 100 for o in origins]
    colors = ["coral" if r > 10 else "steelblue" for r in rates]

    bars = ax.bar(origins, rates, color=colors)

    for bar, origin in zip(bars, origins):
        s = stats[origin]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"n={s.n_total}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Refusal Rate (%)")
    ax.set_xlabel("Origin Dataset")
    ax.set_title("Refusal Rate by Origin Dataset")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_summary_table(stats: dict[str, OriginStats]) -> None:
    """Print summary statistics table."""
    print("\n" + "=" * 100)
    print("REFUSAL-PREFERENCE CORRELATION ANALYSIS")
    print("=" * 100)

    headers = ["Origin", "N", "Refusals", "Rate", "Mean (Ref)", "Mean (Non)", "r_pb", "p-value", "MW-U p"]
    widths = [15, 8, 10, 8, 12, 12, 10, 12, 12]

    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * 100)

    for origin in sorted(stats.keys()):
        s = stats[origin]
        row = [
            origin,
            str(s.n_total),
            str(s.n_refusals),
            f"{s.refusal_rate:.1%}",
            f"{s.mean_score_refused:.3f}" if s.mean_score_refused is not None else "N/A",
            f"{s.mean_score_non_refused:.3f}" if s.mean_score_non_refused is not None else "N/A",
            f"{s.correlation:.3f}" if s.correlation is not None else "N/A",
            f"{s.p_value:.3g}" if s.p_value is not None else "N/A",
            f"{s.mann_whitney_p:.3g}" if s.mann_whitney_p is not None else "N/A",
        ]
        print(" | ".join(v.ljust(w) for v, w in zip(row, widths)))

    print("=" * 100)

    # Overall stats
    all_data = []
    for origin, s in stats.items():
        all_data.extend([(s.refusal_rate, s.n_total)] * 1)

    total_n = sum(s.n_total for s in stats.values())
    total_refusals = sum(s.n_refusals for s in stats.values())
    print(f"\nOverall: {total_refusals}/{total_n} refusals ({total_refusals/total_n:.1%})")


def print_refusal_examples(dataset: list[CompletionWithScores], n: int = 5) -> None:
    """Print example refusals for spot-checking."""
    refusals = [d for d in dataset if d.is_refusal]

    print(f"\n{'='*100}")
    print(f"EXAMPLE REFUSALS (showing {min(n, len(refusals))} of {len(refusals)})")
    print("=" * 100)

    for d in refusals[:n]:
        print(f"\n--- {d.task_id} ({d.origin}) ---")
        print(f"Refusal type: {d.refusal.refusal_type}, confidence: {d.refusal.confidence}")
        print(f"Preference score: {d.mean_score:.2f}")
        print(f"Prompt: {d.task_prompt[:200]}...")
        print(f"Completion: {d.completion[:300]}...")


async def main():
    parser = argparse.ArgumentParser(description="Refusal-preference correlation analysis")
    parser.add_argument("--experiment-id", type=str, required=True, help="Experiment ID with preference measurements")
    parser.add_argument("--activations-dir", type=Path, required=True, help="Path to activations directory (e.g., activations/llama_3_1_8b/)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%m%d%y")

    print(f"Loading completions from {args.activations_dir}...")
    all_completions = load_completions(args.activations_dir)
    print(f"Loaded {len(all_completions)} completions")

    print(f"\nLoading preference scores from experiment: {args.experiment_id}")
    scores = load_preference_scores_from_experiment(args.experiment_id)
    print(f"Loaded scores for {len(scores)} tasks")

    # Filter to only completions with preference scores
    completions = [c for c in all_completions if c["task_id"] in scores]
    print(f"Filtered to {len(completions)} completions with preference scores")

    print("\nRunning refusal detection...")
    refusals = await detect_refusals_batch(completions, args.activations_dir)

    print("\nBuilding dataset...")
    dataset = build_dataset(completions, refusals, scores)
    print(f"Dataset size: {len(dataset)} (with both refusal and score data)")

    print("\nComputing statistics by origin...")
    stats = analyze_by_origin(dataset)

    print_summary_table(stats)
    print_refusal_examples(dataset)

    # Generate plots
    dist_plot_path = OUTPUT_DIR / f"plot_{date_str}_refusal_preference_distribution.png"
    plot_preference_distribution(dataset, stats, dist_plot_path)
    print(f"\nSaved distribution plot to {dist_plot_path}")

    rate_plot_path = OUTPUT_DIR / f"plot_{date_str}_refusal_rate_by_origin.png"
    plot_refusal_rate_by_origin(stats, rate_plot_path)
    print(f"Saved refusal rate plot to {rate_plot_path}")

    # Save raw stats as JSON
    stats_path = OUTPUT_DIR / f"refusal_preference_stats_{date_str}.json"
    stats_dict = {
        origin: {
            "n_total": s.n_total,
            "n_refusals": s.n_refusals,
            "refusal_rate": s.refusal_rate,
            "mean_score_refused": s.mean_score_refused,
            "mean_score_non_refused": s.mean_score_non_refused,
            "correlation": s.correlation,
            "p_value": s.p_value,
            "mann_whitney_u": s.mann_whitney_u,
            "mann_whitney_p": s.mann_whitney_p,
        }
        for origin, s in stats.items()
    }
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    asyncio.run(main())
