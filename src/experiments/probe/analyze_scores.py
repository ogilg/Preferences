"""Analyze probe data scores along different dimensions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze scores by various dimensions")
    parser.add_argument("data_dir", type=Path, help="Directory containing probe data")
    parser.add_argument(
        "--token-bins",
        type=int,
        default=4,
        help="Number of bins for token count analysis",
    )
    return parser.parse_args()


def compute_stats(scores: list[float]) -> dict:
    arr = np.array(scores)
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


def print_stats(name: str, stats: dict) -> None:
    print(
        f"  {name}: n={stats['n']}, "
        f"mean={stats['mean']:.3f} ± {stats['std']:.3f}, "
        f"median={stats['median']:.3f}, "
        f"range=[{stats['min']:.2f}, {stats['max']:.2f}]"
    )


def analyze_by_origin(data_points: list[dict]) -> None:
    print("\n=== Scores by Origin Dataset ===")
    by_origin: dict[str, list[float]] = defaultdict(list)
    for dp in data_points:
        origin = dp.get("origin") or "unknown"
        by_origin[origin].append(dp["score"])

    for origin in sorted(by_origin.keys()):
        stats = compute_stats(by_origin[origin])
        print_stats(origin, stats)


def analyze_by_score_category(data_points: list[dict]) -> None:
    print("\n=== Score Distribution ===")
    categories = {"negative (< 0)": [], "neutral (= 0)": [], "positive (> 0)": []}
    for dp in data_points:
        score = dp["score"]
        if score < 0:
            categories["negative (< 0)"].append(score)
        elif score == 0:
            categories["neutral (= 0)"].append(score)
        else:
            categories["positive (> 0)"].append(score)

    for cat, scores in categories.items():
        if scores:
            print(f"  {cat}: n={len(scores)} ({100*len(scores)/len(data_points):.1f}%)")


def analyze_by_token_bins(data_points: list[dict], n_bins: int, field: str) -> None:
    scores_with_tokens = [
        (dp["score"], dp[field]) for dp in data_points if dp.get(field) is not None
    ]
    if not scores_with_tokens:
        print(f"  No {field} data available")
        return

    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile([v for _, v in scores_with_tokens], percentiles)

    bins: dict[str, list[float]] = defaultdict(list)
    for score, tokens in scores_with_tokens:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= tokens <= bin_edges[i + 1]:
                label = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
                bins[label].append(score)
                break

    for label in bins:
        stats = compute_stats(bins[label])
        print_stats(f"{field} {label}", stats)


def analyze_by_truncation(data_points: list[dict]) -> None:
    print("\n=== Scores by Truncation Status ===")
    truncated = [dp["score"] for dp in data_points if dp.get("truncated")]
    not_truncated = [dp["score"] for dp in data_points if not dp.get("truncated")]

    if truncated:
        stats = compute_stats(truncated)
        print_stats("truncated", stats)
    else:
        print("  truncated: n=0")

    if not_truncated:
        stats = compute_stats(not_truncated)
        print_stats("not truncated", stats)


def analyze_by_metadata(data_points: list[dict]) -> None:
    print("\n=== Scores by Task Metadata ===")

    by_type: dict[str, list[float]] = defaultdict(list)
    by_topic: dict[str, list[float]] = defaultdict(list)

    for dp in data_points:
        metadata = dp.get("task_metadata")
        if metadata:
            task_type = metadata.get("type")
            if task_type:
                by_type[str(task_type)].append(dp["score"])
            topic = metadata.get("topic")
            if topic:
                by_topic[str(topic)].append(dp["score"])

    if by_type:
        print("  By type:")
        for t in sorted(by_type.keys()):
            stats = compute_stats(by_type[t])
            print_stats(f"    {t}", stats)
    else:
        print("  No 'type' metadata found")

    if by_topic:
        print("  By topic:")
        for t in sorted(by_topic.keys())[:10]:
            stats = compute_stats(by_topic[t])
            print_stats(f"    {t}", stats)
        if len(by_topic) > 10:
            print(f"    ... and {len(by_topic) - 10} more topics")
    else:
        print("  No 'topic' metadata found")


def analyze_overall(data_points: list[dict]) -> None:
    print("\n=== Overall Statistics ===")
    scores = [dp["score"] for dp in data_points]
    stats = compute_stats(scores)
    print_stats("all", stats)

    prompt_tokens = [dp["prompt_tokens"] for dp in data_points if dp.get("prompt_tokens")]
    if prompt_tokens:
        print(
            f"  prompt_tokens: mean={np.mean(prompt_tokens):.1f}, "
            f"range=[{min(prompt_tokens)}, {max(prompt_tokens)}]"
        )

    completion_tokens = [
        dp["completion_tokens"] for dp in data_points if dp.get("completion_tokens")
    ]
    if completion_tokens:
        print(
            f"  completion_tokens: mean={np.mean(completion_tokens):.1f}, "
            f"range=[{min(completion_tokens)}, {max(completion_tokens)}]"
        )


def main() -> None:
    args = parse_args()

    print(f"Loading data from {args.data_dir}...")
    with open(args.data_dir / "completions.json") as f:
        data_points = json.load(f)
    print(f"Loaded {len(data_points)} data points")

    analyze_overall(data_points)
    analyze_by_origin(data_points)
    analyze_by_score_category(data_points)
    analyze_by_truncation(data_points)

    print(f"\n=== Scores by Prompt Token Count (quartiles) ===")
    analyze_by_token_bins(data_points, args.token_bins, "prompt_tokens")

    print(f"\n=== Scores by Completion Token Count (quartiles) ===")
    analyze_by_token_bins(data_points, args.token_bins, "completion_tokens")

    analyze_by_metadata(data_points)


if __name__ == "__main__":
    main()
