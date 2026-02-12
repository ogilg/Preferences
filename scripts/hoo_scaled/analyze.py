"""Analyze and compare scaled HOO results across conditions.

Usage:
    python scripts/hoo_scaled/analyze.py [--output-dir experiments/ood_generalization/hoo_scaled/assets]
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS = {
    "Ridge raw": "results/probes/hoo_scaled_raw/hoo_summary.json",
    "Ridge demeaned": "results/probes/hoo_scaled_demeaned/hoo_summary.json",
    "ST baseline": "results/probes/hoo_scaled_st_baseline/hoo_summary.json",
}
BT_RESULT = "results/probes/hoo_scaled_bt/hoo_summary.json"


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_ridge_hoo_r(summary: dict, layer: int) -> list[float]:
    key = f"ridge_L{layer}"
    return [
        f["layers"][key]["hoo_r"]
        for f in summary["folds"]
        if key in f["layers"] and f["layers"][key].get("hoo_r") is not None
    ]


def extract_ridge_val_r(summary: dict, layer: int) -> list[float]:
    key = f"ridge_L{layer}"
    return [
        f["layers"][key]["val_r"]
        for f in summary["folds"]
        if key in f["layers"] and f["layers"][key].get("val_r") is not None
    ]


def extract_bt_hoo_acc(summary: dict, layer: int) -> list[float]:
    key = f"bradley_terry_L{layer}"
    return [
        f["layers"][key]["hoo_acc"]
        for f in summary["folds"]
        if key in f["layers"] and f["layers"][key].get("hoo_acc") is not None
    ]


def extract_bt_val_acc(summary: dict, layer: int) -> list[float]:
    key = f"bradley_terry_L{layer}"
    return [
        f["layers"][key]["val_acc"]
        for f in summary["folds"]
        if key in f["layers"] and f["layers"][key].get("val_acc") is not None
    ]


def per_topic_breakdown(summary: dict, layer: int, method: str = "ridge") -> dict[str, list[float]]:
    """For each topic, collect the HOO metric from all folds where that topic was held out."""
    topic_metrics: dict[str, list[float]] = defaultdict(list)
    for fold in summary["folds"]:
        held_out = fold["held_out_groups"]
        if method == "ridge":
            key = f"ridge_L{layer}"
            metric_name = "hoo_r"
        else:
            key = f"bradley_terry_L{layer}"
            metric_name = "hoo_acc"
        if key not in fold["layers"]:
            continue
        val = fold["layers"][key].get(metric_name)
        if val is None:
            continue
        for topic in held_out:
            topic_metrics[topic].append(val)
    return dict(topic_metrics)


def print_summary_table(summaries: dict[str, dict], bt_summary: dict | None, layers: list[int]) -> str:
    """Print and return the main comparison table."""
    lines = []
    header = "| Method | Layer | Mean HOO metric | Std | Mean val metric | Gap (val - hoo) | N folds |"
    sep = "|--------|-------|-----------------|-----|-----------------|-----------------|---------|"
    lines.append(header)
    lines.append(sep)

    for label, summary in summaries.items():
        for layer in layers:
            hoo_rs = extract_ridge_hoo_r(summary, layer)
            val_rs = extract_ridge_val_r(summary, layer)
            if not hoo_rs:
                continue
            mean_hoo = np.mean(hoo_rs)
            std_hoo = np.std(hoo_rs)
            mean_val = np.mean(val_rs)
            gap = mean_val - mean_hoo
            lines.append(
                f"| {label} | {layer} | {mean_hoo:.4f} | {std_hoo:.4f} | {mean_val:.4f} | {gap:.4f} | {len(hoo_rs)} |"
            )

    # ST baseline (single layer)
    if "ST baseline" in summaries:
        st_summary = summaries["ST baseline"]
        st_layers = st_summary["layers"]
        for layer in st_layers:
            hoo_rs = extract_ridge_hoo_r(st_summary, layer)
            val_rs = extract_ridge_val_r(st_summary, layer)
            if not hoo_rs:
                continue
            mean_hoo = np.mean(hoo_rs)
            std_hoo = np.std(hoo_rs)
            mean_val = np.mean(val_rs)
            gap = mean_val - mean_hoo
            lines.append(
                f"| ST baseline | {layer} | {mean_hoo:.4f} | {std_hoo:.4f} | {mean_val:.4f} | {gap:.4f} | {len(hoo_rs)} |"
            )

    if bt_summary is not None:
        for layer in layers:
            hoo_accs = extract_bt_hoo_acc(bt_summary, layer)
            val_accs = extract_bt_val_acc(bt_summary, layer)
            if not hoo_accs:
                continue
            mean_hoo = np.mean(hoo_accs)
            std_hoo = np.std(hoo_accs)
            mean_val = np.mean(val_accs)
            gap = mean_val - mean_hoo
            lines.append(
                f"| BT raw | {layer} | {mean_hoo:.4f} | {std_hoo:.4f} | {mean_val:.4f} | {gap:.4f} | {len(hoo_accs)} |"
            )

    table = "\n".join(lines)
    print(table)
    return table


def paired_comparison(
    hoo_a: list[float], hoo_b: list[float], label_a: str, label_b: str,
) -> str:
    """Paired t-test and sign test comparing two conditions on same folds."""
    n = min(len(hoo_a), len(hoo_b))
    a, b = np.array(hoo_a[:n]), np.array(hoo_b[:n])
    diff = a - b
    t_stat, t_p = stats.ttest_rel(a, b)
    wins_a = int(np.sum(diff > 0))
    wins_b = int(np.sum(diff < 0))
    ties = n - wins_a - wins_b
    sign_p = stats.binomtest(wins_a, wins_a + wins_b, 0.5).pvalue if (wins_a + wins_b) > 0 else 1.0
    result = (
        f"{label_a} vs {label_b} (n={n}):\n"
        f"  Mean diff: {np.mean(diff):.4f} (± {np.std(diff):.4f})\n"
        f"  Paired t: t={t_stat:.3f}, p={t_p:.4f}\n"
        f"  Sign test: {wins_a} wins / {wins_b} losses / {ties} ties, p={sign_p:.4f}"
    )
    print(result)
    return result


def plot_boxplots(
    summaries: dict[str, dict],
    bt_summary: dict | None,
    layer: int,
    output_dir: Path,
) -> Path:
    """Box plot comparing all conditions for a single layer."""
    data = []
    labels = []
    for label, summary in summaries.items():
        hoo_rs = extract_ridge_hoo_r(summary, layer)
        if hoo_rs:
            data.append(hoo_rs)
            labels.append(f"{label}\n(Pearson r)")

    # Include ST baseline if this is the best layer (ST only has layer 0)
    if "ST baseline" in summaries:
        st_layers = summaries["ST baseline"]["layers"]
        st_layer = st_layers[0] if st_layers else 0
        st_hoo = extract_ridge_hoo_r(summaries["ST baseline"], st_layer)
        if st_hoo:
            data.append(st_hoo)
            labels.append(f"ST baseline\n(Pearson r)")

    if bt_summary is not None:
        hoo_accs = extract_bt_hoo_acc(bt_summary, layer)
        if hoo_accs:
            data.append(hoo_accs)
            labels.append("BT raw\n(accuracy)")

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("HOO metric (Pearson r or accuracy)")
    ax.set_title(f"Scaled HOO (hold_out=3) — Layer {layer}")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(y=0, color="black", linewidth=0.8)

    for i, d in enumerate(data):
        ax.text(i + 1, np.mean(d), f"μ={np.mean(d):.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="red")

    plt.tight_layout()
    plot_path = output_dir / f"plot_021126_hoo_scaled_boxplot_L{layer}.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")
    return plot_path


def plot_per_topic(
    summaries: dict[str, dict],
    layer: int,
    output_dir: Path,
) -> Path:
    """Per-topic breakdown: which topics are hardest to generalize to?"""
    all_topics = set()
    topic_data: dict[str, dict[str, list[float]]] = {}
    for label, summary in summaries.items():
        # ST baseline uses layer 0, others use the requested layer
        effective_layer = summary["layers"][0] if summary["layers"][0] != layer and label == "ST baseline" else layer
        breakdown = per_topic_breakdown(summary, effective_layer, "ridge")
        topic_data[label] = breakdown
        all_topics.update(breakdown.keys())

    topics = sorted(all_topics)
    x = np.arange(len(topics))
    width = 0.8 / len(topic_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#3498db", "#2ecc71", "#e67e22"]
    for i, (label, breakdown) in enumerate(topic_data.items()):
        means = [np.mean(breakdown.get(t, [0])) for t in topics]
        stds = [np.std(breakdown.get(t, [0])) for t in topics]
        ax.bar(x + i * width, means, width, yerr=stds, label=label, color=colors[i % len(colors)],
               alpha=0.7, capsize=3)

    ax.set_xticks(x + width * (len(topic_data) - 1) / 2)
    ax.set_xticklabels(topics, rotation=30, ha="right")
    ax.set_ylabel("Mean HOO Pearson r")
    ax.set_title(f"Per-Topic HOO Generalization — Layer {layer}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(y=0, color="black", linewidth=0.8)

    plt.tight_layout()
    plot_path = output_dir / f"plot_021126_hoo_scaled_per_topic_L{layer}.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")
    return plot_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ood_generalization/hoo_scaled/assets"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all available summaries
    summaries = {}
    for label, path in RESULTS.items():
        if Path(path).exists():
            summaries[label] = load_summary(path)
            print(f"Loaded {label}: {path}")
        else:
            print(f"Skipping {label}: {path} not found")

    bt_summary = None
    if Path(BT_RESULT).exists():
        bt_summary = load_summary(BT_RESULT)

    if not summaries:
        print("No results found!")
        return

    # Determine layers from first summary
    first = next(iter(summaries.values()))
    layers = first["layers"]

    print("\n" + "=" * 80)
    print("MAIN COMPARISON TABLE")
    print("=" * 80 + "\n")
    table = print_summary_table(summaries, bt_summary, layers)

    # Best layer analysis
    best_layer = None
    best_hoo = -1
    for label, summary in summaries.items():
        if "raw" in label.lower() and "ST" not in label:
            for layer in layers:
                vals = extract_ridge_hoo_r(summary, layer)
                if vals and np.mean(vals) > best_hoo:
                    best_hoo = np.mean(vals)
                    best_layer = layer
    if best_layer is None:
        best_layer = layers[len(layers) // 2]

    print(f"\nBest Ridge raw layer: {best_layer} (mean hoo_r={best_hoo:.4f})")

    # Paired comparisons at best layer
    print("\n" + "=" * 80)
    print(f"PAIRED COMPARISONS (Layer {best_layer})")
    print("=" * 80 + "\n")

    comparison_results = []
    if "Ridge raw" in summaries and "ST baseline" in summaries:
        ridge_hoo = extract_ridge_hoo_r(summaries["Ridge raw"], best_layer)
        st_layers = summaries["ST baseline"]["layers"]
        st_layer = st_layers[0] if st_layers else 0
        st_hoo = extract_ridge_hoo_r(summaries["ST baseline"], st_layer)
        comparison_results.append(paired_comparison(ridge_hoo, st_hoo, f"Ridge raw L{best_layer}", f"ST baseline L{st_layer}"))

    if "Ridge raw" in summaries and "Ridge demeaned" in summaries:
        ridge_hoo = extract_ridge_hoo_r(summaries["Ridge raw"], best_layer)
        demeaned_hoo = extract_ridge_hoo_r(summaries["Ridge demeaned"], best_layer)
        comparison_results.append(paired_comparison(ridge_hoo, demeaned_hoo, "Ridge raw", "Ridge demeaned"))

    # Plots
    print("\n" + "=" * 80)
    print("PLOTS")
    print("=" * 80 + "\n")

    plot_paths = []
    for layer in layers:
        plot_paths.append(plot_boxplots(summaries, bt_summary, layer, args.output_dir))
        if len(summaries) > 1:
            plot_paths.append(plot_per_topic(summaries, layer, args.output_dir))

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
