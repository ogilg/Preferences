"""Plot unified metrics: all conditions on same metric.

Usage:
    python -m scripts.hoo_scaled.plot_unified [--output-dir experiments/hoo_scaled/assets]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

UNIFIED_PATH = "results/probes/hoo_scaled_unified_metrics.json"

CONDITIONS_L31 = [
    ("Ridge raw", "Ridge raw_L31"),
    ("Ridge topic-\ndemeaned", "Ridge topic-demeaned_L31"),
    ("BT raw", "BT raw_L31"),
    ("Content\nbaseline", "Content baseline_L0"),
]

COLORS = ["#3498db", "#2ecc71", "#9b59b6", "#e67e22"]


def load_unified() -> list[dict]:
    with open(UNIFIED_PATH) as f:
        return json.load(f)


def extract_metric(results: list[dict], key: str, metric: str) -> list[float]:
    return [f["conditions"][key][metric] for f in results if key in f["conditions"]]


def plot_boxplots(results: list[dict], output_dir: Path) -> Path:
    """Side-by-side boxplots: Pearson r and pairwise accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel, title, chance in [
        (ax1, "pearson_r", "Held-out Pearson r", "Pearson r with true Thurstonian scores", None),
        (ax2, "pairwise_acc", "Held-out pairwise accuracy", "Pairwise accuracy on held-out comparisons", 0.5),
    ]:
        data = []
        labels = []
        for label, key in CONDITIONS_L31:
            vals = extract_metric(results, key, metric)
            if vals:
                data.append(vals)
                labels.append(label)

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
        for patch, color in zip(bp["boxes"], COLORS[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for i, d in enumerate(data):
            ax.text(i + 1, np.mean(d) + 0.01, f"{np.mean(d):.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="red")

        if chance is not None:
            ax.axhline(y=chance, color="gray", linewidth=1, linestyle="--", label=f"Chance ({chance})")
            ax.legend(fontsize=9)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("Held-out topic generalization — Layer 31 (all 56 folds)", fontsize=13, y=1.02)
    plt.tight_layout()

    plot_path = output_dir / "plot_021126_hoo_scaled_unified_L31.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")
    return plot_path


def plot_layer_comparison(results: list[dict], output_dir: Path) -> Path:
    """Line plot: held-out Pearson r across layers for each condition."""
    layers = [31, 43, 55]
    conditions = [
        ("Ridge raw", "Ridge raw", "#3498db"),
        ("Ridge topic-demeaned", "Ridge topic-demeaned", "#2ecc71"),
        ("BT raw", "BT raw", "#9b59b6"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for label, cond_prefix, color in conditions:
        means = []
        stds = []
        for layer in layers:
            key = f"{cond_prefix}_L{layer}"
            vals = extract_metric(results, key, "pearson_r")
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.errorbar(layers, means, yerr=stds, marker='o', label=label,
                     color=color, linewidth=2, capsize=4, markersize=7)

    # Content baseline (single layer, show as horizontal line)
    cb_vals = extract_metric(results, "Content baseline_L0", "pearson_r")
    cb_mean = np.mean(cb_vals)
    ax.axhline(y=cb_mean, color="#e67e22", linewidth=2, linestyle="--",
               label=f"Content baseline (r={cb_mean:.3f})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Held-out Pearson r")
    ax.set_title("Held-out generalization across layers")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}\n({l/62*100:.0f}%)" for l in layers])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plot_path = output_dir / "plot_021126_hoo_scaled_layer_comparison.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")
    return plot_path


def plot_generalization_gap(results: list[dict], output_dir: Path) -> Path:
    """Bar chart: in-dist vs held-out Pearson r, showing the gap."""
    # Use original summaries for in-dist values
    import json as json2
    summaries = {}
    for label, path in [
        ("Ridge raw", "results/probes/hoo_scaled_raw/hoo_summary.json"),
        ("Ridge topic-demeaned", "results/probes/hoo_scaled_demeaned/hoo_summary.json"),
        ("Content baseline", "results/probes/hoo_scaled_st_baseline/hoo_summary.json"),
    ]:
        with open(path) as f:
            summaries[label] = json2.load(f)

    conditions = ["Ridge raw", "Ridge topic-demeaned", "Content baseline"]
    in_dist = []
    held_out = []
    for cond in conditions:
        summary = summaries[cond]
        if cond == "Content baseline":
            layer_key = "ridge_L0"
        else:
            layer_key = "ridge_L31"
        vals_r = [f["layers"][layer_key]["val_r"] for f in summary["folds"]
                  if layer_key in f["layers"]]
        hoo_r = [f["layers"][layer_key]["hoo_r"] for f in summary["folds"]
                 if layer_key in f["layers"]]
        in_dist.append(np.mean(vals_r))
        held_out.append(np.mean(hoo_r))

    # Add BT from unified metrics
    conditions.append("BT raw")
    bt_hoo = extract_metric(results, "BT raw_L31", "pearson_r")
    held_out.append(np.mean(bt_hoo))
    # BT in-dist: use val_acc from original summary as proxy — but we want Pearson r
    # We don't have in-dist Pearson r for BT, so skip this plot for now
    # Actually we can compute it: BT in-dist means training on all 5 topics
    # The val_acc is from CV within training, not the same. Let's just show held-out.

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_id = ax.bar(x - width/2, in_dist[:3], width, label="In-distribution", color="#a8d8ea", edgecolor="black", linewidth=0.5)
    bars_ho = ax.bar(x[:3] + width/2, held_out[:3], width, label="Held-out", color="#3498db", edgecolor="black", linewidth=0.5)

    # Add gap annotations
    for i in range(3):
        gap = in_dist[i] - held_out[i]
        mid_y = (in_dist[i] + held_out[i]) / 2
        ax.annotate(f"gap={gap:.3f}", xy=(i + 0.25, mid_y), fontsize=8, ha="left", color="red")

    ax.set_ylabel("Pearson r")
    ax.set_title("In-distribution vs held-out performance (Layer 31)")
    ax.set_xticks(x[:3])
    ax.set_xticklabels(conditions[:3], rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plot_path = output_dir / "plot_021126_hoo_scaled_gap_comparison.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")
    return plot_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/hoo_scaled/assets"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = load_unified()
    plot_boxplots(results, args.output_dir)
    plot_layer_comparison(results, args.output_dir)


if __name__ == "__main__":
    main()
