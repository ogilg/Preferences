import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("results/experiments/mra_exp3/midway_bias/midway_bias_results.json")
ASSETS_DIR = Path("experiments/probe_generalization/multi_role_ablation/assets")

FOCUS_TOPICS = ["harmful_request", "math", "knowledge_qa", "fiction", "coding", "content_generation"]
SELECTOR_LABELS = {"turn_boundary:-2": "tb:-2", "turn_boundary:-5": "tb:-5"}
LAYERS = [25, 32, 39, 46, 53]


def load_data():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def entry_median_midway(entry: dict) -> float | None:
    """Median midway ratio across focus topics for a single entry."""
    ratios = []
    for topic in FOCUS_TOPICS:
        if topic in entry["topics"]:
            ratios.append(entry["topics"][topic]["midway_ratio"])
    if not ratios:
        return None
    return float(np.median(ratios))


def plot_midway_ratio_by_n(data: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, selector in zip(axes, ["turn_boundary:-2", "turn_boundary:-5"]):
        subset = [e for e in data if e["selector"] == selector]

        for is_in_dist, label, color, marker in [
            (True, "In-distribution", "#2196F3", "o"),
            (False, "OOD", "#E53935", "s"),
        ]:
            ns = sorted(set(e["n_personas"] for e in subset))
            medians = []
            lo_bars = []
            hi_bars = []

            for n in ns:
                entries = [e for e in subset if e["n_personas"] == n and e["is_in_dist"] == is_in_dist]
                per_entry_medians = []
                for e in entries:
                    m = entry_median_midway(e)
                    if m is not None:
                        per_entry_medians.append(m)
                if not per_entry_medians:
                    medians.append(np.nan)
                    lo_bars.append(0)
                    hi_bars.append(0)
                    continue
                arr = np.array(per_entry_medians)
                med = np.median(arr)
                q25 = np.percentile(arr, 25)
                q75 = np.percentile(arr, 75)
                medians.append(med)
                lo_bars.append(med - q25)
                hi_bars.append(q75 - med)

            ax.errorbar(
                ns, medians,
                yerr=[lo_bars, hi_bars],
                label=label, color=color, marker=marker,
                capsize=3, linewidth=1.5, markersize=5,
            )

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("N (training personas)", fontsize=12)
        ax.set_title(SELECTOR_LABELS[selector], fontsize=13)
        ax.set_ylim(-0.7, 1.5)
        ax.set_xticks(sorted(set(e["n_personas"] for e in subset)))
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Median midway ratio", fontsize=12)
    fig.suptitle("Midway bias: multi-persona training", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = ASSETS_DIR / "plot_031526_midway_ratio_by_n.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_midway_ratio_by_layer(data: list[dict]):
    fig, ax = plt.subplots(figsize=(7, 5))

    subset = [e for e in data if e["selector"] == "turn_boundary:-2" and not e["is_in_dist"]]
    # N=8 has no OOD entries (all personas in training), use N=7 instead
    n_values = [1, 2, 4, 7]
    colors = ["#1976D2", "#43A047", "#FB8C00", "#8E24AA"]

    for n, color in zip(n_values, colors):
        medians = []
        for layer in LAYERS:
            entries = [e for e in subset if e["n_personas"] == n and e["layer"] == layer]
            per_entry = [entry_median_midway(e) for e in entries]
            per_entry = [v for v in per_entry if v is not None]
            medians.append(np.median(per_entry) if per_entry else np.nan)

        ax.plot(LAYERS, medians, marker="o", label=f"N={n}", color=color, linewidth=1.5, markersize=5)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Median midway ratio", fontsize=12)
    ax.set_ylim(-1.0, 1.5)
    ax.set_xticks(LAYERS)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Midway ratio by layer (OOD, tb:-2)", fontsize=13, fontweight="bold")

    fig.tight_layout()
    out = ASSETS_DIR / "plot_031526_midway_ratio_by_layer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_pearson_r_by_n(data: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, selector in zip(axes, ["turn_boundary:-2", "turn_boundary:-5"]):
        subset = [e for e in data if e["selector"] == selector]

        for is_in_dist, label, color, marker in [
            (True, "In-distribution", "#2196F3", "o"),
            (False, "OOD", "#E53935", "s"),
        ]:
            ns = sorted(set(e["n_personas"] for e in subset))
            means = []

            for n in ns:
                entries = [e for e in subset if e["n_personas"] == n and e["is_in_dist"] == is_in_dist]
                rs = [e["pearson_r"] for e in entries]
                means.append(np.mean(rs) if rs else np.nan)

            ax.plot(ns, means, label=label, color=color, marker=marker, linewidth=1.5, markersize=5)

        ax.set_xlabel("N (training personas)", fontsize=12)
        ax.set_title(SELECTOR_LABELS[selector], fontsize=13)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(sorted(set(e["n_personas"] for e in subset)))
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Pearson r (mean)", fontsize=12)
    fig.suptitle("Pearson r by training diversity", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = ASSETS_DIR / "plot_031526_pearson_r_by_n.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    data = load_data()
    plot_midway_ratio_by_n(data)
    plot_midway_ratio_by_layer(data)
    plot_pearson_r_by_n(data)
    print("Done.")
