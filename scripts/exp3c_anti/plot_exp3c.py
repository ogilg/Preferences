"""Generate plots for exp3c anti-prompt analysis.

Loads results from exp3c_results.json and generates:
1. Target task rank distribution: A vs B vs C comparison
2. A/C probe delta paired scatter for target tasks
3. Per-pair bar chart showing A and C probe deltas

Usage: python scripts/exp3c_anti/plot_exp3c.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_FILE = REPO_ROOT / "experiments" / "ood_system_prompts" / "exp3c_anti" / "exp3c_results.json"
ASSETS_DIR = REPO_ROOT / "experiments" / "ood_system_prompts" / "exp3c_anti" / "assets"

LAYER = 31


def plot_rank_distributions(results, layer_key, assets_dir):
    """Target task probe rank distribution for A, B, C."""
    ldata = results[layer_key]
    pairs = ldata["pairs"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    n_tasks = 50

    configs = [
        ("A", "probe_rank_desc", "#e41a1c", "A (pro)\n1=highest probe delta"),
        ("B", "probe_rank_desc", "#377eb8", "B (neutral)\n1=highest probe delta"),
        ("C", "probe_rank_asc",  "#ff7f00", "C (anti)\n1=most negative probe delta"),
    ]

    for ax, (version, rank_key, color, title) in zip(axes, configs):
        ranks = [r[version][rank_key] for r in pairs
                 if version in r and r[version].get(rank_key) is not None]
        if not ranks:
            ax.set_title(f"{title}\n(no data)")
            continue
        ranks_arr = np.array(ranks)
        n = len(ranks_arr)
        bins = np.arange(0.5, n_tasks + 1.5, 1)
        ax.hist(ranks_arr, bins=bins, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(ranks_arr.mean(), color="black", linewidth=1.5, linestyle="--",
                   label=f"Mean: {ranks_arr.mean():.1f}")
        ax.axvline(25.5, color="gray", linewidth=1, linestyle=":", label="Chance: 25.5")
        ax.set_xlabel("Rank of target task")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}\nn={n}, mean={ranks_arr.mean():.1f}\ntop5={(ranks_arr<=5).sum()}/{n}")
        ax.set_xlim(0, n_tasks + 1)
        ax.set_ylim(0, max(n // 3, 5))
        ax.legend(fontsize=8)

    fig.suptitle(f"Exp 3 A/B/C: Target task probe rank ({layer_key})\n"
                 "A/B: rank among 50 tasks by probe delta desc; C: rank by probe delta asc",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    save_path = assets_dir / f"plot_022126_exp3c_rank_distributions.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def plot_ac_paired(results, layer_key, assets_dir):
    """Paired A/C probe deltas for target tasks."""
    pairs = results[layer_key]["pairs"]

    a_probes, c_probes, labels = [], [], []
    for pair in pairs:
        if "A" not in pair or "C" not in pair:
            continue
        a_pd = pair["A"].get("target_probe_delta")
        c_pd = pair["C"].get("target_probe_delta")
        if a_pd is None or c_pd is None:
            continue
        a_probes.append(a_pd)
        c_probes.append(c_pd)
        labels.append(f"{pair['base_role'][:2]}/{pair['target'][:5]}")

    if not a_probes:
        print("No A/C paired data")
        return

    a_arr = np.array(a_probes)
    c_arr = np.array(c_probes)
    n_correct = sum(1 for a, c in zip(a_arr, c_arr) if a > 0 and c < 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter
    ax1.scatter(a_arr, c_arr, s=60, color="#377eb8", alpha=0.8, zorder=2)
    for i, (a, c, lab) in enumerate(zip(a_arr, c_arr, labels)):
        ax1.annotate(lab, (a, c), fontsize=6, ha="left", va="bottom",
                     xytext=(2, 2), textcoords="offset points")
    ax1.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax1.axvline(0, color="gray", linewidth=0.7, linestyle="--")
    # Shade Q2 (A>0, C<0 = expected)
    xlim = max(abs(a_arr.min()), abs(a_arr.max())) * 1.1
    ylim = max(abs(c_arr.min()), abs(c_arr.max())) * 1.1
    ax1.fill_betweenx([0, -ylim], 0, xlim, alpha=0.05, color="#e41a1c")
    ax1.set_xlim(-xlim, xlim)
    ax1.set_ylim(-ylim, ylim)
    ax1.set_xlabel("A (pro) target task probe delta")
    ax1.set_ylabel("C (anti) target task probe delta")
    r, _ = stats.pearsonr(a_arr, c_arr)
    ax1.set_title(f"A vs C probe delta for target task ({layer_key})\n"
                  f"r={r:.2f}, n={len(a_arr)}, A>0&C<0: {n_correct}/{len(a_arr)}")
    ax1.text(0.05, 0.05, f"Expected quadrant\n(A>0, C<0): {n_correct}/{len(a_arr)}",
             transform=ax1.transAxes, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Bar chart: side-by-side per pair
    sort_idx = np.argsort(a_arr)[::-1]
    x = np.arange(len(a_arr))
    width = 0.35
    a_sorted = a_arr[sort_idx]
    c_sorted = c_arr[sort_idx]
    labels_sorted = np.array(labels)[sort_idx]

    ax2.bar(x - width/2, a_sorted, width, label="A (pro)", color="#e41a1c", alpha=0.7)
    ax2.bar(x + width/2, c_sorted, width, label="C (anti)", color="#ff7f00", alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_sorted, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Target task probe delta vs baseline")
    ax2.set_title(f"A vs C probe deltas per pair ({layer_key})\nSorted by A delta desc")
    ax2.legend(fontsize=9)
    lim = max(abs(a_sorted).max(), abs(c_sorted).max()) * 1.15
    ax2.set_ylim(-lim, lim)

    fig.tight_layout()
    save_path = assets_dir / f"plot_022126_exp3c_ac_paired.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def plot_correlation_summary(results, assets_dir):
    """Summary plot: Pearson r and sign agreement for A/B/C at each layer."""
    layers = ["L31", "L43", "L55"]
    versions = ["A", "B", "C"]
    colors = {"A": "#e41a1c", "B": "#377eb8", "C": "#ff7f00"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(layers))
    width = 0.25

    for i, version in enumerate(versions):
        rs = []
        signs = []
        for lk in layers:
            if lk not in results or version not in results[lk]["version_stats"]:
                rs.append(float("nan"))
                signs.append(float("nan"))
            else:
                s = results[lk]["version_stats"][version]
                rs.append(s["pearson_r"])
                signs.append(s["sign_agreement"])
        offset = (i - 1) * width
        ax1.bar(x + offset, rs, width, label=version, color=colors[version], alpha=0.75)
        ax2.bar(x + offset, [s * 100 for s in signs], width, color=colors[version], alpha=0.75)

    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_ylabel("Pearson r")
    ax1.set_title("Probe-behavioral correlation by version")
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.axhline(0, color="black", linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_ylabel("Sign agreement (%)")
    ax2.set_title("Sign agreement by version")
    ax2.set_ylim(0, 100)
    ax2.axhline(50, color="gray", linewidth=1, linestyle=":", label="Chance 50%")
    ax2.legend()

    fig.suptitle("Exp 3 A/B/C: Probe-behavioral correlation across layers\n"
                 "A=pro (interest sentence), B=neutral, C=anti (dislike sentence)",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    save_path = assets_dir / f"plot_022126_exp3c_correlation_summary.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_FILE.exists():
        print(f"Results file not found: {RESULTS_FILE}")
        print("Run analyze_exp3c.py first")
        return

    results = json.load(open(RESULTS_FILE))
    layer_key = f"L{LAYER}"

    print("Generating plots...")
    plot_rank_distributions(results, layer_key, ASSETS_DIR)
    plot_ac_paired(results, layer_key, ASSETS_DIR)
    plot_correlation_summary(results, ASSETS_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
