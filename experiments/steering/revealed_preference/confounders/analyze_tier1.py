"""Analyze Tier 1 experiment results: E1, E3, E8."""

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


OUTPUT_DIR = Path("experiments/steering/revealed_preference/confounders")
ASSETS_DIR = Path("docs/logs/assets/steering_confounders")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

DATE_STR = datetime.now().strftime("%m%d%y")


def analyze_e1():
    """E1: Order counterbalancing — does the effect reverse when order swaps?"""
    path = OUTPUT_DIR / "e1_order_counterbalance_results.json"
    if not path.exists():
        print("E1 results not found")
        return

    with open(path) as f:
        results = json.load(f)

    valid = [r for r in results if r["choice"] is not None]
    coefficients = sorted(set(r["coefficient"] for r in valid))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: P(A) by coefficient for both orderings
    ax = axes[0]
    for ordering, color, marker in [("original", "blue", "o"), ("swapped", "red", "s")]:
        p_a_vals = []
        for coef in coefficients:
            matching = [r for r in valid if r["ordering"] == ordering and r["coefficient"] == coef]
            if matching:
                n_a = sum(1 for r in matching if r["choice"] == "a")
                p_a_vals.append(n_a / len(matching))
            else:
                p_a_vals.append(np.nan)
        ax.plot(coefficients, p_a_vals, f"{marker}-", color=color, markersize=8, linewidth=2, label=ordering)

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("P(Choose A)")
    ax.set_title("E1: Order Counterbalancing\n(+coef on first, -coef on second)")
    ax.legend()
    ax.set_ylim(0.2, 0.8)

    # Stats for both orderings
    for ordering in ["original", "swapped"]:
        sub = [r for r in valid if r["ordering"] == ordering]
        coefs = [r["coefficient"] for r in sub]
        choices = [1.0 if r["choice"] == "a" else 0.0 for r in sub]
        slope, _, _, p_val, _ = stats.linregress(coefs, choices)
        print(f"E1 {ordering}: slope={slope:.2e}, p={p_val:.6f}")

    # Plot 2: P(choose original task A) by coefficient — remap swapped to original frame
    # In swapped ordering: choosing "a" means choosing original task B
    ax = axes[1]
    p_orig_a_original = []
    p_orig_a_swapped = []
    for coef in coefficients:
        orig = [r for r in valid if r["ordering"] == "original" and r["coefficient"] == coef]
        swap = [r for r in valid if r["ordering"] == "swapped" and r["coefficient"] == coef]
        if orig:
            p_orig_a_original.append(sum(1 for r in orig if r["choice"] == "a") / len(orig))
        else:
            p_orig_a_original.append(np.nan)
        if swap:
            # In swapped: "b" = original task A
            p_orig_a_swapped.append(sum(1 for r in swap if r["choice"] == "b") / len(swap))
        else:
            p_orig_a_swapped.append(np.nan)

    ax.plot(coefficients, p_orig_a_original, "o-", color="blue", markersize=8, linewidth=2, label="Original order")
    ax.plot(coefficients, p_orig_a_swapped, "s-", color="red", markersize=8, linewidth=2, label="Swapped order")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("P(Choose original task A)")
    ax.set_title("E1: Remapped to Original Task\n(evaluative = same slope, positional = opposite)")
    ax.legend()
    ax.set_ylim(0.2, 0.8)

    # Plot 3: Per-pair comparison
    ax = axes[2]
    pairs = sorted(set(r["pair_idx"] for r in valid))
    for pair_idx in pairs:
        orig_slopes = []
        swap_slopes = []
        for ordering in ["original", "swapped"]:
            sub = [r for r in valid if r["pair_idx"] == pair_idx and r["ordering"] == ordering]
            if len(sub) > 10:
                coefs = [r["coefficient"] for r in sub if r["choice"] is not None]
                choices = [1.0 if r["choice"] == "a" else 0.0 for r in sub if r["choice"] is not None]
                if len(set(coefs)) > 1:
                    slope, _, _, _, _ = stats.linregress(coefs, choices)
                    if ordering == "original":
                        orig_slopes.append(slope)
                    else:
                        swap_slopes.append(slope)
        if orig_slopes and swap_slopes:
            ax.scatter(orig_slopes[0], swap_slopes[0], s=60, alpha=0.7)

    lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]), abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.1
    if lim > 0:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.3)
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="Position artifact (same slope)")
    ax.plot([-lim, lim], [lim, -lim], "g--", alpha=0.3, label="Evaluative (reversed slope)")
    ax.set_xlabel("Slope (original order)")
    ax.set_ylabel("Slope (swapped order)")
    ax.set_title("E1: Per-pair Slopes")
    ax.legend(fontsize=8)

    plt.suptitle("E1: Order Counterbalancing — Borderline Pairs", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"plot_{DATE_STR}_e1_order_counterbalance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    shutil.copy(plot_path, ASSETS_DIR / plot_path.name)
    print(f"Saved {plot_path}")


def analyze_e3():
    """E3: Same-task pairs — does steering shift P(A) when both tasks are identical?"""
    path = OUTPUT_DIR / "e3_same_task_results.json"
    if not path.exists():
        print("E3 results not found")
        return

    with open(path) as f:
        results = json.load(f)

    valid = [r for r in results if r["choice"] is not None]
    coefficients = sorted(set(r["coefficient"] for r in valid))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: P(A) by coefficient
    ax = axes[0]
    p_a_vals = []
    n_vals = []
    for coef in coefficients:
        matching = [r for r in valid if r["coefficient"] == coef]
        if matching:
            n_a = sum(1 for r in matching if r["choice"] == "a")
            p_a_vals.append(n_a / len(matching))
            n_vals.append(len(matching))
        else:
            p_a_vals.append(np.nan)
            n_vals.append(0)

    ax.plot(coefficients, p_a_vals, "o-", color="purple", markersize=8, linewidth=2)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="No bias (expected if evaluative)")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("P(Choose A)")
    ax.set_title("E3: Same-Task Pairs\n(identical content, different position)")
    ax.legend()
    ax.set_ylim(0.2, 0.8)

    # Stats
    coefs_list = [r["coefficient"] for r in valid]
    choices = [1.0 if r["choice"] == "a" else 0.0 for r in valid]
    slope, intercept, _, p_val, _ = stats.linregress(coefs_list, choices)
    ax.text(0.97, 0.03, f"slope={slope:.2e}\np={p_val:.4f}\nintercept={intercept:.3f}",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})
    print(f"E3: slope={slope:.2e}, p={p_val:.6f}, intercept={intercept:.3f}")

    # Plot 2: Per-task P(A) at coef=0
    ax = axes[1]
    tasks = sorted(set(r["task_idx"] for r in valid))
    baseline_pa = []
    for task_idx in tasks:
        matching = [r for r in valid if r["task_idx"] == task_idx and r["coefficient"] == 0.0]
        if matching:
            n_a = sum(1 for r in matching if r["choice"] == "a")
            baseline_pa.append(n_a / len(matching))

    ax.hist(baseline_pa, bins=10, range=(0, 1), color="purple", alpha=0.7, edgecolor="black")
    ax.axvline(0.5, color="red", linestyle="--", label="No position bias")
    ax.set_xlabel("P(A) at coef=0")
    ax.set_ylabel("Count")
    ax.set_title("E3: Baseline Position Bias\n(per-task P(A) with no steering)")
    ax.legend()

    plt.suptitle("E3: Same-Task Pairs — Position Artifact Test", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"plot_{DATE_STR}_e3_same_task.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    shutil.copy(plot_path, ASSETS_DIR / plot_path.name)
    print(f"Saved {plot_path}")


def analyze_e8():
    """E8: Random direction control on borderline pairs."""
    path = OUTPUT_DIR / "e8_random_control_results.json"
    if not path.exists():
        print("E8 results not found")
        return

    with open(path) as f:
        results = json.load(f)

    # Also load E1 results for probe direction comparison
    e1_path = OUTPUT_DIR / "e1_order_counterbalance_results.json"
    probe_delta = None
    if e1_path.exists():
        with open(e1_path) as f:
            e1_results = json.load(f)
        orig_valid = [r for r in e1_results if r["ordering"] == "original" and r["choice"] is not None]
        neg = [r for r in orig_valid if r["coefficient"] == -3000.0]
        pos = [r for r in orig_valid if r["coefficient"] == 3000.0]
        if neg and pos:
            p_neg = sum(1 for r in neg if r["choice"] == "a") / len(neg)
            p_pos = sum(1 for r in pos if r["choice"] == "a") / len(pos)
            probe_delta = p_pos - p_neg

    valid = [r for r in results if r["choice"] is not None]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    deltas = []
    for dir_idx in range(max(r["dir_idx"] for r in valid) + 1):
        neg = [r for r in valid if r["dir_idx"] == dir_idx and r["coefficient"] == -3000.0]
        pos = [r for r in valid if r["dir_idx"] == dir_idx and r["coefficient"] == 3000.0]
        if neg and pos:
            p_neg = sum(1 for r in neg if r["choice"] == "a") / len(neg)
            p_pos = sum(1 for r in pos if r["choice"] == "a") / len(pos)
            delta = p_pos - p_neg
            deltas.append(delta)

    ax.bar(range(len(deltas)), deltas, color="gray", alpha=0.7, label="Random directions")
    if probe_delta is not None:
        ax.axhline(probe_delta, color="red", linewidth=2, linestyle="--", label=f"Probe direction (Δ={probe_delta:+.3f})")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Random Direction Index")
    ax.set_ylabel("ΔP(A) (coef +3000 minus -3000)")
    ax.set_title("E8: Random Control on Borderline Pairs\n(probe direction vs random orthogonal directions)")
    ax.legend()

    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    ax.text(0.97, 0.97, f"Random: mean={mean_delta:+.3f}, std={std_delta:.3f}\nProbe: Δ={probe_delta:+.3f}" if probe_delta else "",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})

    print(f"E8: Random deltas: {[f'{d:+.3f}' for d in deltas]}")
    print(f"E8: mean={mean_delta:+.3f}, std={std_delta:.3f}")
    if probe_delta is not None:
        print(f"E8: Probe delta={probe_delta:+.3f}")
        if std_delta > 0:
            z = (probe_delta - mean_delta) / std_delta
            print(f"E8: Probe z-score vs random: {z:.2f}")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"plot_{DATE_STR}_e8_random_control.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    shutil.copy(plot_path, ASSETS_DIR / plot_path.name)
    print(f"Saved {plot_path}")


def main():
    print("=" * 60)
    print("TIER 1 ANALYSIS")
    print("=" * 60)

    analyze_e1()
    print()
    analyze_e3()
    print()
    analyze_e8()


if __name__ == "__main__":
    main()
