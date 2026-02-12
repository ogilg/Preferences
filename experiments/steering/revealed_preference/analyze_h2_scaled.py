"""Analyze H2 scaled differential steering results."""

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


OUTPUT_DIR = Path("experiments/steering/revealed_preference")
ASSETS_DIR = Path("docs/logs/assets/steering")


def main():
    path = OUTPUT_DIR / "revealed_preference_h2_scaled_results.json"
    if not path.exists():
        print(f"No results at {path}")
        return

    with open(path) as f:
        results = json.load(f)

    date_str = datetime.now().strftime("%m%d%y")
    valid = [r for r in results if r["choice"] is not None]
    coefficients = sorted(set(r["coefficient"] for r in valid))

    # P(A) by coefficient
    p_a_vals = []
    n_vals = []
    for coef in coefficients:
        matching = [r for r in valid if r["coefficient"] == coef]
        n_a = sum(1 for r in matching if r["choice"] == "a")
        p_a_vals.append(n_a / len(matching))
        n_vals.append(len(matching))

    # Also load original H2 for comparison
    orig_path = OUTPUT_DIR / "revealed_preference_h2_results.json"
    orig_p_a = []
    orig_coefs = []
    if orig_path.exists():
        with open(orig_path) as f:
            orig_results = json.load(f)
        orig_valid = [r for r in orig_results if r["choice"] is not None]
        orig_coefs = sorted(set(r["coefficient"] for r in orig_valid))
        for coef in orig_coefs:
            matching = [r for r in orig_valid if r["coefficient"] == coef]
            n_a = sum(1 for r in matching if r["choice"] == "a")
            orig_p_a.append(n_a / len(matching))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: P(A) by coefficient (both original and scaled)
    ax = axes[0]
    ax.plot(coefficients, p_a_vals, "o-", color="darkgreen", markersize=8, linewidth=2,
            label=f"Scaled (N=60, n={n_vals[0]}/coef)")
    if orig_p_a:
        ax.plot(orig_coefs, orig_p_a, "s--", color="gray", markersize=6, linewidth=1.5,
                alpha=0.7, label="Original (N=20, n=200/coef)")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Coefficient", fontsize=12)
    ax.set_ylabel("P(Choose A)", fontsize=12)
    ax.set_title("H2 Differential: Original vs Scaled", fontsize=13)
    ax.set_ylim(0.35, 0.75)
    ax.legend(fontsize=10)

    # Stats
    all_coefs = [r["coefficient"] for r in valid]
    all_choices = [1.0 if r["choice"] == "a" else 0.0 for r in valid]
    slope, _, r_val, p_val, _ = stats.linregress(all_coefs, all_choices)
    ax.text(0.97, 0.03, f"slope={slope:.2e}\np={p_val:.6f}\nΔP(A)={p_a_vals[-1]-p_a_vals[0]:+.3f}",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})

    # Plot 2: Per-pair P(A) at min vs max coefficient
    ax = axes[1]
    pairs = sorted(set(r["pair_idx"] for r in valid))
    min_coef = min(coefficients)
    max_coef = max(coefficients)

    pair_pa_neg = []
    pair_pa_pos = []
    for pair in pairs:
        neg = [r for r in valid if r["pair_idx"] == pair and r["coefficient"] == min_coef]
        pos = [r for r in valid if r["pair_idx"] == pair and r["coefficient"] == max_coef]
        if neg and pos:
            pair_pa_neg.append(sum(1 for r in neg if r["choice"] == "a") / len(neg))
            pair_pa_pos.append(sum(1 for r in pos if r["choice"] == "a") / len(pos))

    ax.scatter(pair_pa_neg, pair_pa_pos, alpha=0.6, s=50)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="No effect line")
    ax.set_xlabel(f"P(A) at coef={min_coef:.0f}", fontsize=12)
    ax.set_ylabel(f"P(A) at coef={max_coef:.0f}", fontsize=12)
    ax.set_title("Per-pair: Effect of Steering", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    # Count pairs above/below diagonal
    n_above = sum(1 for n, p in zip(pair_pa_neg, pair_pa_pos) if p > n)
    n_below = sum(1 for n, p in zip(pair_pa_neg, pair_pa_pos) if p < n)
    n_equal = sum(1 for n, p in zip(pair_pa_neg, pair_pa_pos) if p == n)
    ax.text(0.03, 0.97, f"Above: {n_above}\nBelow: {n_below}\nEqual: {n_equal}",
            transform=ax.transAxes, fontsize=10, va="top", ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})

    plt.suptitle("H2 Differential Steering — Scaled (60 pairs, 15 resamples)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"plot_{date_str}_h2_scaled.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(plot_path, ASSETS_DIR / plot_path.name)

    # Summary
    print(f"\nH2 Scaled Results:")
    print(f"  Pairs: {len(pairs)}")
    print(f"  N per coefficient: {n_vals[0]}")
    print(f"  P(A) range: {p_a_vals[0]:.3f} → {p_a_vals[-1]:.3f} (Δ={p_a_vals[-1]-p_a_vals[0]:+.3f})")
    print(f"  Regression: slope={slope:.2e}, p={p_val:.6f}")
    print(f"  Pairs above diagonal: {n_above}/{len(pairs)}")
    print(f"  Most pairs firmly decided — effect driven by borderline pairs")


if __name__ == "__main__":
    main()
