"""Summary plot for all confounder experiments."""

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


def main():
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Panel 1: E1 — Order counterbalancing (raw P(A))
    ax = axes[0, 0]
    with open(OUTPUT_DIR / "e1_order_counterbalance_results.json") as f:
        e1 = json.load(f)
    e1_valid = [r for r in e1 if r["choice"] is not None]
    coefficients = sorted(set(r["coefficient"] for r in e1_valid))

    for ordering, color, marker, label in [
        ("original", "blue", "o", "Original (A,B)"),
        ("swapped", "red", "s", "Swapped (B,A)"),
    ]:
        p_a = []
        for coef in coefficients:
            matching = [r for r in e1_valid if r["ordering"] == ordering and r["coefficient"] == coef]
            n_a = sum(1 for r in matching if r["choice"] == "a") if matching else 0
            p_a.append(n_a / len(matching) if matching else np.nan)
        ax.plot(coefficients, p_a, f"{marker}-", color=color, markersize=6, linewidth=2, label=label)

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("P(Choose A)")
    ax.set_title("E1: Order Counterbalancing\n(borderline pairs)")
    ax.legend(fontsize=9)
    ax.set_ylim(0.1, 0.95)

    # Panel 2: E3 — Same-task pairs
    ax = axes[0, 1]
    with open(OUTPUT_DIR / "e3_same_task_results.json") as f:
        e3 = json.load(f)
    e3_valid = [r for r in e3 if r["choice"] is not None]

    p_a = []
    for coef in coefficients:
        matching = [r for r in e3_valid if r["coefficient"] == coef]
        n_a = sum(1 for r in matching if r["choice"] == "a")
        p_a.append(n_a / len(matching) if matching else np.nan)
    ax.plot(coefficients, p_a, "o-", color="purple", markersize=6, linewidth=2)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="No bias")
    ax.fill_between(coefficients, 0.5, p_a, alpha=0.2, color="purple")
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("P(Choose A)")
    ax.set_title("E3: Same-Task Pairs\n(position confound test)")
    ax.set_ylim(0.4, 0.9)

    slope, _, _, p_val, _ = stats.linregress(
        [r["coefficient"] for r in e3_valid],
        [1.0 if r["choice"] == "a" else 0.0 for r in e3_valid],
    )
    ax.text(0.03, 0.97, f"slope={slope:.2e}\np={p_val:.4f}\nΔ={p_a[-1]-p_a[0]:+.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})

    # Panel 3: E8 — Random direction comparison
    ax = axes[0, 2]
    with open(OUTPUT_DIR / "e8_extended_results.json") as f:
        e8 = json.load(f)
    e8_valid = [r for r in e8 if r["choice"] is not None]
    directions = sorted(set(r["direction"] for r in e8_valid))

    deltas = {}
    for d in directions:
        neg = [r for r in e8_valid if r["direction"] == d and r["coefficient"] == -3000.0]
        pos = [r for r in e8_valid if r["direction"] == d and r["coefficient"] == 3000.0]
        if neg and pos:
            p_neg = sum(1 for r in neg if r["choice"] == "a") / len(neg)
            p_pos = sum(1 for r in pos if r["choice"] == "a") / len(pos)
            deltas[d] = p_pos - p_neg

    random_abs = sorted([abs(deltas[d]) for d in deltas if d != "probe"])
    ax.hist(random_abs, bins=10, color="steelblue", alpha=0.7, edgecolor="black", label="Random directions")
    ax.axvline(abs(deltas["probe"]), color="red", linewidth=2, linestyle="--",
               label=f"Probe |Δ|={abs(deltas['probe']):.3f}")
    ax.set_xlabel("|ΔP(A)|")
    ax.set_ylabel("Count")
    ax.set_title("E8: Specificity Test\n(probe vs 20 random directions)")
    ax.legend(fontsize=9)

    # Panel 4: E5 — Logit lens
    ax = axes[1, 0]
    with open(OUTPUT_DIR / "e5_logit_lens_results.json") as f:
        e5 = json.load(f)

    for pair_type, color, marker in [("borderline", "red", "o"), ("firm", "gray", "s")]:
        means = []
        sems = []
        for coef in coefficients:
            matching = [r for r in e5 if r["pair_type"] == pair_type and r["coefficient"] == coef]
            diffs = [r["logit_diff_a_minus_b"] for r in matching]
            means.append(np.mean(diffs))
            sems.append(np.std(diffs) / np.sqrt(len(diffs)))
        ax.errorbar(coefficients, means, yerr=sems, fmt=f"{marker}-", color=color,
                    markersize=6, linewidth=2, capsize=3,
                    label=f"{pair_type} (n={len(set(r['pair_idx'] for r in e5 if r['pair_type']==pair_type))})")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Logit(a) - Logit(b)")
    ax.set_title("E5: Logit Lens\n(continuous measure)")
    ax.legend(fontsize=9)

    # Panel 5: Effect decomposition summary
    ax = axes[1, 1]
    categories = ["H2 original\n(60 pairs, all)", "E1 borderline\n(12 pairs)", "E3 same-task\n(position only)", "E8 random\n(mean |Δ|)"]
    values = [0.083, 0.711, 0.106, 0.286]
    colors_bar = ["green", "blue", "purple", "steelblue"]
    bars = ax.bar(range(len(categories)), values, color=colors_bar, alpha=0.7, edgecolor="black")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("|ΔP(A)| (coef -3k to +3k)")
    ax.set_title("Effect Size Comparison")

    for i, (cat, val) in enumerate(zip(categories, values)):
        ax.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Panel 6: Key conclusions
    ax = axes[1, 2]
    ax.axis("off")
    conclusions = """Key Findings:

1. POSITION CONFOUND CONFIRMED (E3)
   Same-task Δ = +0.106, p = 0.002
   Baseline bias: P(A) = 0.749

2. PROBE IS PARTIALLY SPECIFIC (E8)
   Probe |Δ| = 0.742 > all 20 randoms
   z = 2.73, p = 0.003
   But randoms also shift (mean |Δ| = 0.286)

3. BORDERLINE ENRICHMENT WORKS (E2)
   Effect 8.5x larger on borderline pairs
   12/250 pairs (4.8%) are borderline

4. LOGIT LENS CONFIRMS (E5)
   Borderline: r = 0.779, p < 1e-6
   Firm: r = 0.057, p = 0.500

Bottom line: The H2 effect is real but
confounded. ~15% positional, ~39% non-specific
perturbation, ~46% probe-specific."""

    ax.text(0.05, 0.95, conclusions, transform=ax.transAxes, fontsize=9,
            va="top", ha="left", fontfamily="monospace",
            bbox={"facecolor": "#f9f9f9", "edgecolor": "gray", "alpha": 0.9})

    plt.suptitle("H2 Differential Steering: Confounder Analysis Summary",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"plot_{DATE_STR}_confounders_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    shutil.copy(plot_path, ASSETS_DIR / plot_path.name)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
