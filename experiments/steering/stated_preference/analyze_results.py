"""Analyze Phase 2 stated preference dose-response results."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


OUTPUT_DIR = Path("experiments/steering/stated_preference")
ASSETS_DIR = Path("docs/logs/assets/steering")


def main():
    with open(OUTPUT_DIR / "steering_results.json") as f:
        data = json.load(f)

    results = data["results"]
    coefficients = sorted(set(r["coefficient"] for r in results))

    # Aggregate by coefficient
    by_coef: dict[float, list[float]] = {}
    for r in results:
        coef = r["coefficient"]
        parsed = r["parsed_value"]
        if isinstance(parsed, (int, float)):
            if coef not in by_coef:
                by_coef[coef] = []
            by_coef[coef].append(parsed)

    # Per-coefficient statistics
    print("=" * 70)
    print(f"{'Coefficient':>12} {'Mean':>8} {'SEM':>8} {'P(good)':>8} {'N':>5}")
    print("-" * 70)
    for coef in coefficients:
        scores = by_coef[coef]
        mean = np.mean(scores)
        sem = np.std(scores) / np.sqrt(len(scores))
        p_good = sum(1 for s in scores if s > 0) / len(scores)
        print(f"{coef:>12.0f} {mean:>+8.3f} {sem:>8.3f} {p_good:>8.1%} {len(scores):>5}")

    # Effect sizes
    neg_scores = by_coef[min(coefficients)]
    pos_scores = by_coef[max(coefficients)]
    pooled_std = np.sqrt((np.var(neg_scores) + np.var(pos_scores)) / 2)
    cohens_d = (np.mean(pos_scores) - np.mean(neg_scores)) / pooled_std if pooled_std > 0 else 0

    # Linear regression
    all_coefs = []
    all_scores = []
    for c, scores in by_coef.items():
        all_coefs.extend([c] * len(scores))
        all_scores.extend(scores)

    slope, intercept, r_value, p_slope, std_err = stats.linregress(all_coefs, all_scores)

    print(f"\nEffect size (Cohen's d, min vs max): {cohens_d:.3f}")
    print(f"Regression: slope={slope:.6f}, R²={r_value**2:.4f}, p={p_slope:.6f}")

    # Per-task breakdown: which tasks flip?
    tasks = sorted(set(r["task_id"] for r in results))
    print(f"\n{'Task':>30} {'coef=-3k':>9} {'coef=0':>9} {'coef=+3k':>9} {'Flips?':>8}")
    print("-" * 70)
    for task_id in tasks:
        for target_coef in [min(coefficients), 0.0, max(coefficients)]:
            task_coef_scores = [
                r["parsed_value"] for r in results
                if r["task_id"] == task_id and r["coefficient"] == target_coef
                and isinstance(r["parsed_value"], (int, float))
            ]
            mean_score = np.mean(task_coef_scores) if task_coef_scores else float("nan")
            if target_coef == min(coefficients):
                neg_mean = mean_score
            elif target_coef == 0:
                zero_mean = mean_score
            else:
                pos_mean = mean_score
        flips = "YES" if (neg_mean < 0 and pos_mean > 0) or (neg_mean > 0 and pos_mean < 0) else ""
        if neg_mean != pos_mean:
            flips = flips or "shift"
        print(f"{task_id[:30]:>30} {neg_mean:>+9.2f} {zero_mean:>+9.2f} {pos_mean:>+9.2f} {flips:>8}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Mean valence with CI
    means = [np.mean(by_coef[c]) for c in coefficients]
    sems = [np.std(by_coef[c]) / np.sqrt(len(by_coef[c])) for c in coefficients]

    ax1.errorbar(coefficients, means, yerr=[1.96 * s for s in sems],
                 fmt="o-", color="darkblue", markersize=8, linewidth=2, capsize=5,
                 label="Mean ± 95% CI")
    x_range = np.linspace(min(coefficients), max(coefficients), 100)
    y_pred = slope * x_range + intercept
    ax1.plot(x_range, y_pred, "--", color="red", alpha=0.7,
             label=f"Linear fit (R²={r_value**2:.3f})")
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Steering Coefficient", fontsize=12)
    ax1.set_ylabel("Mean Stated Valence (good=+1, bad=-1)", fontsize=12)
    ax1.set_title("Dose-Response: Steering → Stated Preference", fontsize=13)
    ax1.legend(loc="upper left")
    stats_text = f"Cohen's d: {cohens_d:.2f}\nSlope: {slope:.2e}\np: {p_slope:.4f}"
    ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes, fontsize=10,
             va="bottom", ha="right",
             bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})

    # Plot 2: P(good) with logistic-like curve
    p_goods = [sum(1 for s in by_coef[c] if s > 0) / len(by_coef[c]) for c in coefficients]
    ax2.plot(coefficients, p_goods, "o-", color="darkgreen", markersize=8, linewidth=2)
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax2.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Steering Coefficient", fontsize=12)
    ax2.set_ylabel("P(good)", fontsize=12)
    ax2.set_title("P(good) vs Steering Coefficient", fontsize=13)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    date_str = datetime.now().strftime("%m%d%y")
    plot_path = OUTPUT_DIR / f"plot_{date_str}_dose_response.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot to {plot_path}")

    # Copy to assets
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    assets_plot_path = ASSETS_DIR / f"plot_{date_str}_dose_response.png"
    import shutil
    shutil.copy(plot_path, assets_plot_path)
    print(f"Copied to {assets_plot_path}")


if __name__ == "__main__":
    main()
