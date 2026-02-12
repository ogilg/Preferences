"""Analyze semantically scored open-ended steering results."""

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


OUTPUT_DIR = Path("experiments/steering/open_ended")
ASSETS_DIR = Path("docs/logs/assets/steering")


def main():
    path = OUTPUT_DIR / "scored_steering_results.json"
    if not path.exists():
        print(f"No results at {path}")
        return

    with open(path) as f:
        results = json.load(f)

    date_str = datetime.now().strftime("%m%d%y")
    categories = sorted(set(r["category"] for r in results))
    coefficients = sorted(set(r["coefficient"] for r in results))

    # === Valence by coefficient and category ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, cat in zip(axes, categories):
        means = []
        sems = []
        for coef in coefficients:
            matching = [r for r in results
                       if r["coefficient"] == coef and r["category"] == cat
                       and r.get("valence_score") is not None]
            vals = [r["valence_score"] for r in matching]
            means.append(np.mean(vals) if vals else 0)
            sems.append(stats.sem(vals) if len(vals) > 1 else 0)

        ax.errorbar(coefficients, means, yerr=sems, fmt="o-", capsize=4, linewidth=2, markersize=8)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Steering Coefficient", fontsize=11)
        ax.set_ylabel("Mean Valence Score", fontsize=11)
        ax.set_title(f"Category: {cat}", fontsize=12)
        ax.set_ylim(-0.5, 1.0)

        # Regression
        all_coefs = []
        all_vals = []
        for r in results:
            if r["category"] == cat and r.get("valence_score") is not None:
                all_coefs.append(r["coefficient"])
                all_vals.append(r["valence_score"])
        if len(all_coefs) > 2:
            slope, _, _, p_val, _ = stats.linregress(all_coefs, all_vals)
            ax.text(0.97, 0.03, f"slope={slope:.2e}\np={p_val:.4f}",
                    transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})

    plt.suptitle("Open-ended Steering: Semantic Valence by Category", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"plot_{date_str}_valence_by_category.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(plot_path, ASSETS_DIR / plot_path.name)

    # === Coherence by coefficient ===
    fig, ax = plt.subplots(figsize=(8, 5))
    means = []
    sems = []
    for coef in coefficients:
        matching = [r for r in results
                   if r["coefficient"] == coef and r.get("coherence_score") is not None]
        vals = [r["coherence_score"] for r in matching]
        means.append(np.mean(vals) if vals else 0)
        sems.append(stats.sem(vals) if len(vals) > 1 else 0)

    ax.errorbar(coefficients, means, yerr=sems, fmt="s-", capsize=4, linewidth=2, markersize=8, color="darkred")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Coefficient", fontsize=12)
    ax.set_ylabel("Mean Coherence Score", fontsize=12)
    ax.set_title("Coherence vs Steering Coefficient\n(degeneration check)", fontsize=13)
    ax.set_ylim(0.5, 1.05)

    plot_path2 = OUTPUT_DIR / f"plot_{date_str}_coherence_check.png"
    plt.savefig(plot_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path2}")
    shutil.copy(plot_path2, ASSETS_DIR / plot_path2.name)

    # === Math attitude for preference-eliciting prompts ===
    fig, ax = plt.subplots(figsize=(8, 5))
    pref_results = [r for r in results if r["category"] == "preference_eliciting"]

    means = []
    sems = []
    for coef in coefficients:
        matching = [r for r in pref_results
                   if r["coefficient"] == coef and r.get("math_attitude_score") is not None]
        vals = [r["math_attitude_score"] for r in matching]
        means.append(np.mean(vals) if vals else 0)
        sems.append(stats.sem(vals) if len(vals) > 1 else 0)

    ax.errorbar(coefficients, means, yerr=sems, fmt="D-", capsize=4, linewidth=2, markersize=8, color="darkgreen")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Coefficient", fontsize=12)
    ax.set_ylabel("Mean Math Attitude Score", fontsize=12)
    ax.set_title("Math Attitude vs Steering (preference-eliciting prompts)", fontsize=13)
    ax.set_ylim(-0.5, 1.0)

    # Regression
    all_coefs = [r["coefficient"] for r in pref_results if r.get("math_attitude_score") is not None]
    all_vals = [r["math_attitude_score"] for r in pref_results if r.get("math_attitude_score") is not None]
    if len(all_coefs) > 2:
        slope, _, _, p_val, _ = stats.linregress(all_coefs, all_vals)
        ax.text(0.97, 0.03, f"slope={slope:.2e}\np={p_val:.4f}",
                transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})

    plot_path3 = OUTPUT_DIR / f"plot_{date_str}_math_attitude.png"
    plt.savefig(plot_path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path3}")
    shutil.copy(plot_path3, ASSETS_DIR / plot_path3.name)

    # === Print numeric tables ===
    print(f"\n{'='*80}")
    print("Valence by Coefficient × Category")
    print(f"{'='*80}")
    header = f"{'Coef':>8}"
    for cat in categories:
        header += f"  {cat:>20}"
    print(header)

    for coef in coefficients:
        row = f"{coef:>8.0f}"
        for cat in categories:
            matching = [r for r in results
                       if r["coefficient"] == coef and r["category"] == cat
                       and r.get("valence_score") is not None]
            if matching:
                mean_v = np.mean([r["valence_score"] for r in matching])
                row += f"  {mean_v:>20.3f}"
            else:
                row += f"  {'N/A':>20}"
        print(row)


if __name__ == "__main__":
    main()
