"""Analyze multi-template stated preference dose-response results."""

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


OUTPUT_DIR = Path("experiments/steering/stated_preference")
ASSETS_DIR = Path("docs/logs/assets/steering")


def main():
    path = OUTPUT_DIR / "multi_template_results.json"
    if not path.exists():
        print(f"No results at {path}")
        return

    with open(path) as f:
        results = json.load(f)

    date_str = datetime.now().strftime("%m%d%y")
    templates = sorted(set(r["template"] for r in results))
    coefficients = sorted(set(r["coefficient"] for r in results))

    # Normalize ratings to [0, 1] for comparison across templates
    def normalize(rating, template_name):
        if rating is None:
            return None
        if template_name in ("binary",):
            return (rating + 1) / 2  # -1..1 → 0..1
        if template_name in ("ternary",):
            return (rating + 1) / 2  # -1..1 → 0..1
        if template_name in ("scale_1_5", "anchored_precise_1_5", "fruit_rating", "fruit_qualitative"):
            return (rating - 1) / 4  # 1..5 → 0..1
        return rating

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, template_name in enumerate(templates):
        if i >= len(axes):
            break
        ax = axes[i]
        tmpl_results = [r for r in results if r["template"] == template_name]

        means = []
        sems = []
        ns = []
        for coef in coefficients:
            matching = [r for r in tmpl_results if r["coefficient"] == coef and r["rating"] is not None]
            if matching:
                norm_vals = [normalize(r["rating"], template_name) for r in matching]
                norm_vals = [v for v in norm_vals if v is not None]
                means.append(np.mean(norm_vals) if norm_vals else 0)
                sems.append(stats.sem(norm_vals) if len(norm_vals) > 1 else 0)
                ns.append(len(norm_vals))
            else:
                means.append(0)
                sems.append(0)
                ns.append(0)

        ax.errorbar(coefficients, means, yerr=sems, fmt="o-", capsize=4, linewidth=2, markersize=8)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Steering Coefficient")
        ax.set_ylabel("Normalized Rating [0-1]")
        ax.set_title(template_name, fontsize=12)
        ax.set_ylim(0.2, 0.9)

        # Regression on raw per-trial data
        all_coefs = [r["coefficient"] for r in tmpl_results if r["rating"] is not None]
        all_norm = [normalize(r["rating"], template_name) for r in tmpl_results if r["rating"] is not None]
        all_norm = [v for v in all_norm if v is not None]
        if len(all_coefs) > 2 and len(all_coefs) == len(all_norm):
            slope, _, _, p_val, _ = stats.linregress(all_coefs, all_norm)
            # Parse rate
            n_valid = sum(1 for r in tmpl_results if r["rating"] is not None)
            n_total = len(tmpl_results)
            parse_rate = n_valid / n_total if n_total > 0 else 0
            ax.text(0.97, 0.03, f"slope={slope:.2e}\np={p_val:.4f}\nparse={parse_rate:.0%}",
                    transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})

    # Hide unused axes
    for j in range(len(templates), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Stated Preference Dose-Response by Template", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"plot_{date_str}_multi_template_dose_response.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(plot_path, ASSETS_DIR / plot_path.name)

    # === Summary table ===
    print(f"\n{'='*80}")
    print("Multi-Template Dose-Response Summary")
    print(f"{'='*80}")

    header = f"{'Template':>25} {'Parse%':>7} {'Slope':>12} {'p-value':>10} {'Δ(norm)':>10} {'Min→Max':>15}"
    print(header)
    print("-" * len(header))

    for template_name in templates:
        tmpl_results = [r for r in results if r["template"] == template_name]
        n_valid = sum(1 for r in tmpl_results if r["rating"] is not None)
        n_total = len(tmpl_results)
        parse_rate = n_valid / n_total if n_total > 0 else 0

        all_coefs = [r["coefficient"] for r in tmpl_results if r["rating"] is not None]
        all_norm = [normalize(r["rating"], template_name) for r in tmpl_results if r["rating"] is not None]
        all_norm = [v for v in all_norm if v is not None]

        if len(all_coefs) > 2 and len(all_coefs) == len(all_norm):
            slope, _, _, p_val, _ = stats.linregress(all_coefs, all_norm)

            min_vals = [normalize(r["rating"], template_name) for r in tmpl_results
                       if r["coefficient"] == min(coefficients) and r["rating"] is not None]
            max_vals = [normalize(r["rating"], template_name) for r in tmpl_results
                       if r["coefficient"] == max(coefficients) and r["rating"] is not None]
            min_mean = np.mean(min_vals) if min_vals else 0
            max_mean = np.mean(max_vals) if max_vals else 0

            print(f"{template_name:>25} {parse_rate:>7.0%} {slope:>12.2e} {p_val:>10.4f} {max_mean-min_mean:>+10.3f} {min_mean:.2f}→{max_mean:.2f}")
        else:
            print(f"{template_name:>25} {parse_rate:>7.0%} {'N/A':>12} {'N/A':>10}")


if __name__ == "__main__":
    main()
