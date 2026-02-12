"""Analyze E5 logit lens results."""

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
    path = OUTPUT_DIR / "e5_logit_lens_results.json"
    if not path.exists():
        print(f"No results at {path}")
        return

    with open(path) as f:
        results = json.load(f)

    coefficients = sorted(set(r["coefficient"] for r in results))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Mean logit diff by coefficient, borderline vs firm
    ax = axes[0]
    for pair_type, color, marker in [("borderline", "red", "o"), ("firm", "gray", "s")]:
        means = []
        stds = []
        for coef in coefficients:
            matching = [r for r in results if r["pair_type"] == pair_type and r["coefficient"] == coef]
            diffs = [r["logit_diff_a_minus_b"] for r in matching]
            means.append(np.mean(diffs))
            stds.append(np.std(diffs) / np.sqrt(len(diffs)))  # SEM
        ax.errorbar(coefficients, means, yerr=stds, fmt=f"{marker}-", color=color,
                    markersize=8, linewidth=2, capsize=3, label=f"{pair_type} (n={len(set(r['pair_idx'] for r in results if r['pair_type']==pair_type))})")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Logit diff (a - b)")
    ax.set_title("E5: Logit Lens\nmean ± SEM")
    ax.legend()

    # Regression on borderline
    bl = [r for r in results if r["pair_type"] == "borderline"]
    bl_coefs = [r["coefficient"] for r in bl]
    bl_diffs = [r["logit_diff_a_minus_b"] for r in bl]
    slope, intercept, r_val, p_val, stderr = stats.linregress(bl_coefs, bl_diffs)
    print(f"Borderline regression: slope={slope:.2e}, r={r_val:.3f}, p={p_val:.6f}")

    fm = [r for r in results if r["pair_type"] == "firm"]
    fm_coefs = [r["coefficient"] for r in fm]
    fm_diffs = [r["logit_diff_a_minus_b"] for r in fm]
    slope_f, _, r_f, p_f, _ = stats.linregress(fm_coefs, fm_diffs)
    print(f"Firm regression: slope={slope_f:.2e}, r={r_f:.3f}, p={p_f:.6f}")

    ax.text(0.03, 0.97,
            f"Borderline: slope={slope:.2e}, r={r_val:.2f}, p={p_val:.1e}\nFirm: slope={slope_f:.2e}, r={r_f:.2f}, p={p_f:.3f}",
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})

    # Plot 2: Per-pair logit diff trajectories (borderline)
    ax = axes[1]
    bl_pairs = sorted(set(r["pair_idx"] for r in results if r["pair_type"] == "borderline"))
    for pair_idx in bl_pairs:
        pair_data = [r for r in results if r["pair_idx"] == pair_idx]
        pair_coefs = [r["coefficient"] for r in pair_data]
        pair_diffs = [r["logit_diff_a_minus_b"] for r in pair_data]
        ax.plot(pair_coefs, pair_diffs, "o-", alpha=0.5, markersize=4, label=f"Pair {pair_idx}")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Logit diff (a - b)")
    ax.set_title("E5: Per-pair Trajectories (borderline)")
    ax.legend(fontsize=6, ncol=2)

    # Plot 3: Per-pair logit diff trajectories (firm) — just show range
    ax = axes[2]
    fm_pairs = sorted(set(r["pair_idx"] for r in results if r["pair_type"] == "firm"))
    for pair_idx in fm_pairs:
        pair_data = [r for r in results if r["pair_idx"] == pair_idx]
        pair_coefs = [r["coefficient"] for r in pair_data]
        pair_diffs = [r["logit_diff_a_minus_b"] for r in pair_data]
        ax.plot(pair_coefs, pair_diffs, "o-", alpha=0.3, markersize=3, color="gray")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Logit diff (a - b)")
    ax.set_title("E5: Per-pair Trajectories (firm)")

    plt.suptitle("E5: Logit Lens Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"plot_{DATE_STR}_e5_logit_lens.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    shutil.copy(plot_path, ASSETS_DIR / plot_path.name)
    print(f"Saved {plot_path}")

    # Also compute effect sizes
    print(f"\nEffect sizes:")
    print(f"  Borderline: Δ logit_diff = {np.mean([r['logit_diff_a_minus_b'] for r in bl if r['coefficient']==3000]) - np.mean([r['logit_diff_a_minus_b'] for r in bl if r['coefficient']==-3000]):.2f}")
    print(f"  Firm: Δ logit_diff = {np.mean([r['logit_diff_a_minus_b'] for r in fm if r['coefficient']==3000]) - np.mean([r['logit_diff_a_minus_b'] for r in fm if r['coefficient']==-3000]):.2f}")


if __name__ == "__main__":
    main()
