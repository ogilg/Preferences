"""Analyze E8 extended: probe direction vs 20 random orthogonal directions on borderline pairs."""

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
    path = OUTPUT_DIR / "e8_extended_results.json"
    if not path.exists():
        print(f"No results at {path}")
        return

    with open(path) as f:
        results = json.load(f)

    valid = [r for r in results if r["choice"] is not None]
    directions = sorted(set(r["direction"] for r in valid))

    # Compute ΔP(A) for each direction
    deltas = {}
    for dir_name in directions:
        neg = [r for r in valid if r["direction"] == dir_name and r["coefficient"] == -3000.0]
        pos = [r for r in valid if r["direction"] == dir_name and r["coefficient"] == 3000.0]
        if neg and pos:
            p_neg = sum(1 for r in neg if r["choice"] == "a") / len(neg)
            p_pos = sum(1 for r in pos if r["choice"] == "a") / len(pos)
            deltas[dir_name] = p_pos - p_neg

    probe_delta = deltas.get("probe", 0)
    random_deltas = [deltas[d] for d in sorted(deltas) if d.startswith("random_")]
    random_abs_deltas = [abs(d) for d in random_deltas]

    print(f"Probe ΔP(A): {probe_delta:+.3f}")
    print(f"Random ΔP(A): {[f'{d:+.3f}' for d in random_deltas]}")
    print(f"Random mean: {np.mean(random_deltas):+.3f}, std: {np.std(random_deltas):.3f}")
    print(f"Random |Δ| mean: {np.mean(random_abs_deltas):.3f}, std: {np.std(random_abs_deltas):.3f}")

    # Signed comparison
    z_signed = (probe_delta - np.mean(random_deltas)) / np.std(random_deltas) if np.std(random_deltas) > 0 else 0
    p_signed = 1 - stats.norm.cdf(z_signed)

    # Absolute comparison
    z_abs = (abs(probe_delta) - np.mean(random_abs_deltas)) / np.std(random_abs_deltas) if np.std(random_abs_deltas) > 0 else 0
    p_abs = 1 - stats.norm.cdf(z_abs)

    # Rank-based: what fraction of randoms have larger |Δ|?
    n_larger = sum(1 for d in random_abs_deltas if d >= abs(probe_delta))
    rank_p = (n_larger + 1) / (len(random_abs_deltas) + 1)  # includes probe itself

    print(f"\nSigned z-score: {z_signed:.2f} (p={p_signed:.4f}, one-tailed)")
    print(f"|Δ| z-score: {z_abs:.2f} (p={p_abs:.4f}, one-tailed)")
    print(f"Rank p-value: {rank_p:.4f} ({n_larger}/{len(random_abs_deltas)} randoms have |Δ| >= probe)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: All deltas sorted
    ax = axes[0]
    all_names = sorted(deltas.keys(), key=lambda d: deltas[d])
    colors = ["red" if d == "probe" else "steelblue" for d in all_names]
    bars = ax.barh(range(len(all_names)), [deltas[d] for d in all_names], color=colors, alpha=0.7)
    ax.set_yticks(range(len(all_names)))
    ax.set_yticklabels([d.replace("random_", "R") for d in all_names], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("ΔP(A) (coef +3000 minus -3000)")
    ax.set_title("All Directions Sorted by Effect Size")

    # Plot 2: Histogram of |Δ| with probe marked
    ax = axes[1]
    ax.hist(random_abs_deltas, bins=12, color="steelblue", alpha=0.7, edgecolor="black", label="Random directions")
    ax.axvline(abs(probe_delta), color="red", linewidth=2, linestyle="--", label=f"Probe |Δ|={abs(probe_delta):.3f}")
    ax.set_xlabel("|ΔP(A)|")
    ax.set_ylabel("Count")
    ax.set_title(f"|Δ| Distribution\n(rank p={rank_p:.3f}, z={z_abs:.2f})")
    ax.legend()

    # Plot 3: P(A) at each coef for probe vs random mean ± std
    ax = axes[2]
    test_coefs = sorted(set(r["coefficient"] for r in valid))

    probe_pa = []
    for coef in test_coefs:
        matching = [r for r in valid if r["direction"] == "probe" and r["coefficient"] == coef]
        if matching:
            probe_pa.append(sum(1 for r in matching if r["choice"] == "a") / len(matching))

    random_pa_by_coef = []
    random_pa_std = []
    for coef in test_coefs:
        per_dir = []
        for d in sorted(deltas):
            if d.startswith("random_"):
                matching = [r for r in valid if r["direction"] == d and r["coefficient"] == coef]
                if matching:
                    per_dir.append(sum(1 for r in matching if r["choice"] == "a") / len(matching))
        random_pa_by_coef.append(np.mean(per_dir))
        random_pa_std.append(np.std(per_dir))

    ax.plot(test_coefs, probe_pa, "ro-", markersize=8, linewidth=2, label="Probe direction")
    ax.errorbar(test_coefs, random_pa_by_coef, yerr=random_pa_std, fmt="s-", color="steelblue",
                markersize=6, linewidth=1.5, capsize=3, label=f"Random (n={len(random_deltas)}, mean ± std)")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("P(Choose A)")
    ax.set_title("Probe vs Random: Dose-Response")
    ax.legend()

    plt.suptitle(f"E8 Extended: Probe vs {len(random_deltas)} Random Directions (Borderline Pairs)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"plot_{DATE_STR}_e8_extended.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    shutil.copy(plot_path, ASSETS_DIR / plot_path.name)
    print(f"\nSaved {plot_path}")


if __name__ == "__main__":
    main()
