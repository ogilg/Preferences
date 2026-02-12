"""Comprehensive analysis of all Phase 3 revealed preference results."""

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


OUTPUT_DIR = Path("experiments/steering/revealed_preference")
ASSETS_DIR = Path("docs/logs/assets/steering")


def load_results(filename: str) -> list[dict]:
    path = OUTPUT_DIR / filename
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def compute_p_a(results: list[dict], coef: float, **filters) -> tuple[float, int]:
    """Compute P(A) for a given coefficient and optional filters."""
    matching = [r for r in results if r["coefficient"] == coef and r["choice"] is not None]
    for key, val in filters.items():
        matching = [r for r in matching if r.get(key) == val]
    if not matching:
        return 0.0, 0
    n_a = sum(1 for r in matching if r["choice"] == "a")
    return n_a / len(matching), len(matching)


def main():
    h3_results = load_results("revealed_preference_h3_results.json")
    h1_results = load_results("revealed_preference_h1_results.json")
    h2_results = load_results("revealed_preference_h2_results.json")

    date_str = datetime.now().strftime("%m%d%y")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # === H3 ===
    if h3_results:
        coefficients = sorted(set(r["coefficient"] for r in h3_results))
        p_a_vals = [compute_p_a(h3_results, c)[0] for c in coefficients]

        ax = axes[0]
        ax.plot(coefficients, p_a_vals, "o-", color="darkblue", markersize=8, linewidth=2)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Steering Coefficient", fontsize=11)
        ax.set_ylabel("P(Choose A)", fontsize=11)
        ax.set_title("H3: Autoregressive (last-token)", fontsize=12)
        ax.set_ylim(0.3, 0.7)

        # Regression
        all_coefs = [r["coefficient"] for r in h3_results if r["choice"] is not None]
        all_choices = [1.0 if r["choice"] == "a" else 0.0 for r in h3_results if r["choice"] is not None]
        slope, _, r_val, p_val, _ = stats.linregress(all_coefs, all_choices)
        ax.text(0.97, 0.03, f"slope={slope:.2e}\np={p_val:.4f}",
                transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})

    # === H1 ===
    if h1_results:
        coefficients = sorted(set(r["coefficient"] for r in h1_results))
        pa_steer_a = [compute_p_a(h1_results, c, steer_target="task_a")[0] for c in coefficients]
        pa_steer_b = [compute_p_a(h1_results, c, steer_target="task_b")[0] for c in coefficients]

        ax = axes[1]
        ax.plot(coefficients, pa_steer_a, "o-", color="blue", markersize=8, linewidth=2, label="Steer on A")
        ax.plot(coefficients, pa_steer_b, "s-", color="red", markersize=8, linewidth=2, label="Steer on B")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Steering Coefficient", fontsize=11)
        ax.set_ylabel("P(Choose A)", fontsize=11)
        ax.set_title("H1: Task-selective steering", fontsize=12)
        ax.set_ylim(0.3, 0.7)
        ax.legend(loc="upper left", fontsize=9)

        # Interaction
        steer_a_eff = pa_steer_a[-1] - pa_steer_a[0]
        steer_b_eff = pa_steer_b[-1] - pa_steer_b[0]
        interaction = steer_a_eff - steer_b_eff
        ax.text(0.97, 0.03, f"Steer-A Δ={steer_a_eff:+.3f}\nSteer-B Δ={steer_b_eff:+.3f}\nInteract={interaction:+.3f}",
                transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})

    # === H2 ===
    if h2_results:
        coefficients = sorted(set(r["coefficient"] for r in h2_results))
        p_a_vals = [compute_p_a(h2_results, c)[0] for c in coefficients]

        ax = axes[2]
        ax.plot(coefficients, p_a_vals, "o-", color="darkgreen", markersize=8, linewidth=2)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Steering Coefficient", fontsize=11)
        ax.set_ylabel("P(Choose A)", fontsize=11)
        ax.set_title("H2: Differential (+A, -B)", fontsize=12)
        ax.set_ylim(0.3, 0.7)

        # Regression
        all_coefs = [r["coefficient"] for r in h2_results if r["choice"] is not None]
        all_choices = [1.0 if r["choice"] == "a" else 0.0 for r in h2_results if r["choice"] is not None]
        slope, _, r_val, p_val, _ = stats.linregress(all_coefs, all_choices)

        # Chi-square
        neg = [r for r in h2_results if r["coefficient"] == min(coefficients) and r["choice"] is not None]
        pos = [r for r in h2_results if r["coefficient"] == max(coefficients) and r["choice"] is not None]
        neg_a = sum(1 for r in neg if r["choice"] == "a")
        pos_a = sum(1 for r in pos if r["choice"] == "a")
        table = np.array([[neg_a, len(neg) - neg_a], [pos_a, len(pos) - pos_a]])
        chi2, p_chi2 = stats.chi2_contingency(table)[:2]

        ax.text(0.97, 0.03, f"slope={slope:.2e}\np={p_val:.6f}\nΔP(A)={p_a_vals[-1]-p_a_vals[0]:+.3f}",
                transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})

        print(f"\nH2 Differential Steering Statistics:")
        print(f"  Regression: slope={slope:.2e}, R²={r_val**2:.4f}, p={p_val:.6f}")
        print(f"  Chi² (min vs max): χ²={chi2:.2f}, p={p_chi2:.4f}")
        print(f"  P(A) range: {p_a_vals[0]:.3f} → {p_a_vals[-1]:.3f} (Δ={p_a_vals[-1]-p_a_vals[0]:+.3f})")

    plt.suptitle("Phase 3: Revealed Preference Steering", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"plot_{date_str}_revealed_preference_all.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot to {plot_path}")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    assets_path = ASSETS_DIR / f"plot_{date_str}_revealed_preference_all.png"
    shutil.copy(plot_path, assets_path)
    print(f"Copied to {assets_path}")


if __name__ == "__main__":
    main()
