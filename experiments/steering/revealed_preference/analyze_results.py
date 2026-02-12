"""Analyze Phase 3 revealed preference steering results."""

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


OUTPUT_DIR = Path("experiments/steering/revealed_preference")
ASSETS_DIR = Path("docs/logs/assets/steering")


def analyze_h3():
    """Analyze H3: autoregressive steering."""
    path = OUTPUT_DIR / "revealed_preference_h3_results.json"
    if not path.exists():
        print("No H3 results found")
        return

    with open(path) as f:
        results = json.load(f)

    coefficients = sorted(set(r["coefficient"] for r in results))
    valid = [r for r in results if r["choice"] is not None]

    print("=" * 60)
    print("H3: Autoregressive Steering — Last-token during choice")
    print("=" * 60)

    p_a_values = []
    n_values = []
    for coef in coefficients:
        matching = [r for r in valid if r["coefficient"] == coef]
        n_a = sum(1 for r in matching if r["choice"] == "a")
        p_a = n_a / len(matching)
        p_a_values.append(p_a)
        n_values.append(len(matching))
        print(f"  coef={coef:+7.0f}: P(A)={p_a:.3f} ({n_a}/{len(matching)})")

    # Regression: P(A) ~ coefficient
    all_coefs = []
    all_choices = []
    for r in valid:
        all_coefs.append(r["coefficient"])
        all_choices.append(1.0 if r["choice"] == "a" else 0.0)

    slope, intercept, r_value, p_value, std_err = stats.linregress(all_coefs, all_choices)
    print(f"\nRegression: slope={slope:.2e}, R²={r_value**2:.4f}, p={p_value:.4f}")

    # Chi-square test: most negative vs most positive
    neg_results = [r for r in valid if r["coefficient"] == min(coefficients)]
    pos_results = [r for r in valid if r["coefficient"] == max(coefficients)]
    neg_a = sum(1 for r in neg_results if r["choice"] == "a")
    pos_a = sum(1 for r in pos_results if r["choice"] == "a")
    n_neg = len(neg_results)
    n_pos = len(pos_results)

    # 2x2 contingency table
    table = np.array([[neg_a, n_neg - neg_a], [pos_a, n_pos - pos_a]])
    chi2, p_chi2 = stats.chi2_contingency(table)[:2]
    print(f"Chi² (min vs max): χ²={chi2:.2f}, p={p_chi2:.4f}")

    return coefficients, p_a_values, slope, r_value**2, p_value


def analyze_h1():
    """Analyze H1: task-selective steering."""
    path = OUTPUT_DIR / "revealed_preference_h1_results.json"
    if not path.exists():
        print("No H1 results found")
        return

    with open(path) as f:
        results = json.load(f)

    valid = [r for r in results if r["choice"] is not None]
    coefficients = sorted(set(r["coefficient"] for r in valid))

    print("\n" + "=" * 60)
    print("H1: Task-Selective Steering — Steer on one task's tokens")
    print("=" * 60)

    h1_data = {}
    for target in ["task_a", "task_b"]:
        print(f"\n  Steered on: {target}")
        p_a_values = []
        for coef in coefficients:
            matching = [r for r in valid if r["coefficient"] == coef and r["steer_target"] == target]
            n_a = sum(1 for r in matching if r["choice"] == "a")
            p_a = n_a / len(matching) if matching else 0
            p_a_values.append(p_a)
            print(f"    coef={coef:+7.0f}: P(A)={p_a:.3f} ({n_a}/{len(matching) if matching else 0})")
        h1_data[target] = p_a_values

    # Key test: steer_on_A at max_coef vs steer_on_B at max_coef
    max_coef = max(coefficients)
    min_coef = min(coefficients)

    steer_a_pos = [r for r in valid if r["steer_target"] == "task_a" and r["coefficient"] == max_coef]
    steer_b_pos = [r for r in valid if r["steer_target"] == "task_b" and r["coefficient"] == max_coef]
    steer_a_neg = [r for r in valid if r["steer_target"] == "task_a" and r["coefficient"] == min_coef]
    steer_b_neg = [r for r in valid if r["steer_target"] == "task_b" and r["coefficient"] == min_coef]

    # Differential effect: does steering on A (pos) vs B (pos) shift P(A)?
    pa_steer_a_pos = sum(1 for r in steer_a_pos if r["choice"] == "a") / len(steer_a_pos)
    pa_steer_b_pos = sum(1 for r in steer_b_pos if r["choice"] == "a") / len(steer_b_pos)
    diff_pos = pa_steer_a_pos - pa_steer_b_pos

    pa_steer_a_neg = sum(1 for r in steer_a_neg if r["choice"] == "a") / len(steer_a_neg)
    pa_steer_b_neg = sum(1 for r in steer_b_neg if r["choice"] == "a") / len(steer_b_neg)
    diff_neg = pa_steer_a_neg - pa_steer_b_neg

    print(f"\n  Differential effect at coef={max_coef:+.0f}: P(A|steer_A) - P(A|steer_B) = {diff_pos:+.3f}")
    print(f"  Differential effect at coef={min_coef:+.0f}: P(A|steer_A) - P(A|steer_B) = {diff_neg:+.3f}")

    # Interaction test: 2x2 (steer_target × sign)
    # Does the effect of positive vs negative steering depend on which task is steered?
    # steer_A_pos vs steer_A_neg vs steer_B_pos vs steer_B_neg
    steer_a_pos_a = sum(1 for r in steer_a_pos if r["choice"] == "a")
    steer_b_pos_a = sum(1 for r in steer_b_pos if r["choice"] == "a")
    steer_a_neg_a = sum(1 for r in steer_a_neg if r["choice"] == "a")
    steer_b_neg_a = sum(1 for r in steer_b_neg if r["choice"] == "a")

    # The interaction: (steer_A_pos - steer_A_neg) vs (steer_B_pos - steer_B_neg)
    # should have opposite signs
    steer_a_effect = steer_a_pos_a / len(steer_a_pos) - steer_a_neg_a / len(steer_a_neg)
    steer_b_effect = steer_b_pos_a / len(steer_b_pos) - steer_b_neg_a / len(steer_b_neg)
    interaction = steer_a_effect - steer_b_effect

    print(f"\n  Steer-A effect (pos-neg): {steer_a_effect:+.3f}")
    print(f"  Steer-B effect (pos-neg): {steer_b_effect:+.3f}")
    print(f"  Interaction (should be positive): {interaction:+.3f}")

    # Regression for each target
    for target in ["task_a", "task_b"]:
        target_results = [r for r in valid if r["steer_target"] == target]
        coefs = [r["coefficient"] for r in target_results]
        choices = [1.0 if r["choice"] == "a" else 0.0 for r in target_results]
        slope, intercept, r_value, p_value, _ = stats.linregress(coefs, choices)
        expected_sign = "+" if target == "task_a" else "-"
        print(f"  Regression ({target}): slope={slope:+.2e}, p={p_value:.4f} (expected {expected_sign})")

    return coefficients, h1_data


def plot_results():
    """Create combined plot for Phase 3 results."""
    date_str = datetime.now().strftime("%m%d%y")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # H3 plot
    h3_data = analyze_h3()
    if h3_data:
        coefficients, p_a_values, slope, r2, p_val = h3_data
        ax = axes[0]
        ax.plot(coefficients, p_a_values, "o-", color="darkblue", markersize=8, linewidth=2)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Steering Coefficient", fontsize=12)
        ax.set_ylabel("P(Choose A)", fontsize=12)
        ax.set_title("H3: Autoregressive Steering\n(last-token during choice)", fontsize=13)
        ax.set_ylim(0.3, 0.7)
        ax.text(0.98, 0.02, f"slope={slope:.2e}\nR²={r2:.3f}\np={p_val:.4f}",
                transform=ax.transAxes, fontsize=10, va="bottom", ha="right",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})

    # H1 plot
    h1_data = analyze_h1()
    if h1_data:
        coefficients, data = h1_data
        ax = axes[1]
        ax.plot(coefficients, data["task_a"], "o-", color="blue", markersize=8, linewidth=2,
                label="Steer on Task A")
        ax.plot(coefficients, data["task_b"], "s-", color="red", markersize=8, linewidth=2,
                label="Steer on Task B")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Steering Coefficient", fontsize=12)
        ax.set_ylabel("P(Choose A)", fontsize=12)
        ax.set_title("H1: Task-Selective Steering\n(steer one task's tokens only)", fontsize=13)
        ax.set_ylim(0.3, 0.7)
        ax.legend(loc="upper right")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"plot_{date_str}_revealed_preference.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot to {plot_path}")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    assets_path = ASSETS_DIR / f"plot_{date_str}_revealed_preference.png"
    shutil.copy(plot_path, assets_path)
    print(f"Copied to {assets_path}")


if __name__ == "__main__":
    plot_results()
