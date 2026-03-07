"""Analysis for persona steering preference experiment.

Dose-response plots and statistical tests for whether persona vectors shift preferences.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS_DIR = Path("results/experiments/persona_steering/preference_steering")
TRIAGE_PATH = Path("results/experiments/persona_steering/coherence_triage.json")
ASSETS_DIR = Path("experiments/persona_vectors/persona_steering/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

PERSONA_COLORS = {
    "sadist": "#d62728",
    "villain": "#1f77b4",
    "predator": "#ff7f0e",
    "aesthete": "#9467bd",
    "stem_obsessive": "#2ca02c",
}


def load_data():
    with open(RESULTS_DIR / "steering_results.json") as f:
        return json.load(f)


def load_triage():
    with open(TRIAGE_PATH) as f:
        return json.load(f)


def compute_dose_response(results: dict) -> dict:
    """For each persona, compute mean P(A) and SEM at each multiplier."""
    dose_response = {}
    for persona, records in results.items():
        by_mult = {}
        for r in records:
            m = r["multiplier"]
            if m not in by_mult:
                by_mult[m] = []
            by_mult[m].append(r["p_task_a"])

        mults = sorted(by_mult.keys())
        means = [np.mean(by_mult[m]) for m in mults]
        sems = [stats.sem(by_mult[m]) for m in mults]
        n_pairs = [len(by_mult[m]) for m in mults]

        dose_response[persona] = {
            "multipliers": mults,
            "means": means,
            "sems": sems,
            "n_pairs": n_pairs,
        }
    return dose_response


def compute_flip_stats(results: dict) -> dict:
    """For each persona, count how many pairs flip choice relative to baseline."""
    flip_stats = {}
    for persona, records in results.items():
        baseline_by_pair = {}
        steered_by_pair = {}
        for r in records:
            pi = r["pair_idx"]
            if r["multiplier"] == 0.0:
                baseline_by_pair[pi] = r["p_task_a"]
            else:
                if pi not in steered_by_pair:
                    steered_by_pair[pi] = []
                steered_by_pair[pi].append((r["multiplier"], r["p_task_a"]))

        n_flipped = 0
        n_total = 0
        flip_details = []
        for pi in baseline_by_pair:
            base_choice = "A" if baseline_by_pair[pi] >= 0.5 else "B"
            if pi in steered_by_pair:
                for mult, pa in steered_by_pair[pi]:
                    steered_choice = "A" if pa >= 0.5 else "B"
                    flipped = base_choice != steered_choice
                    if flipped:
                        n_flipped += 1
                    n_total += 1
                    flip_details.append({
                        "pair_idx": pi,
                        "multiplier": mult,
                        "baseline_pa": baseline_by_pair[pi],
                        "steered_pa": pa,
                        "flipped": flipped,
                    })

        flip_stats[persona] = {
            "n_flipped": n_flipped,
            "n_total": n_total,
            "flip_rate": n_flipped / n_total if n_total > 0 else 0,
            "details": flip_details,
        }
    return flip_stats


def compute_regression_stats(results: dict) -> dict:
    """Linear regression of P(A) on multiplier for each persona.

    Tests whether there's a systematic linear trend in preferences as
    steering coefficient increases.
    """
    reg_stats = {}
    for persona, records in results.items():
        mults = np.array([r["multiplier"] for r in records])
        p_as = np.array([r["p_task_a"] for r in records])

        # Exclude baseline for slope test (we want to see if steering has an effect)
        mask_nonzero = mults != 0.0
        if mask_nonzero.sum() < 3:
            continue

        # Full regression including baseline
        slope, intercept, r_value, p_value, std_err = stats.linregress(mults, p_as)

        # Also compute Spearman rank correlation (more robust)
        spearman_r, spearman_p = stats.spearmanr(mults, p_as)

        reg_stats[persona] = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
        }
    return reg_stats


def compute_unclear_stats(results: dict) -> dict:
    """Track unclear rates by multiplier — high unclear = degraded output."""
    stats_out = {}
    for persona, records in results.items():
        by_mult = {}
        for r in records:
            m = r["multiplier"]
            if m not in by_mult:
                by_mult[m] = {"unclear": 0, "total_trials": 0}
            by_mult[m]["unclear"] += r["n_unclear"]
            by_mult[m]["total_trials"] += r["total_valid"] + r["n_unclear"]

        stats_out[persona] = {
            m: {
                "unclear_rate": d["unclear"] / d["total_trials"] if d["total_trials"] > 0 else 0,
                "n_unclear": d["unclear"],
                "n_total": d["total_trials"],
            }
            for m, d in sorted(by_mult.items())
        }
    return stats_out


def plot_dose_response(dose_response: dict, unclear_stats: dict):
    """Main dose-response plot: mean P(A) vs multiplier for each persona."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: dose-response curves
    ax = axes[0]
    for persona, dr in dose_response.items():
        color = PERSONA_COLORS[persona]
        ax.errorbar(
            dr["multipliers"], dr["means"], yerr=dr["sems"],
            marker="o", capsize=3, label=persona, color=color, linewidth=1.5,
        )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Steering multiplier (× mean activation norm)")
    ax.set_ylabel("Mean P(choose Task A)")
    ax.set_title("Persona Steering: Dose-Response")
    ax.set_ylim(0.3, 0.7)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Right: unclear rate by multiplier
    ax = axes[1]
    for persona, us in unclear_stats.items():
        color = PERSONA_COLORS[persona]
        mults = sorted(us.keys())
        rates = [us[m]["unclear_rate"] for m in mults]
        ax.plot(mults, rates, marker="s", label=persona, color=color, linewidth=1.5, markersize=4)
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Steering multiplier")
    ax.set_ylabel("Unclear rate")
    ax.set_title("Output Degradation by Coefficient")
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_030726_dose_response.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved dose-response plot")


def plot_per_pair_shifts(results: dict):
    """For each persona, scatter P(A|steered) vs P(A|baseline) at max coefficient."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    for idx, (persona, records) in enumerate(results.items()):
        ax = axes[idx]
        color = PERSONA_COLORS[persona]

        baseline_by_pair = {}
        max_pos_by_pair = {}
        max_neg_by_pair = {}

        mults = sorted(set(r["multiplier"] for r in records))
        max_pos = max(m for m in mults if m > 0)
        max_neg = min(m for m in mults if m < 0)

        for r in records:
            pi = r["pair_idx"]
            if r["multiplier"] == 0.0:
                baseline_by_pair[pi] = r["p_task_a"]
            elif r["multiplier"] == max_pos:
                max_pos_by_pair[pi] = r["p_task_a"]
            elif r["multiplier"] == max_neg:
                max_neg_by_pair[pi] = r["p_task_a"]

        # Plot positive steering
        pairs_common = sorted(set(baseline_by_pair.keys()) & set(max_pos_by_pair.keys()))
        base_vals = [baseline_by_pair[p] for p in pairs_common]
        pos_vals = [max_pos_by_pair[p] for p in pairs_common]
        ax.scatter(base_vals, pos_vals, alpha=0.5, s=20, color=color, label=f"+{max_pos}×")

        # Plot negative steering
        pairs_common_neg = sorted(set(baseline_by_pair.keys()) & set(max_neg_by_pair.keys()))
        base_vals_neg = [baseline_by_pair[p] for p in pairs_common_neg]
        neg_vals = [max_neg_by_pair[p] for p in pairs_common_neg]
        ax.scatter(base_vals_neg, neg_vals, alpha=0.5, s=20, color="gray", marker="^", label=f"{max_neg}×")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("P(A) baseline")
        if idx == 0:
            ax.set_ylabel("P(A) steered")
        ax.set_title(persona, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.2)

    plt.suptitle("Per-Pair Preference Shifts at Max Coefficients", y=1.02)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_030726_per_pair_shifts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved per-pair shifts plot")


def main():
    results = load_data()
    triage = load_triage()

    # Compute all stats
    dose_response = compute_dose_response(results)
    flip_stats = compute_flip_stats(results)
    reg_stats = compute_regression_stats(results)
    unclear_stats = compute_unclear_stats(results)

    # Print summary
    print("=" * 70)
    print("PERSONA STEERING ANALYSIS SUMMARY")
    print("=" * 70)

    for persona in results:
        dr = dose_response[persona]
        fs = flip_stats[persona]
        rs = reg_stats.get(persona, {})

        print(f"\n{'─'*50}")
        print(f"  {persona.upper()}")
        print(f"  Layer: {triage[persona]['layer']}, Mean norm: {triage[persona]['mean_norm']:.0f}")
        print(f"  Coherent range: [{triage[persona]['negative_max_multiplier']*-1:+.1f}×, +{triage[persona]['positive_max_multiplier']:.1f}×]")
        print()
        print(f"  Dose-response (mean P(A) ± SEM):")
        for m, mean, sem in zip(dr["multipliers"], dr["means"], dr["sems"]):
            marker = " ← baseline" if m == 0.0 else ""
            print(f"    {m:+.2f}×: {mean:.3f} ± {sem:.3f}{marker}")
        print()
        print(f"  Flip rate: {fs['n_flipped']}/{fs['n_total']} = {fs['flip_rate']:.1%}")
        if rs:
            print(f"  Linear regression: slope={rs['slope']:.4f}, R²={rs['r_squared']:.4f}, p={rs['p_value']:.4f}")
            print(f"  Spearman: r={rs['spearman_r']:.4f}, p={rs['spearman_p']:.4f}")

    # Unclear stats summary
    print(f"\n{'='*70}")
    print("UNCLEAR RATES")
    for persona, us in unclear_stats.items():
        rates = [(m, d["unclear_rate"]) for m, d in us.items()]
        max_rate = max(r for _, r in rates)
        print(f"  {persona}: max unclear rate = {max_rate:.1%} (at mult={max(rates, key=lambda x: x[1])[0]:+.2f})")

    # Generate plots
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    plot_dose_response(dose_response, unclear_stats)
    plot_per_pair_shifts(results)

    # Save analysis JSON for report
    analysis = {
        "dose_response": {p: {k: v for k, v in d.items()} for p, d in dose_response.items()},
        "flip_stats": {p: {k: v for k, v in d.items() if k != "details"} for p, d in flip_stats.items()},
        "regression_stats": reg_stats,
        "unclear_stats": {p: {str(k): v for k, v in d.items()} for p, d in unclear_stats.items()},
    }
    with open(RESULTS_DIR / "analysis_summary.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print("\nSaved analysis_summary.json")


if __name__ == "__main__":
    main()
