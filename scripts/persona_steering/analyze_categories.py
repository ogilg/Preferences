"""Analysis: P(choose category) dose-response, per-pair shifts, and statistics.

Uses cross-category task pairs with category metadata to compute:
- P(choose harmful/creative/math) vs coefficient for each persona
- Per-pair scatter at max coefficient
- Regression statistics
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

RESULTS_DIR = Path("results/experiments/persona_steering/preference_steering")
ASSETS_DIR = Path("experiments/persona_vectors/persona_steering/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

PERSONAS = ["sadist", "villain", "predator", "aesthete", "stem_obsessive"]
CATEGORIES = ["harmful", "creative", "math"]

PERSONA_COLORS = {
    "sadist": "#d62728",
    "villain": "#1f77b4",
    "predator": "#ff7f0e",
    "aesthete": "#9467bd",
    "stem_obsessive": "#2ca02c",
}

CATEGORY_COLORS = {
    "harmful": "#d62728",
    "creative": "#9467bd",
    "math": "#2ca02c",
}


def load_all_results() -> dict:
    with open(RESULTS_DIR / "steering_results.json") as f:
        return json.load(f)


def compute_category_choice_rates(records: list[dict]) -> dict:
    """For a set of records, compute P(choose category) for each category.

    Each pair involves two categories. For each pair, the judgment tells us
    which task was chosen (task_a or task_b), and we know the category of each.
    """
    category_chosen = {cat: 0 for cat in CATEGORIES}
    category_total = {cat: 0 for cat in CATEGORIES}

    for r in records:
        cat_a = r["task_a_category"]
        cat_b = r["task_b_category"]
        p_a = r["p_task_a"]  # probability of choosing task_a

        # Both categories get counted for total appearances
        category_total[cat_a] += 1
        category_total[cat_b] += 1

        # Weighted choice counts
        category_chosen[cat_a] += p_a
        category_chosen[cat_b] += (1 - p_a)

    rates = {}
    for cat in CATEGORIES:
        if category_total[cat] > 0:
            rates[cat] = category_chosen[cat] / category_total[cat]
        else:
            rates[cat] = 0.5
    return rates


def compute_dose_response_by_category(all_results: dict) -> dict:
    """For each persona, compute P(choose category) at each multiplier."""
    dose_response = {}

    for persona in PERSONAS:
        if persona not in all_results:
            continue
        records = all_results[persona]
        baseline_records = all_results["baseline"]

        by_mult = {}
        for r in records:
            m = r["multiplier"]
            if m not in by_mult:
                by_mult[m] = []
            by_mult[m].append(r)

        # Add baseline at mult=0
        by_mult[0.0] = baseline_records

        mults = sorted(by_mult.keys())
        per_category = {cat: {"mults": mults, "rates": [], "per_pair_rates": []} for cat in CATEGORIES}

        for m in mults:
            recs = by_mult[m]
            rates = compute_category_choice_rates(recs)
            for cat in CATEGORIES:
                per_category[cat]["rates"].append(rates[cat])

            # Also compute per-pair rates for error bars
            for cat in CATEGORIES:
                pair_rates = []
                for r in recs:
                    if r["task_a_category"] == cat:
                        pair_rates.append(r["p_task_a"])
                    elif r["task_b_category"] == cat:
                        pair_rates.append(1 - r["p_task_a"])
                per_category[cat]["per_pair_rates"].append(pair_rates)

        dose_response[persona] = per_category

    return dose_response


def compute_pair_type_shifts(all_results: dict) -> dict:
    """Compute P(A|steered) vs P(A|baseline) grouped by pair type."""
    baseline = all_results["baseline"]
    baseline_by_pair = {r["pair_idx"]: r["p_task_a"] for r in baseline}

    shifts = {}
    for persona in PERSONAS:
        if persona not in all_results:
            continue
        records = all_results[persona]
        # Find max positive multiplier
        mults = sorted(set(r["multiplier"] for r in records))
        max_pos = max(m for m in mults if m > 0)
        max_neg = min(m for m in mults if m < 0)

        persona_shifts = {"max_pos_mult": max_pos, "max_neg_mult": max_neg, "pairs": []}
        for r in records:
            if r["multiplier"] in (max_pos, max_neg):
                pi = r["pair_idx"]
                if pi in baseline_by_pair:
                    persona_shifts["pairs"].append({
                        "pair_idx": pi,
                        "pair_type": r["pair_type"],
                        "multiplier": r["multiplier"],
                        "baseline_pa": baseline_by_pair[pi],
                        "steered_pa": r["p_task_a"],
                        "task_a_category": r["task_a_category"],
                        "task_b_category": r["task_b_category"],
                    })
        shifts[persona] = persona_shifts

    return shifts


def compute_regression_stats(all_results: dict) -> dict:
    """Per-persona, per-category regression of P(choose cat) on multiplier."""
    dose_response = compute_dose_response_by_category(all_results)
    reg_stats = {}

    for persona in PERSONAS:
        if persona not in dose_response:
            continue
        reg_stats[persona] = {}
        for cat in CATEGORIES:
            dr = dose_response[persona][cat]
            mults = np.array(dr["mults"])
            rates = np.array(dr["rates"])

            if len(mults) < 3:
                continue

            slope, intercept, r_value, p_value, std_err = sp_stats.linregress(mults, rates)
            reg_stats[persona][cat] = {
                "slope": round(slope, 4),
                "intercept": round(intercept, 4),
                "r_squared": round(r_value**2, 4),
                "p_value": round(p_value, 4),
            }

    return reg_stats


def plot_dose_response(dose_response: dict):
    """5 subplots (one per persona), 3 lines per subplot (one per category)."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)

    for idx, persona in enumerate(PERSONAS):
        ax = axes[idx]
        if persona not in dose_response:
            ax.set_title(persona, fontsize=10)
            continue

        for cat in CATEGORIES:
            dr = dose_response[persona][cat]
            mults = dr["mults"]
            rates = dr["rates"]

            # Compute SEM from per-pair rates
            sems = []
            for pair_rates in dr["per_pair_rates"]:
                if len(pair_rates) > 1:
                    sems.append(sp_stats.sem(pair_rates))
                else:
                    sems.append(0)

            ax.errorbar(
                mults, rates, yerr=sems,
                marker="o", capsize=3, label=cat,
                color=CATEGORY_COLORS[cat], linewidth=1.5, markersize=4,
            )

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axvline(0.0, color="gray", linestyle=":", alpha=0.3)
        ax.set_xlabel("Multiplier")
        if idx == 0:
            ax.set_ylabel("P(choose category)")
        ax.set_title(persona, fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.2)

    plt.suptitle("Category Preference vs Steering Coefficient", y=1.02, fontsize=12)
    plt.tight_layout()
    out = ASSETS_DIR / "plot_030726_category_dose_response.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_per_pair_scatter(shifts: dict):
    """Per-pair P(A|steered) vs P(A|baseline) colored by pair type."""
    pair_type_colors = {
        "creative-harmful": "#d62728",
        "harmful-math": "#ff7f0e",
        "creative-math": "#2ca02c",
    }

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)

    for idx, persona in enumerate(PERSONAS):
        ax = axes[idx]
        if persona not in shifts:
            ax.set_title(persona, fontsize=10)
            continue

        data = shifts[persona]
        for pair_data in data["pairs"]:
            if pair_data["multiplier"] == data["max_pos_mult"]:
                pt = pair_data["pair_type"]
                color = pair_type_colors.get(pt, "gray")
                ax.scatter(
                    pair_data["baseline_pa"], pair_data["steered_pa"],
                    color=color, alpha=0.6, s=25, edgecolors="none",
                )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("P(A) baseline")
        if idx == 0:
            ax.set_ylabel("P(A) steered (max+)")
        ax.set_title(f"{persona}\n(+{data['max_pos_mult']:.2f}x)", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    # Add legend
    for pt, color in pair_type_colors.items():
        axes[-1].scatter([], [], color=color, label=pt, s=25)
    axes[-1].legend(fontsize=7, loc="lower right")

    plt.suptitle("Per-Pair Preference Shifts at Max Positive Coefficient", y=1.02, fontsize=12)
    plt.tight_layout()
    out = ASSETS_DIR / "plot_030726_per_pair_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_unclear_rates(all_results: dict):
    """Plot unclear rate by multiplier for each persona."""
    fig, ax = plt.subplots(figsize=(8, 4))

    for persona in PERSONAS:
        if persona not in all_results:
            continue
        records = all_results[persona]
        by_mult = {}
        for r in records:
            m = r["multiplier"]
            if m not in by_mult:
                by_mult[m] = {"unclear": 0, "total": 0}
            by_mult[m]["unclear"] += r["n_unclear"]
            by_mult[m]["total"] += r["total_valid"] + r["n_unclear"]

        # Add baseline
        for r in all_results["baseline"]:
            if 0.0 not in by_mult:
                by_mult[0.0] = {"unclear": 0, "total": 0}
            by_mult[0.0]["unclear"] += r["n_unclear"]
            by_mult[0.0]["total"] += r["total_valid"] + r["n_unclear"]

        mults = sorted(by_mult.keys())
        rates = [by_mult[m]["unclear"] / by_mult[m]["total"] if by_mult[m]["total"] > 0 else 0 for m in mults]
        ax.plot(mults, rates, marker="s", label=persona, color=PERSONA_COLORS[persona], linewidth=1.5, markersize=4)

    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Multiplier")
    ax.set_ylabel("Unclear rate")
    ax.set_title("Choice Ambiguity by Steering Coefficient")
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = ASSETS_DIR / "plot_030726_unclear_rates.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    all_results = load_all_results()

    # Compute all stats
    dose_response = compute_dose_response_by_category(all_results)
    shifts = compute_pair_type_shifts(all_results)
    reg_stats = compute_regression_stats(all_results)

    # Print summary
    print("=" * 70)
    print("CATEGORY PREFERENCE ANALYSIS")
    print("=" * 70)

    for persona in PERSONAS:
        if persona not in dose_response:
            continue
        print(f"\n{'─'*50}")
        print(f"  {persona.upper()}")
        dr = dose_response[persona]
        for cat in CATEGORIES:
            mults = dr[cat]["mults"]
            rates = dr[cat]["rates"]
            print(f"\n  P(choose {cat}):")
            for m, r in zip(mults, rates):
                marker = " <-- baseline" if m == 0.0 else ""
                print(f"    {m:+.3f}x: {r:.3f}{marker}")

        if persona in reg_stats:
            print(f"\n  Regression stats:")
            for cat, rs in reg_stats[persona].items():
                print(f"    {cat}: slope={rs['slope']:.4f}, R²={rs['r_squared']:.4f}, p={rs['p_value']:.4f}")

    # Baseline category rates
    print(f"\n{'='*70}")
    print("BASELINE CATEGORY RATES")
    baseline_rates = compute_category_choice_rates(all_results["baseline"])
    for cat, rate in baseline_rates.items():
        print(f"  P(choose {cat}): {rate:.3f}")

    # Generate plots
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    plot_dose_response(dose_response)
    plot_per_pair_scatter(shifts)
    plot_unclear_rates(all_results)

    # Save analysis JSON
    analysis = {
        "baseline_rates": baseline_rates,
        "regression_stats": reg_stats,
        "dose_response": {
            p: {
                cat: {"mults": dr[cat]["mults"], "rates": dr[cat]["rates"]}
                for cat in CATEGORIES
            }
            for p, dr in dose_response.items()
        },
    }
    with open(RESULTS_DIR / "category_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print("\nSaved category_analysis.json")


if __name__ == "__main__":
    main()
