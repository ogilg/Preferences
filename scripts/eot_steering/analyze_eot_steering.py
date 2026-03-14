"""Analyze multi-turn EOT steering experiment results.

Produces:
1. Dose-response curve: P(choose high-mu task) vs coefficient, bootstrap 95% CIs
2. By Δmu stratum: separate curves for borderline, moderate, decisive
3. Per-pair slopes: distribution of linear regression slopes
4. Parse rate table per coefficient
5. Success criteria checks (PASS/FAIL)

Usage:
    python scripts/eot_steering/analyze_eot_steering.py [--checkpoint PATH]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUTPUT_DIR = Path("experiments/steering/multi_turn_pairwise/eot_steering")
ASSETS_DIR = OUTPUT_DIR / "assets"
DEFAULT_CHECKPOINT = OUTPUT_DIR / "checkpoint.jsonl"


def load_results(checkpoint_path: Path) -> list[dict]:
    results = []
    with open(checkpoint_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_parse_rates(results: list[dict]) -> dict[float, dict]:
    """Compute parse rate per multiplier."""
    by_mult: dict[float, dict] = defaultdict(lambda: {"total": 0, "parsed": 0, "parse_fail": 0})
    for r in results:
        m = r["multiplier"]
        by_mult[m]["total"] += 1
        if r["choice"] == "parse_fail":
            by_mult[m]["parse_fail"] += 1
        else:
            by_mult[m]["parsed"] += 1
    return dict(by_mult)


def compute_p_high_mu(results: list[dict], multiplier: float) -> tuple[float, int, int]:
    """Compute P(choose high-mu task) for a given multiplier, excluding parse failures."""
    n_chose_high = 0
    n_valid = 0
    for r in results:
        if r["multiplier"] != multiplier:
            continue
        if r["chose_high_mu"] is None:
            continue
        n_valid += 1
        if r["chose_high_mu"]:
            n_chose_high += 1
    if n_valid == 0:
        return float("nan"), 0, 0
    return n_chose_high / n_valid, n_chose_high, n_valid


def bootstrap_p_high_mu(
    results: list[dict],
    multiplier: float,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap mean and 95% CI for P(choose high-mu) at a given multiplier.

    Resamples at the pair level (cluster bootstrap) to respect within-pair correlation.
    """
    rng = np.random.RandomState(seed)

    # Group valid results by pair_id
    by_pair: dict[int, list[bool]] = defaultdict(list)
    for r in results:
        if r["multiplier"] != multiplier:
            continue
        if r["chose_high_mu"] is None:
            continue
        by_pair[r["pair_id"]].append(r["chose_high_mu"])

    pair_ids = list(by_pair.keys())
    n_pairs = len(pair_ids)
    if n_pairs == 0:
        return float("nan"), float("nan"), float("nan")

    # Compute pair-level means
    pair_means = np.array([np.mean(by_pair[pid]) for pid in pair_ids])
    observed = pair_means.mean()

    # Bootstrap
    boot_means = []
    for _ in range(n_boot):
        idx = rng.randint(0, n_pairs, n_pairs)
        boot_means.append(pair_means[idx].mean())
    boot_means = np.array(boot_means)

    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)
    return observed, ci_lo, ci_hi


def compute_stratum_results(
    results: list[dict],
    stratum: str,
) -> dict[float, tuple[float, float, float]]:
    """Compute P(high-mu) with bootstrap CIs per multiplier for a given stratum."""
    filtered = [r for r in results if r["stratum"] == stratum]
    multipliers = sorted(set(r["multiplier"] for r in filtered))
    return {m: bootstrap_p_high_mu(filtered, m) for m in multipliers}


def compute_per_pair_slopes(results: list[dict]) -> np.ndarray:
    """For each pair, regress chose_high_mu on multiplier, return slopes."""
    by_pair: dict[int, dict[float, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["chose_high_mu"] is None:
            continue
        by_pair[r["pair_id"]][r["multiplier"]].append(r["chose_high_mu"])

    slopes = []
    for pair_id, mult_data in by_pair.items():
        mults = sorted(mult_data.keys())
        if len(mults) < 3:
            continue
        x = np.array(mults)
        y = np.array([np.mean(mult_data[m]) for m in mults])
        if np.std(x) == 0:
            continue
        slope, _, _, _, _ = stats.linregress(x, y)
        slopes.append(slope)

    return np.array(slopes)


def plot_dose_response(
    results: list[dict],
    output_path: Path,
):
    """Plot overall dose-response curve with bootstrap CIs."""
    multipliers = sorted(set(r["multiplier"] for r in results))

    means, ci_los, ci_his = [], [], []
    for m in multipliers:
        mean, ci_lo, ci_hi = bootstrap_p_high_mu(results, m)
        means.append(mean)
        ci_los.append(ci_lo)
        ci_his.append(ci_hi)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(multipliers, means, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.fill_between(multipliers, ci_los, ci_his, alpha=0.2, color="steelblue")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering multiplier", fontsize=12)
    ax.set_ylabel("P(choose high-mu task)", fontsize=12)
    ax.set_title("EOT Steering: Dose-Response", fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_by_stratum(
    results: list[dict],
    output_path: Path,
):
    """Plot dose-response curves per Δmu stratum."""
    strata = ["borderline", "moderate", "decisive"]
    colors = {"borderline": "coral", "moderate": "steelblue", "decisive": "seagreen"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for stratum in strata:
        stratum_data = compute_stratum_results(results, stratum)
        if not stratum_data:
            continue
        mults = sorted(stratum_data.keys())
        means = [stratum_data[m][0] for m in mults]
        ci_los = [stratum_data[m][1] for m in mults]
        ci_his = [stratum_data[m][2] for m in mults]
        color = colors[stratum]
        n_pairs = len(set(r["pair_id"] for r in results if r["stratum"] == stratum))
        ax.plot(mults, means, "o-", color=color, linewidth=2, markersize=6,
                label=f"{stratum} (n={n_pairs})")
        ax.fill_between(mults, ci_los, ci_his, alpha=0.15, color=color)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering multiplier", fontsize=12)
    ax.set_ylabel("P(choose high-mu task)", fontsize=12)
    ax.set_title("EOT Steering by Preference Strength", fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_slope_distribution(
    slopes: np.ndarray,
    output_path: Path,
):
    """Plot histogram of per-pair slopes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(slopes, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="zero")
    ax.axvline(np.median(slopes), color="orange", linestyle="-", linewidth=2,
               label=f"median={np.median(slopes):.2f}")
    ax.set_xlabel("Per-pair slope (Δ P(high-mu) / Δ multiplier)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Per-Pair Steering Slopes", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_results(args.checkpoint)
    print(f"  {len(results)} total trials")

    # --- Parse rate table ---
    print("\n=== Parse Rates ===")
    parse_rates = compute_parse_rates(results)
    multipliers = sorted(parse_rates.keys())
    print(f"{'Multiplier':>12} {'Total':>8} {'Parsed':>8} {'Fail':>8} {'Rate':>8}")
    for m in multipliers:
        d = parse_rates[m]
        rate = d["parsed"] / d["total"] if d["total"] > 0 else 0
        print(f"{m:>+12.3f} {d['total']:>8d} {d['parsed']:>8d} {d['parse_fail']:>8d} {rate:>8.1%}")

    overall_parsed = sum(d["parsed"] for d in parse_rates.values())
    overall_total = sum(d["total"] for d in parse_rates.values())
    overall_rate = overall_parsed / overall_total if overall_total > 0 else 0
    print(f"{'Overall':>12} {overall_total:>8d} {overall_parsed:>8d} {overall_total - overall_parsed:>8d} {overall_rate:>8.1%}")

    # --- Dose-response ---
    print("\n=== Dose-Response ===")
    print(f"{'Multiplier':>12} {'P(high-mu)':>12} {'95% CI':>18} {'N':>6}")
    for m in multipliers:
        mean, ci_lo, ci_hi = bootstrap_p_high_mu(results, m)
        _, _, n_valid = compute_p_high_mu(results, m)
        print(f"{m:>+12.3f} {mean:>12.3f} [{ci_lo:.3f}, {ci_hi:.3f}] {n_valid:>6d}")

    # --- Steering effect ---
    print("\n=== Steering Effect ===")
    max_mult = max(m for m in multipliers if m > 0)
    min_mult = min(m for m in multipliers if m < 0)
    p_pos, _, _ = bootstrap_p_high_mu(results, max_mult)
    p_neg, _, _ = bootstrap_p_high_mu(results, min_mult)
    steering_effect = p_pos - p_neg
    print(f"  P(high-mu | mult={max_mult:+.3f}) = {p_pos:.3f}")
    print(f"  P(high-mu | mult={min_mult:+.3f}) = {p_neg:.3f}")
    print(f"  Steering effect = {steering_effect:.3f} ({steering_effect*100:.1f} pp)")

    # --- By stratum ---
    print("\n=== By Stratum ===")
    strata_effects = {}
    for stratum in ["borderline", "moderate", "decisive"]:
        filtered = [r for r in results if r["stratum"] == stratum]
        if not filtered:
            continue
        p_pos_s, _, _ = bootstrap_p_high_mu(filtered, max_mult)
        p_neg_s, _, _ = bootstrap_p_high_mu(filtered, min_mult)
        effect = p_pos_s - p_neg_s
        strata_effects[stratum] = effect
        n_pairs = len(set(r["pair_id"] for r in filtered))
        print(f"  {stratum:>12}: effect = {effect:.3f} ({effect*100:.1f} pp), n={n_pairs} pairs")

    # --- Per-pair slopes ---
    print("\n=== Per-Pair Slopes ===")
    slopes = compute_per_pair_slopes(results)
    print(f"  N pairs with slopes: {len(slopes)}")
    print(f"  Mean slope: {slopes.mean():.4f}")
    print(f"  Median slope: {np.median(slopes):.4f}")
    print(f"  Fraction positive: {(slopes > 0).mean():.3f}")

    # --- Plots ---
    print("\n=== Generating Plots ===")
    from datetime import date
    date_str = date.today().strftime("%m%d%y")

    plot_dose_response(results, ASSETS_DIR / f"plot_{date_str}_dose_response.png")
    plot_by_stratum(results, ASSETS_DIR / f"plot_{date_str}_by_stratum.png")
    plot_slope_distribution(slopes, ASSETS_DIR / f"plot_{date_str}_slope_distribution.png")

    # --- Success Criteria ---
    print("\n=== Success Criteria ===")

    # 1. Monotonic dose-response
    dose_means = []
    for m in multipliers:
        mean, _, _ = bootstrap_p_high_mu(results, m)
        dose_means.append(mean)
    spearman_r, spearman_p = stats.spearmanr(multipliers, dose_means)
    criterion_1 = spearman_r > 0 and spearman_p < 0.05
    print(f"  1. Monotonic dose-response: Spearman r={spearman_r:.3f}, p={spearman_p:.4f} -> {'PASS' if criterion_1 else 'FAIL'}")

    # 2. Steering effect > 10pp
    criterion_2 = steering_effect > 0.10
    print(f"  2. Steering effect > 10pp: {steering_effect*100:.1f}pp -> {'PASS' if criterion_2 else 'FAIL'}")

    # 3. Gradient by difficulty
    if "borderline" in strata_effects and "decisive" in strata_effects:
        criterion_3 = strata_effects["borderline"] > strata_effects["decisive"]
        print(f"  3. Borderline > Decisive: {strata_effects['borderline']*100:.1f}pp vs {strata_effects['decisive']*100:.1f}pp -> {'PASS' if criterion_3 else 'FAIL'}")
    else:
        criterion_3 = False
        print(f"  3. Gradient by difficulty: insufficient data -> FAIL")

    # 4. Parse rates > 90%
    min_parse_rate = min(d["parsed"] / d["total"] for d in parse_rates.values() if d["total"] > 0)
    criterion_4 = min_parse_rate > 0.90
    print(f"  4. Parse rates > 90%: min={min_parse_rate:.1%} -> {'PASS' if criterion_4 else 'FAIL'}")

    all_pass = criterion_1 and criterion_2 and criterion_3 and criterion_4
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    # Save summary JSON
    summary = {
        "n_trials": len(results),
        "n_pairs": len(set(r["pair_id"] for r in results)),
        "overall_parse_rate": overall_rate,
        "steering_effect_pp": float(steering_effect * 100),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "strata_effects_pp": {k: v * 100 for k, v in strata_effects.items()},
        "mean_slope": float(slopes.mean()),
        "median_slope": float(np.median(slopes)),
        "fraction_positive_slopes": float((slopes > 0).mean()),
        "criteria": {
            "monotonic_dose_response": bool(criterion_1),
            "steering_effect_gt_10pp": bool(criterion_2),
            "borderline_gt_decisive": bool(criterion_3),
            "parse_rates_gt_90pct": bool(criterion_4),
        },
    }
    summary_path = OUTPUT_DIR / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
