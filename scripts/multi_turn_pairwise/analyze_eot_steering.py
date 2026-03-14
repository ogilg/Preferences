"""Analysis of multi-turn differential EOT steering experiment.

Reads checkpoint JSONL and produces:
1. Dose-response curve: P(choose high-mu) vs multiplier with bootstrap 95% CIs
2. Ordering bias per condition: P(A-position | AB) - P(A-position | BA)
3. Steering effect per magnitude: P(high-mu | +m) - P(high-mu | -m)
4. By delta-mu stratum: steering effect for borderline, moderate, decisive
5. Parse rate table per condition
6. Success criteria PASS/FAIL assertions

Usage:
    python scripts/multi_turn_pairwise/analyze_eot_steering.py [--pilot]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

EXPERIMENT_DIR = Path("experiments/steering/multi_turn_pairwise/eot_steering")
ASSETS_DIR = EXPERIMENT_DIR / "assets"


def load_records(checkpoint_path: Path) -> list[dict]:
    records = []
    with open(checkpoint_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def bootstrap_ci(values: list[bool], n_boot: int = 10000, ci: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap mean and CI for binary values."""
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    if len(arr) < 2:
        return mean, mean, mean
    boot_means = np.array([
        np.random.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return mean, lo, hi


def classify_stratum(delta_mu: float) -> str:
    if abs(delta_mu) < 1:
        return "borderline"
    elif abs(delta_mu) < 3:
        return "moderate"
    else:
        return "decisive"


def analyze(records: list[dict], suffix: str = ""):
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    multipliers = sorted(set(r["multiplier"] for r in records))
    valid_records = [r for r in records if r["chose_high_mu"] is not None]
    print(f"Total records: {len(records)}, valid (parsed): {len(valid_records)}")
    print(f"Multipliers: {multipliers}")
    print()

    # ── 1. Parse rate table ────────────────────────────────────────────────
    print("=" * 60)
    print("1. PARSE RATE TABLE")
    print("=" * 60)
    print(f"{'Multiplier':>12} {'Total':>8} {'Parsed':>8} {'Rate':>8}")
    for m in multipliers:
        total = sum(1 for r in records if r["multiplier"] == m)
        parsed = sum(1 for r in records if r["multiplier"] == m and r["chose_high_mu"] is not None)
        rate = parsed / total if total > 0 else 0
        print(f"{m:+12.3f} {total:>8} {parsed:>8} {rate:>8.1%}")
    print()

    # ── 2. Dose-response curve ─────────────────────────────────────────────
    print("=" * 60)
    print("2. DOSE-RESPONSE: P(choose high-mu) vs multiplier")
    print("=" * 60)
    dose_data = {}
    print(f"{'Multiplier':>12} {'P(high-mu)':>12} {'95% CI':>20} {'N':>6}")
    for m in multipliers:
        subset = [r["chose_high_mu"] for r in valid_records if r["multiplier"] == m]
        if subset:
            mean, lo, hi = bootstrap_ci(subset)
            dose_data[m] = (mean, lo, hi, len(subset))
            print(f"{m:+12.3f} {mean:>12.3f} [{lo:.3f}, {hi:.3f}]{len(subset):>6}")

    # Spearman correlation for monotonicity
    mult_vals = []
    p_high_vals = []
    for m in multipliers:
        if m in dose_data:
            mult_vals.append(m)
            p_high_vals.append(dose_data[m][0])

    spearman_r, spearman_p = stats.spearmanr(mult_vals, p_high_vals)
    print(f"\nSpearman r = {spearman_r:.3f}, p = {spearman_p:.4f}")
    print()

    # Plot dose-response
    fig, ax = plt.subplots(figsize=(8, 5))
    mults = sorted(dose_data.keys())
    means = [dose_data[m][0] for m in mults]
    los = [dose_data[m][1] for m in mults]
    his = [dose_data[m][2] for m in mults]

    ax.errorbar(mults, means,
                yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                fmt="o-", capsize=5, color="steelblue", markersize=8)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Multiplier")
    ax.set_ylabel("P(choose high-mu task)")
    ax.set_title(f"Dose-response: EOT differential steering\n(Spearman r={spearman_r:.3f}, p={spearman_p:.4f})")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plot_path = ASSETS_DIR / f"plot_031426_dose_response{suffix}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")

    # ── 3. Ordering bias per condition ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. ORDERING BIAS: P(A-position | AB) - P(A-position | BA)")
    print("=" * 60)
    print(f"{'Multiplier':>12} {'P(A|AB)':>10} {'P(A|BA)':>10} {'Bias':>10}")
    for m in multipliers:
        ab = [r for r in valid_records if r["multiplier"] == m and r["ordering"] == 0]
        ba = [r for r in valid_records if r["multiplier"] == m and r["ordering"] == 1]
        if ab and ba:
            p_a_ab = sum(r["chose_a_position"] for r in ab) / len(ab)
            p_a_ba = sum(r["chose_a_position"] for r in ba) / len(ba)
            bias = p_a_ab - p_a_ba
            print(f"{m:+12.3f} {p_a_ab:>10.3f} {p_a_ba:>10.3f} {bias:>+10.3f}")
    print()

    # ── 4. Steering effect per magnitude ───────────────────────────────────
    print("=" * 60)
    print("4. STEERING EFFECT: P(high-mu | +m) - P(high-mu | -m)")
    print("=" * 60)
    magnitudes = sorted(set(abs(m) for m in multipliers if m != 0))
    steering_effects = {}
    print(f"{'|Multiplier|':>12} {'P(+m)':>10} {'P(-m)':>10} {'Effect':>10} {'95% CI':>20}")
    for mag in magnitudes:
        pos = [r["chose_high_mu"] for r in valid_records if r["multiplier"] == mag]
        neg = [r["chose_high_mu"] for r in valid_records if r["multiplier"] == -mag]
        if pos and neg:
            p_pos = np.mean(pos)
            p_neg = np.mean(neg)
            effect = p_pos - p_neg

            # Bootstrap CI on the effect
            pos_arr = np.array(pos, dtype=float)
            neg_arr = np.array(neg, dtype=float)
            boot_effects = []
            for _ in range(10000):
                bp = np.random.choice(pos_arr, size=len(pos_arr), replace=True).mean()
                bn = np.random.choice(neg_arr, size=len(neg_arr), replace=True).mean()
                boot_effects.append(bp - bn)
            lo = np.percentile(boot_effects, 2.5)
            hi = np.percentile(boot_effects, 97.5)
            steering_effects[mag] = (effect, lo, hi)
            print(f"{mag:>12.3f} {p_pos:>10.3f} {p_neg:>10.3f} {effect:>+10.3f} [{lo:+.3f}, {hi:+.3f}]")
    print()

    # ── 5. By Δmu stratum ─────────────────────────────────────────────────
    print("=" * 60)
    print("5. STEERING EFFECT BY STRATUM")
    print("=" * 60)
    strata = ["borderline", "moderate", "decisive"]
    stratum_effects = {}
    for stratum in strata:
        stratum_records = [r for r in valid_records if classify_stratum(r["delta_mu"]) == stratum]
        if not stratum_records:
            print(f"  {stratum}: no data")
            continue

        # Baseline P(high-mu)
        baseline = [r["chose_high_mu"] for r in stratum_records if r["multiplier"] == 0]
        p_base = np.mean(baseline) if baseline else float("nan")

        print(f"\n  {stratum.upper()} (n={len(stratum_records)}, baseline P(high-mu)={p_base:.3f}):")
        for mag in magnitudes:
            pos = [r["chose_high_mu"] for r in stratum_records if r["multiplier"] == mag]
            neg = [r["chose_high_mu"] for r in stratum_records if r["multiplier"] == -mag]
            if pos and neg:
                effect = np.mean(pos) - np.mean(neg)
                stratum_effects[(stratum, mag)] = effect
                print(f"    |m|={mag:.3f}: P(+m)={np.mean(pos):.3f}, P(-m)={np.mean(neg):.3f}, effect={effect:+.3f}")
    print()

    # Plot stratum effects
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"borderline": "#e74c3c", "moderate": "#f39c12", "decisive": "#27ae60"}
    for stratum in strata:
        mags = []
        effects = []
        for mag in magnitudes:
            if (stratum, mag) in stratum_effects:
                mags.append(mag)
                effects.append(stratum_effects[(stratum, mag)])
        if mags:
            ax.plot(mags, effects, "o-", label=stratum, color=colors[stratum], markersize=8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("|Multiplier|")
    ax.set_ylabel("Steering effect (pp)")
    ax.set_title("Steering effect by |delta_mu| stratum")
    ax.set_ylim(-0.3, 0.3)
    ax.legend()
    plt.tight_layout()
    plot_path = ASSETS_DIR / f"plot_031426_stratum_effects{suffix}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")

    # ── 6. Success criteria ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. SUCCESS CRITERIA")
    print("=" * 60)

    # Criterion 1: Monotonic dose-response
    mono = spearman_r > 0 and spearman_p < 0.05
    print(f"  1. Monotonic dose-response (Spearman r > 0, p < 0.05): {'PASS' if mono else 'FAIL'}")
    print(f"     r = {spearman_r:.3f}, p = {spearman_p:.4f}")

    # Criterion 2: Steering effect > 10pp at any magnitude
    max_effect = max((abs(v[0]) for v in steering_effects.values()), default=0)
    effect_pass = max_effect > 0.10
    print(f"  2. Steering effect > 10pp at any magnitude: {'PASS' if effect_pass else 'FAIL'}")
    print(f"     Max effect: {max_effect:.3f}")

    # Criterion 3: Ordering bias shifts
    control_bias = None
    pos_biases = []
    neg_biases = []
    for m in multipliers:
        ab = [r for r in valid_records if r["multiplier"] == m and r["ordering"] == 0]
        ba = [r for r in valid_records if r["multiplier"] == m and r["ordering"] == 1]
        if ab and ba:
            p_a_ab = sum(r["chose_a_position"] for r in ab) / len(ab)
            p_a_ba = sum(r["chose_a_position"] for r in ba) / len(ba)
            bias = p_a_ab - p_a_ba
            if m == 0:
                control_bias = bias
            elif m > 0:
                pos_biases.append(bias)
            else:
                neg_biases.append(bias)

    if control_bias is not None and pos_biases and neg_biases:
        bias_pass = np.mean(pos_biases) > control_bias and np.mean(neg_biases) < control_bias
        print(f"  3. Ordering bias shifts (pos > control > neg): {'PASS' if bias_pass else 'FAIL'}")
        print(f"     Control: {control_bias:+.3f}, Mean pos: {np.mean(pos_biases):+.3f}, Mean neg: {np.mean(neg_biases):+.3f}")
    else:
        bias_pass = False
        print(f"  3. Ordering bias shifts: FAIL (insufficient data)")

    # Criterion 4: Parse rates > 90%
    all_rates = []
    for m in multipliers:
        total = sum(1 for r in records if r["multiplier"] == m)
        parsed = sum(1 for r in records if r["multiplier"] == m and r["chose_high_mu"] is not None)
        rate = parsed / total if total > 0 else 0
        all_rates.append(rate)
    parse_pass = all(r > 0.90 for r in all_rates)
    print(f"  4. Parse rates > 90% at all conditions: {'PASS' if parse_pass else 'FAIL'}")
    print(f"     Min rate: {min(all_rates):.1%}")

    n_pass = sum([mono, effect_pass, bias_pass, parse_pass])
    print(f"\n  OVERALL: {n_pass}/4 criteria passed")

    return {
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "max_effect": max_effect,
        "dose_data": dose_data,
        "steering_effects": steering_effects,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    suffix = "_pilot" if args.pilot else ""
    checkpoint_path = EXPERIMENT_DIR / f"checkpoint{suffix}.jsonl"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    records = load_records(checkpoint_path)
    analyze(records, suffix)


if __name__ == "__main__":
    main()
