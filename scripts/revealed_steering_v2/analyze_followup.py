"""Analyze revealed steering v2 follow-up results.

Outputs:
  - experiments/revealed_steering_v2/followup/analysis_results.json
  - Plots to experiments/revealed_steering_v2/followup/assets/
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FOLLOWUP_DIR = Path("experiments/revealed_steering_v2/followup")
CHECKPOINT_PATH = FOLLOWUP_DIR / "checkpoint.jsonl"
PAIRS_500_PATH = FOLLOWUP_DIR / "pairs_500.json"
ASSETS_DIR = FOLLOWUP_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

MULTIPLIERS = [-0.15, -0.10, -0.07, -0.05, -0.03, -0.02, -0.01,
               0.0,
               0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]


def load_records():
    records = []
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_pairs():
    with open(PAIRS_500_PATH) as f:
        return json.load(f)


# ── 1. Aggregate dose-response ──────────────────────────────────────────────

def compute_dose_response(records, pair_subset=None):
    """Compute per-multiplier stats. If pair_subset given, filter to those pair_ids."""
    steering = [r for r in records if r["condition"] == "probe"]
    if pair_subset is not None:
        steering = [r for r in steering if r["pair_id"] in pair_subset]

    by_mult = defaultdict(lambda: {"ab": [], "ba": [], "all": []})

    for r in steering:
        if r["choice_original"] is None:
            continue
        chose_a = 1 if r["choice_original"] == "a" else 0
        mult = r["multiplier"]
        by_mult[mult]["all"].append(chose_a)
        if r["ordering"] == 0:
            by_mult[mult]["ab"].append(chose_a)
        else:
            by_mult[mult]["ba"].append(chose_a)

    results = []
    for mult in sorted(by_mult.keys()):
        d = by_mult[mult]
        ab_pa = np.mean(d["ab"]) if d["ab"] else float("nan")
        ba_pa = np.mean(d["ba"]) if d["ba"] else float("nan")
        overall_pa = np.mean(d["all"]) if d["all"] else float("nan")
        ord_diff = ab_pa - ba_pa
        n_valid = len(d["all"])
        n_total_ab = len(d["ab"])
        n_total_ba = len(d["ba"])

        # Bootstrap CI on ordering difference
        ci_lo, ci_hi = bootstrap_ordering_diff_ci(d["ab"], d["ba"])

        results.append({
            "multiplier": mult,
            "p_a": round(overall_pa, 4),
            "ab_p_a": round(ab_pa, 4),
            "ba_p_a": round(ba_pa, 4),
            "ord_diff": round(ord_diff, 4),
            "ord_diff_ci": [round(ci_lo, 4), round(ci_hi, 4)],
            "n_valid": n_valid,
            "n_ab": n_total_ab,
            "n_ba": n_total_ba,
            "parse_rate": round(n_valid / (n_total_ab + n_total_ba + (n_valid == 0)) * 1, 4),
        })

    return results


def bootstrap_ordering_diff_ci(ab_choices, ba_choices, n_boot=10000, ci=0.95):
    rng = np.random.default_rng(42)
    ab = np.array(ab_choices)
    ba = np.array(ba_choices)
    diffs = []
    for _ in range(n_boot):
        ab_sample = rng.choice(ab, size=len(ab), replace=True)
        ba_sample = rng.choice(ba, size=len(ba), replace=True)
        diffs.append(ab_sample.mean() - ba_sample.mean())
    alpha = (1 - ci) / 2
    return float(np.percentile(diffs, 100 * alpha)), float(np.percentile(diffs, 100 * (1 - alpha)))


def compute_steering_effects(dose_response):
    baseline = next((d for d in dose_response if d["multiplier"] == 0.0), None)
    if baseline is None:
        return []
    baseline_diff = baseline["ord_diff"]
    effects = []
    for d in dose_response:
        effect = (d["ord_diff"] - baseline_diff) / 2
        effects.append({
            "multiplier": d["multiplier"],
            "ord_diff": d["ord_diff"],
            "steering_effect": round(effect, 4),
        })
    return effects


# ── 2. Baseline resolution ──────────────────────────────────────────────────

def analyze_baseline(records, pairs):
    baseline_recs = [r for r in records if r["condition"] == "baseline"]
    pair_choices = defaultdict(lambda: {"a": 0, "total": 0})
    for r in baseline_recs:
        if r["choice_original"] is not None:
            pair_choices[r["pair_id"]]["total"] += 1
            if r["choice_original"] == "a":
                pair_choices[r["pair_id"]]["a"] += 1

    pa_values = []
    for pid, counts in pair_choices.items():
        if counts["total"] > 0:
            pa_values.append(counts["a"] / counts["total"])

    pa_arr = np.array(pa_values)
    at_extreme = np.sum((pa_arr == 0.0) | (pa_arr == 1.0))
    n_unique = len(set(np.round(pa_arr, 4)))

    return {
        "n_pairs_with_baseline": len(pa_values),
        "n_at_extreme": int(at_extreme),
        "pct_at_extreme": round(100 * at_extreme / len(pa_values), 1),
        "n_unique_pa_values": n_unique,
        "mean_pa": round(float(np.mean(pa_arr)), 4),
        "std_pa": round(float(np.std(pa_arr)), 4),
        "pa_distribution": pa_values,
    }


# ── 3. Steerability vs decidedness ─────────────────────────────────────────

def compute_per_pair_steerability(records, pairs):
    """Compute steerability at mult=±0.02 and decidedness from 20-trial baseline."""
    # Get baseline P(A) per pair from checkpoint (more reliable than pairs_500)
    baseline_recs = [r for r in records if r["condition"] == "baseline"]
    pair_baseline = defaultdict(lambda: {"a": 0, "total": 0})
    for r in baseline_recs:
        if r["choice_original"] is not None:
            pair_baseline[r["pair_id"]]["total"] += 1
            if r["choice_original"] == "a":
                pair_baseline[r["pair_id"]]["a"] += 1

    # Get P(A) at mult=+0.02 and mult=-0.02 per pair
    steering_recs = [r for r in records if r["condition"] == "probe"
                     and r["multiplier"] in [0.02, -0.02]]

    pair_steered = defaultdict(lambda: defaultdict(lambda: {"a": 0, "total": 0}))
    for r in steering_recs:
        if r["choice_original"] is not None:
            pair_steered[r["pair_id"]][r["multiplier"]]["total"] += 1
            if r["choice_original"] == "a":
                pair_steered[r["pair_id"]][r["multiplier"]]["a"] += 1

    # Also compute ordering-based steerability
    steering_by_ord = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in [r for r in records if r["condition"] == "probe" and r["choice_original"] is not None]:
        chose_a = 1 if r["choice_original"] == "a" else 0
        steering_by_ord[r["pair_id"]][r["multiplier"]][r["ordering"]].append(chose_a)

    baseline_by_ord = defaultdict(lambda: defaultdict(list))
    for r in baseline_recs:
        if r["choice_original"] is not None:
            chose_a = 1 if r["choice_original"] == "a" else 0
            baseline_by_ord[r["pair_id"]][r["ordering"]].append(chose_a)

    results = []
    for pid, bl in pair_baseline.items():
        if bl["total"] == 0:
            continue
        pa_baseline = bl["a"] / bl["total"]
        decidedness = abs(pa_baseline - 0.5)

        # Steerability as max |shift in P(A)| at ±0.02
        shifts = []
        for mult in [0.02, -0.02]:
            if pid in pair_steered and mult in pair_steered[pid]:
                sd = pair_steered[pid][mult]
                if sd["total"] > 0:
                    pa_steered = sd["a"] / sd["total"]
                    shifts.append(abs(pa_steered - pa_baseline))

        steerability = max(shifts) if shifts else float("nan")

        # Ordering-based steerability at mult=+0.02
        ord_steerability = float("nan")
        if pid in steering_by_ord and 0.02 in steering_by_ord[pid]:
            ab = steering_by_ord[pid][0.02].get(0, [])
            ba = steering_by_ord[pid][0.02].get(1, [])
            bl_ab = baseline_by_ord[pid].get(0, [])
            bl_ba = baseline_by_ord[pid].get(1, [])
            if ab and ba and bl_ab and bl_ba:
                steered_diff = np.mean(ab) - np.mean(ba)
                baseline_diff = np.mean(bl_ab) - np.mean(bl_ba)
                ord_steerability = abs(steered_diff - baseline_diff) / 2

        # Determine subset
        pair_num = int(pid.split("_")[1])
        subset = "original_300" if pair_num < 300 else "new_200"

        results.append({
            "pair_id": pid,
            "pa_baseline": round(pa_baseline, 4),
            "decidedness": round(decidedness, 4),
            "steerability": round(steerability, 4) if not np.isnan(steerability) else None,
            "ord_steerability": round(ord_steerability, 4) if not np.isnan(ord_steerability) else None,
            "subset": subset,
        })

    return results


# ── 4. Plots ────────────────────────────────────────────────────────────────

COHERENCE_BOUNDARY = 0.05


def _plot_faded_line(ax, mults, vals, color, label=None, marker="o", linestyle="-",
                     base_alpha=1.0, ci_lo=None, ci_hi=None):
    """Plot a line with points outside ±COHERENCE_BOUNDARY faded."""
    mults = np.array(mults)
    vals = np.array(vals)
    coherent = np.abs(mults) <= COHERENCE_BOUNDARY
    left = mults < -COHERENCE_BOUNDARY
    right = mults > COHERENCE_BOUNDARY

    if ci_lo is not None and ci_hi is not None:
        ci_lo, ci_hi = np.array(ci_lo), np.array(ci_hi)
        if coherent.any():
            ax.fill_between(mults[coherent], ci_lo[coherent], ci_hi[coherent],
                            alpha=0.2 * base_alpha, color=color)
        for mask in [left, right]:
            if mask.any():
                ax.fill_between(mults[mask], ci_lo[mask], ci_hi[mask],
                                alpha=0.06 * base_alpha, color=color)

    if coherent.any():
        ax.plot(mults[coherent], vals[coherent], f"{marker}{linestyle}",
                color=color, alpha=base_alpha, label=label)
    for mask in [left, right]:
        if mask.any():
            ax.plot(mults[mask], vals[mask], f"{marker}{linestyle}",
                    color=color, alpha=0.25 * base_alpha, markersize=4)


def plot_dose_response(dose_all, dose_old, dose_new, effects_all):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Ordering difference with CIs
    ax = axes[0]
    mults = [d["multiplier"] for d in dose_all]
    ord_diffs = [d["ord_diff"] for d in dose_all]
    ci_lo = [d["ord_diff_ci"][0] for d in dose_all]
    ci_hi = [d["ord_diff_ci"][1] for d in dose_all]
    _plot_faded_line(ax, mults, ord_diffs, "C0", label="All 500", ci_lo=ci_lo, ci_hi=ci_hi)

    mults_old = [d["multiplier"] for d in dose_old]
    ord_old = [d["ord_diff"] for d in dose_old]
    _plot_faded_line(ax, mults_old, ord_old, "C1", label="Original 300",
                     marker="s", linestyle="--", base_alpha=0.7)

    mults_new = [d["multiplier"] for d in dose_new]
    ord_new = [d["ord_diff"] for d in dose_new]
    _plot_faded_line(ax, mults_new, ord_new, "C2", label="New 200",
                     marker="^", linestyle="--", base_alpha=0.7)

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=-COHERENCE_BOUNDARY, color="black", linestyle="--", alpha=0.3)
    ax.axvline(x=COHERENCE_BOUNDARY, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Steering multiplier")
    ax.set_ylabel("Ordering difference: P(A|AB) - P(A|BA)")
    ax.set_title("Ordering difference by multiplier")
    ax.set_ylim(-0.1, 0.7)
    ax.legend()

    # Plot 2: Derived steering effect
    ax = axes[1]
    eff_mults = [e["multiplier"] for e in effects_all]
    eff_vals = [e["steering_effect"] for e in effects_all]
    _plot_faded_line(ax, eff_mults, eff_vals, "C0")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=-COHERENCE_BOUNDARY, color="black", linestyle="--", alpha=0.3)
    ax.axvline(x=COHERENCE_BOUNDARY, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Steering multiplier")
    ax.set_ylabel("Derived steering effect")
    ax.set_title("Steering effect = (ord_diff - baseline_diff) / 2")
    ax.set_ylim(-0.25, 0.25)

    # Plot 3: Parse rates
    ax = axes[2]
    n_valid = [d["n_valid"] for d in dose_all]
    expected = [d["n_ab"] + d["n_ba"] for d in dose_all]
    parse_rates = [v / e if e > 0 else 0 for v, e in zip(n_valid, expected)]
    _plot_faded_line(ax, mults, parse_rates, "C3")
    ax.set_xlabel("Steering multiplier")
    ax.set_ylabel("Parse rate")
    ax.set_title("Parse rate by multiplier")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_030326_dose_response.png", dpi=150)
    plt.close()
    print("Saved dose_response plot")


def plot_baseline_distribution(baseline_info):
    fig, ax = plt.subplots(figsize=(8, 5))
    pa_vals = baseline_info["pa_distribution"]
    ax.hist(pa_vals, bins=21, range=(0, 1), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Baseline P(A) (20 trials)")
    ax.set_ylabel("Number of pairs")
    ax.set_title(f"Baseline P(A) distribution (n={len(pa_vals)}, "
                 f"{baseline_info['pct_at_extreme']}% at 0 or 1)")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_030326_baseline_distribution.png", dpi=150)
    plt.close()
    print("Saved baseline_distribution plot")


def plot_steerability_vs_decidedness(per_pair):
    valid = [p for p in per_pair if p["steerability"] is not None]
    decidedness = np.array([p["decidedness"] for p in valid])
    steerability = np.array([p["steerability"] for p in valid])

    # Color by subset
    colors = ["C1" if p["subset"] == "original_300" else "C2" for p in valid]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    ax = axes[0]
    for subset, color, label in [("original_300", "C1", "Original 300"), ("new_200", "C2", "New 200")]:
        mask = np.array([p["subset"] == subset for p in valid])
        ax.scatter(decidedness[mask], steerability[mask], c=color, alpha=0.3, s=15, label=label)
    r = np.corrcoef(decidedness, steerability)[0, 1]
    ax.set_xlabel("Decidedness = |P(A) - 0.5|")
    ax.set_ylabel("Steerability = max |ΔP(A)| at mult=±0.02")
    ax.set_title(f"Steerability vs decidedness (r={r:.3f})")
    ax.set_xlim(0, 0.55)
    ax.set_ylim(0, 1.0)
    ax.legend()

    # Binned
    ax = axes[1]
    bins = np.linspace(0, 0.5, 6)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for subset, color, label in [("original_300", "C1", "Original 300"), ("new_200", "C2", "New 200")]:
        mask = np.array([p["subset"] == subset for p in valid])
        d_sub = decidedness[mask]
        s_sub = steerability[mask]
        bin_means = []
        bin_sems = []
        for i in range(len(bins) - 1):
            in_bin = (d_sub >= bins[i]) & (d_sub < bins[i + 1])
            if in_bin.sum() > 0:
                bin_means.append(s_sub[in_bin].mean())
                bin_sems.append(s_sub[in_bin].std() / np.sqrt(in_bin.sum()))
            else:
                bin_means.append(float("nan"))
                bin_sems.append(float("nan"))
        offset = -0.005 if subset == "original_300" else 0.005
        ax.errorbar(bin_centers + offset, bin_means, yerr=bin_sems,
                    fmt="o-", color=color, label=label, capsize=3)

    ax.set_xlabel("Decidedness = |P(A) - 0.5|")
    ax.set_ylabel("Mean steerability ± SEM")
    ax.set_title("Binned steerability by decidedness")
    ax.set_xlim(0, 0.55)
    ax.set_ylim(0, 0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_030326_steerability_vs_decidedness.png", dpi=150)
    plt.close()
    print("Saved steerability_vs_decidedness plot")


def plot_subset_comparison(per_pair):
    valid = [p for p in per_pair if p["steerability"] is not None]
    old = [p["steerability"] for p in valid if p["subset"] == "original_300"]
    new = [p["steerability"] for p in valid if p["subset"] == "new_200"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(old, bins=20, range=(0, 1), alpha=0.5, label=f"Original 300 (mean={np.mean(old):.3f})", color="C1")
    ax.hist(new, bins=20, range=(0, 1), alpha=0.5, label=f"New 200 (mean={np.mean(new):.3f})", color="C2")
    ax.set_xlabel("Steerability = max |ΔP(A)| at mult=±0.02")
    ax.set_ylabel("Number of pairs")
    ax.set_title("Steerability: original borderline pairs vs unrestricted pairs")
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_030326_subset_comparison.png", dpi=150)
    plt.close()
    print("Saved subset_comparison plot")


def main():
    print("Loading data...")
    records = load_records()
    pairs = load_pairs()
    print(f"Records: {len(records)}, Pairs: {len(pairs)}")

    # Identify subsets
    old_pids = {f"pair_{i:04d}" for i in range(300)}
    new_pids = {f"pair_{i:04d}" for i in range(300, 500)}

    # 1. Dose-response
    print("\n=== Dose-response ===")
    dose_all = compute_dose_response(records)
    dose_old = compute_dose_response(records, pair_subset=old_pids)
    dose_new = compute_dose_response(records, pair_subset=new_pids)
    effects_all = compute_steering_effects(dose_all)

    print(f"\n{'mult':>7} | {'P(A)':>6} | {'AB P(A)':>7} | {'BA P(A)':>7} | {'ord.diff':>8} | {'95% CI':>16} | {'effect':>7} | {'N':>6}")
    print("-" * 85)
    for d, e in zip(dose_all, effects_all):
        ci = f"[{d['ord_diff_ci'][0]:.3f}, {d['ord_diff_ci'][1]:.3f}]"
        print(f"{d['multiplier']:>+7.3f} | {d['p_a']:>6.3f} | {d['ab_p_a']:>7.3f} | {d['ba_p_a']:>7.3f} | "
              f"{d['ord_diff']:>+8.3f} | {ci:>16} | {e['steering_effect']:>+7.3f} | {d['n_valid']:>6}")

    # Old vs new comparison at key multipliers
    print("\n=== Old 300 vs New 200 at key multipliers ===")
    for mult in [-0.02, 0.0, 0.02]:
        d_old = next((d for d in dose_old if d["multiplier"] == mult), None)
        d_new = next((d for d in dose_new if d["multiplier"] == mult), None)
        if d_old and d_new:
            print(f"mult={mult:+.2f}: Old ord.diff={d_old['ord_diff']:+.3f}, New ord.diff={d_new['ord_diff']:+.3f}")

    # 2. Baseline resolution
    print("\n=== Baseline resolution ===")
    baseline_info = analyze_baseline(records, pairs)
    print(f"Pairs with baseline: {baseline_info['n_pairs_with_baseline']}")
    print(f"At extreme (0 or 1): {baseline_info['n_at_extreme']} ({baseline_info['pct_at_extreme']}%)")
    print(f"Unique P(A) values: {baseline_info['n_unique_pa_values']}")
    print(f"Mean P(A): {baseline_info['mean_pa']}, Std: {baseline_info['std_pa']}")

    # 3. Steerability vs decidedness
    print("\n=== Steerability vs decidedness ===")
    per_pair = compute_per_pair_steerability(records, pairs)
    valid = [p for p in per_pair if p["steerability"] is not None]
    decidedness = np.array([p["decidedness"] for p in valid])
    steerability = np.array([p["steerability"] for p in valid])
    r = np.corrcoef(decidedness, steerability)[0, 1]
    print(f"Pairs with steerability: {len(valid)}")
    print(f"Pearson r(decidedness, steerability): {r:.3f}")
    print(f"Mean steerability: {np.mean(steerability):.3f}")

    # Subset comparison
    old_steer = [p["steerability"] for p in valid if p["subset"] == "original_300"]
    new_steer = [p["steerability"] for p in valid if p["subset"] == "new_200"]
    print(f"\nOriginal 300: mean steerability={np.mean(old_steer):.3f} (n={len(old_steer)})")
    print(f"New 200: mean steerability={np.mean(new_steer):.3f} (n={len(new_steer)})")

    # 4. Plots
    print("\n=== Generating plots ===")
    plot_dose_response(dose_all, dose_old, dose_new, effects_all)
    plot_baseline_distribution(baseline_info)
    plot_steerability_vs_decidedness(per_pair)
    plot_subset_comparison(per_pair)

    # 5. Save analysis results
    analysis = {
        "dose_response_all": dose_all,
        "dose_response_old_300": dose_old,
        "dose_response_new_200": dose_new,
        "steering_effects": effects_all,
        "baseline": {k: v for k, v in baseline_info.items() if k != "pa_distribution"},
        "per_pair_summary": {
            "n_valid": len(valid),
            "r_decidedness_steerability": round(r, 4),
            "mean_steerability": round(float(np.mean(steerability)), 4),
            "mean_decidedness": round(float(np.mean(decidedness)), 4),
            "old_300_mean_steerability": round(float(np.mean(old_steer)), 4),
            "new_200_mean_steerability": round(float(np.mean(new_steer)), 4),
        },
    }
    out_path = FOLLOWUP_DIR / "analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis to {out_path}")


if __name__ == "__main__":
    main()
