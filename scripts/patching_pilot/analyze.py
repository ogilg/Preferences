"""Analysis for activation patching pilot experiment."""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

EXPERIMENT_DIR = Path("experiments/patching/pilot")
RESULTS_PATH = EXPERIMENT_DIR / "results.json"
BASELINE_PATH = EXPERIMENT_DIR / "baseline_p_choose.json"


def load_data():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    with open(BASELINE_PATH) as f:
        baselines = json.load(f)
    baseline_lookup = {}
    for b in baselines:
        baseline_lookup[(b["task_a_id"], b["task_b_id"])] = b
    return results, baseline_lookup


def compute_pair_stats(results, baseline_lookup):
    """Aggregate results by canonical pair (ignoring ordering)."""
    pair_data = defaultdict(lambda: {"baseline": [], "last_token_swap": [], "span_swap": []})

    for r in results:
        if r["ordering"] == "AB":
            # task_a_id is the canonical first, task_b_id is canonical second
            canonical_key = (r["task_a_id"], r["task_b_id"])
        else:
            # In BA ordering: task_a_id = canonical second, task_b_id = canonical first
            canonical_key = (r["task_b_id"], r["task_a_id"])

        for cond in ["baseline", "last_token_swap", "span_swap"]:
            choices = r["conditions"][cond]["choices"]
            for c in choices:
                if c == "parse_fail":
                    continue
                if r["ordering"] == "AB":
                    # "a" = chose canonical first, "b" = chose canonical second
                    pair_data[canonical_key][cond].append(c)
                else:
                    # "a" = chose position A = canonical second → map to "b"
                    # "b" = chose position B = canonical first → map to "a"
                    flipped = "b" if c == "a" else "a"
                    pair_data[canonical_key][cond].append(flipped)

    return dict(pair_data)


def p_choose_b(choices):
    """Fraction choosing 'b' (higher utility task in canonical ordering)."""
    if not choices:
        return float("nan")
    return choices.count("b") / len(choices)


def analyze_position_bias(results):
    """Analyze how often the model picks position A vs B in baseline."""
    n_a, n_b, total = 0, 0, 0
    for r in results:
        choices = r["conditions"]["baseline"]["choices"]
        n_a += choices.count("a")
        n_b += choices.count("b")
        total += len(choices)
    return {"n_a": n_a, "n_b": n_b, "total": total, "p_a": n_a / total}


def analyze_flips(pair_data, baseline_lookup):
    """Compute flip rates and statistics for each condition."""
    conditions = ["last_token_swap", "span_swap"]
    flip_analysis = {}

    for cond in conditions:
        flips = []
        shifts = []
        for (ta, tb), data in sorted(pair_data.items()):
            base_choices = data["baseline"]
            cond_choices = data[cond]
            if not base_choices or not cond_choices:
                continue

            p_b_base = p_choose_b(base_choices)
            p_b_cond = p_choose_b(cond_choices)
            shift = p_b_cond - p_b_base

            # Binomial test: is the condition distribution different from baseline?
            n_b_cond = cond_choices.count("b")
            n_total = len(cond_choices)
            # Test against baseline p(b)
            if p_b_base in (0.0, 1.0):
                # Can still test: if baseline is 0, any b is a shift
                p_val = stats.binomtest(n_b_cond, n_total, max(p_b_base, 0.01)).pvalue
            else:
                p_val = stats.binomtest(n_b_cond, n_total, p_b_base).pvalue

            # Get Thurstonian baseline
            bl = baseline_lookup.get((ta, tb))
            delta_mu = bl["delta_mu"] if bl else float("nan")
            p_thurstone = bl["p_b_over_a"] if bl else float("nan")

            is_flip = (p_b_base < 0.5 and p_b_cond > 0.5) or (p_b_base > 0.5 and p_b_cond < 0.5)

            flips.append({
                "task_a": ta,
                "task_b": tb,
                "p_b_baseline": round(p_b_base, 3),
                "p_b_cond": round(p_b_cond, 3),
                "shift": round(shift, 3),
                "p_value": round(p_val, 4),
                "is_flip": is_flip,
                "delta_mu": round(delta_mu, 2),
                "p_thurstone": round(p_thurstone, 4),
                "n_trials": n_total,
            })
            shifts.append(shift)

        n_flips = sum(1 for f in flips if f["is_flip"])
        n_sig_shifts = sum(1 for f in flips if f["p_value"] < 0.05)
        mean_shift = np.mean(shifts) if shifts else 0
        mean_abs_shift = np.mean(np.abs(shifts)) if shifts else 0

        flip_analysis[cond] = {
            "flips": flips,
            "n_pairs": len(flips),
            "n_flips": n_flips,
            "flip_rate": round(n_flips / len(flips), 3) if flips else 0,
            "n_sig_shifts": n_sig_shifts,
            "mean_shift": round(mean_shift, 4),
            "mean_abs_shift": round(mean_abs_shift, 4),
        }

    return flip_analysis


def analyze_direction_of_flips(flip_analysis):
    """Check if flips are systematic reversals (swap ≈ 1-baseline) or random corruption."""
    for cond, data in flip_analysis.items():
        reversals = []
        for f in data["flips"]:
            if f["is_flip"]:
                # Perfect reversal: p_b_cond ≈ 1 - p_b_baseline
                expected_reversal = 1 - f["p_b_baseline"]
                reversal_quality = 1 - abs(f["p_b_cond"] - expected_reversal)
                reversals.append({
                    "pair": f"{f['task_a']} vs {f['task_b']}",
                    "p_b_base": f["p_b_baseline"],
                    "p_b_cond": f["p_b_cond"],
                    "expected_reversal": round(expected_reversal, 3),
                    "reversal_quality": round(reversal_quality, 3),
                })
        data["reversals"] = reversals


def analyze_utility_gap_effect(flip_analysis, baseline_lookup):
    """Check if large utility gaps resist flipping."""
    for cond, data in flip_analysis.items():
        gaps_and_shifts = []
        for f in data["flips"]:
            gaps_and_shifts.append((abs(f["delta_mu"]), abs(f["shift"])))
        if gaps_and_shifts:
            gaps, shifts = zip(*gaps_and_shifts)
            corr, p_val = stats.spearmanr(gaps, shifts)
            data["gap_shift_correlation"] = round(corr, 4)
            data["gap_shift_p_value"] = round(p_val, 4)


def print_summary(pair_data, flip_analysis, pos_bias, baseline_lookup):
    print("=" * 70)
    print("ACTIVATION PATCHING PILOT - ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\n## Position Bias (Baseline)")
    print(f"  P(choose position A) = {pos_bias['p_a']:.3f} ({pos_bias['n_a']}/{pos_bias['total']})")

    for cond in ["last_token_swap", "span_swap"]:
        data = flip_analysis[cond]
        print(f"\n## {cond.replace('_', ' ').title()}")
        print(f"  Pairs analyzed: {data['n_pairs']}")
        print(f"  Pairs flipped: {data['n_flips']} ({data['flip_rate']*100:.1f}%)")
        print(f"  Significant shifts (p<0.05): {data['n_sig_shifts']}")
        print(f"  Mean shift: {data['mean_shift']:.4f}")
        print(f"  Mean |shift|: {data['mean_abs_shift']:.4f}")
        if "gap_shift_correlation" in data:
            print(f"  Gap-shift corr (Spearman): r={data['gap_shift_correlation']:.3f}, p={data['gap_shift_p_value']:.3f}")

        if data["reversals"]:
            print(f"\n  Flipped pairs:")
            for rev in data["reversals"]:
                print(f"    {rev['pair']}: base={rev['p_b_base']:.2f} → cond={rev['p_b_cond']:.2f} "
                      f"(expected reversal={rev['expected_reversal']:.2f}, quality={rev['reversal_quality']:.2f})")

    # Per-pair detail table
    print(f"\n## Per-Pair Detail (sorted by |Δμ|)")
    print(f"{'Pair':<55} {'|Δμ|':>5} {'P(B)_bl':>7} {'P(B)_lt':>7} {'P(B)_sp':>7}")
    print("-" * 90)
    rows = []
    for (ta, tb), data in pair_data.items():
        p_base = p_choose_b(data["baseline"])
        p_lt = p_choose_b(data["last_token_swap"])
        p_sp = p_choose_b(data["span_swap"])
        bl_entry = baseline_lookup.get((ta, tb))
        delta = abs(bl_entry["delta_mu"]) if bl_entry else 0
        rows.append((ta, tb, delta, p_base, p_lt, p_sp))
    rows.sort(key=lambda x: x[2])
    for ta, tb, delta, p_base, p_lt, p_sp in rows:
        pair_str = f"{ta[:25]} vs {tb[:25]}"
        marker_lt = "*" if abs(p_lt - p_base) > 0.3 else " "
        marker_sp = "*" if abs(p_sp - p_base) > 0.3 else " "
        print(f"{pair_str:<55} {delta:5.1f} {p_base:7.2f} {p_lt:7.2f}{marker_lt} {p_sp:7.2f}{marker_sp}")


def save_analysis(flip_analysis, pos_bias, pair_data, baseline_lookup):
    """Save analysis results to JSON."""
    output = {
        "position_bias": pos_bias,
        "conditions": {},
        "per_pair": [],
    }
    for cond in ["last_token_swap", "span_swap"]:
        d = flip_analysis[cond]
        output["conditions"][cond] = {
            "n_pairs": d["n_pairs"],
            "n_flips": d["n_flips"],
            "flip_rate": d["flip_rate"],
            "n_sig_shifts": d["n_sig_shifts"],
            "mean_shift": d["mean_shift"],
            "mean_abs_shift": d["mean_abs_shift"],
            "gap_shift_correlation": d.get("gap_shift_correlation"),
            "gap_shift_p_value": d.get("gap_shift_p_value"),
        }

    for (ta, tb), data in sorted(pair_data.items()):
        bl = baseline_lookup.get((ta, tb))
        row = {
            "task_a": ta, "task_b": tb,
            "delta_mu": bl["delta_mu"] if bl else None,
            "p_thurstone": bl["p_b_over_a"] if bl else None,
        }
        for cond in ["baseline", "last_token_swap", "span_swap"]:
            row[f"p_b_{cond}"] = round(p_choose_b(data[cond]), 3)
            row[f"n_{cond}"] = len(data[cond])
        output["per_pair"].append(row)

    with open(EXPERIMENT_DIR / "analysis.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nAnalysis saved to {EXPERIMENT_DIR / 'analysis.json'}")


def main():
    results, baseline_lookup = load_data()
    pair_data = compute_pair_stats(results, baseline_lookup)
    pos_bias = analyze_position_bias(results)
    flip_analysis = analyze_flips(pair_data, baseline_lookup)
    analyze_direction_of_flips(flip_analysis)
    analyze_utility_gap_effect(flip_analysis, baseline_lookup)
    print_summary(pair_data, flip_analysis, pos_bias, baseline_lookup)
    save_analysis(flip_analysis, pos_bias, pair_data, baseline_lookup)


if __name__ == "__main__":
    main()
