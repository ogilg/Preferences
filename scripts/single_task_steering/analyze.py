"""Analyze single-task steering experiment results.

Computes position-controlled P(pick steered task) and regression analyses.

Critical: Position-controlled combination works by TASK IDENTITY, not position.
For +probe on task X:
  - "boost_a, original" → task X is in position A → P(A) measures effect on X
  - "boost_b, swapped" → task X is in position B → 1-P(A) measures effect on X
Average these to get position-controlled P(pick steered task X).

Usage:
    python scripts/single_task_steering/analyze.py --steering-results <path> [--screening-results <path>]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_p_a(trials: list[dict]) -> float | None:
    """Compute P(chose A) from a list of trials, ignoring parse failures."""
    valid = [t for t in trials if t["choice"] != "parse_fail"]
    if not valid:
        return None
    return sum(1 for t in valid if t["choice"] == "a") / len(valid)


def compute_parse_rate(trials: list[dict]) -> float:
    if not trials:
        return 0.0
    return sum(1 for t in trials if t["choice"] != "parse_fail") / len(trials)


# ── Position-controlled analysis ────────────────────────────────────────

def analyze_position_controlled(trials: list[dict], pairs_meta: dict[int, dict]) -> dict:
    """Compute position-controlled P(pick steered task) for boost and suppress conditions.

    For each pair, the "original" ordering has task_x in A and task_y in B.
    The "swapped" ordering has task_y in A and task_x in B.

    For +probe on task X:
      - boost_a + original: steers position A which contains task X → P(A)
      - boost_b + swapped: steers position B which contains task X → 1-P(A)
      Average = position-controlled P(pick X when boosted)

    For +probe on task Y:
      - boost_a + swapped: steers position A which contains task Y → P(A)
      - boost_b + original: steers position B which contains task Y → 1-P(A)
      Average = position-controlled P(pick Y when boosted)

    Similarly for suppress:
      - suppress_a + original: -probe on position A (task X) → P(A) measures how X does
      - suppress_b + swapped: -probe on position B (task X) → 1-P(A) measures how X does
    """
    # Group trials by (pair_id, condition, coefficient, ordering)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for t in trials:
        key = (t["pair_id"], t["condition"], t["coefficient"], t["ordering"])
        grouped[key].append(t)

    results = {}

    # For boost conditions: compute position-controlled P(pick boosted task)
    for condition_pair, label in [
        (("boost_a", "boost_b"), "boost"),
        (("suppress_a", "suppress_b"), "suppress"),
        (("diff_ab", "diff_ba"), "differential"),
    ]:
        cond_steer_a_position, cond_steer_b_position = condition_pair

        # Collect per-(pair, coef) position-controlled estimates
        by_coef: dict[int, list[float]] = defaultdict(list)

        pair_ids = set(t["pair_id"] for t in trials)
        coefs = sorted(set(t["coefficient"] for t in trials))

        for pair_id in pair_ids:
            for coef in coefs:
                # Steered task = task_x (original task_a)
                # Case 1: steer position A in original ordering → task X boosted in position A
                trials_steer_x_posA = grouped.get((pair_id, cond_steer_a_position, coef, "original"), [])
                # Case 2: steer position B in swapped ordering → task X boosted in position B
                trials_steer_x_posB = grouped.get((pair_id, cond_steer_b_position, coef, "swapped"), [])

                p_a_posA = compute_p_a(trials_steer_x_posA)
                p_a_posB = compute_p_a(trials_steer_x_posB)

                if p_a_posA is not None and p_a_posB is not None:
                    # In posA case, task X is in A, so P(pick X) = P(A)
                    # In posB case, task X is in B, so P(pick X) = 1 - P(A)
                    p_pick_x = (p_a_posA + (1 - p_a_posB)) / 2
                    by_coef[coef].append(p_pick_x)

                # Steered task = task_y (original task_b)
                # Case 1: steer position A in swapped ordering → task Y boosted in position A
                trials_steer_y_posA = grouped.get((pair_id, cond_steer_a_position, coef, "swapped"), [])
                # Case 2: steer position B in original ordering → task Y boosted in position B
                trials_steer_y_posB = grouped.get((pair_id, cond_steer_b_position, coef, "original"), [])

                p_a_posA_y = compute_p_a(trials_steer_y_posA)
                p_a_posB_y = compute_p_a(trials_steer_y_posB)

                if p_a_posA_y is not None and p_a_posB_y is not None:
                    p_pick_y = (p_a_posA_y + (1 - p_a_posB_y)) / 2
                    by_coef[coef].append(p_pick_y)

        # Aggregate across pairs at each coefficient
        agg = {}
        for coef in sorted(by_coef.keys()):
            vals = by_coef[coef]
            agg[coef] = {
                "mean": float(np.mean(vals)),
                "se": float(np.std(vals) / np.sqrt(len(vals))),
                "n": len(vals),
            }

        # Linear regression: P(pick steered) vs coefficient
        all_coefs = []
        all_ps = []
        for coef, vals in by_coef.items():
            for v in vals:
                all_coefs.append(coef)
                all_ps.append(v)

        if len(all_coefs) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(all_coefs, all_ps)
            regression = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "std_err": float(std_err),
                "n": len(all_coefs),
            }
        else:
            regression = None

        results[label] = {
            "by_coefficient": agg,
            "regression": regression,
        }

    return results


def analyze_per_ordering(trials: list[dict]) -> dict:
    """Analyze slopes separately per ordering to decompose task-dependent vs position-sensitive."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for t in trials:
        key = (t["pair_id"], t["condition"], t["coefficient"], t["ordering"])
        grouped[key].append(t)

    results = {}

    for condition in ["boost_a", "boost_b"]:
        for ordering in ["original", "swapped"]:
            by_coef: dict[int, list[float]] = defaultdict(list)
            pair_ids = set(t["pair_id"] for t in trials)
            coefs = sorted(set(t["coefficient"] for t in trials))

            for pair_id in pair_ids:
                for coef in coefs:
                    ts = grouped.get((pair_id, condition, coef, ordering), [])
                    p_a = compute_p_a(ts)
                    if p_a is not None:
                        by_coef[coef].append(p_a)

            all_coefs = []
            all_ps = []
            for coef, vals in by_coef.items():
                for v in vals:
                    all_coefs.append(coef)
                    all_ps.append(v)

            if len(all_coefs) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(all_coefs, all_ps)
            else:
                slope = intercept = r_value = p_value = std_err = float("nan")

            results[f"{condition}_{ordering}"] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "n_observations": len(all_coefs),
            }

    # Decomposition
    boost_a_orig = results.get("boost_a_original", {}).get("slope", 0)
    boost_b_orig = results.get("boost_b_original", {}).get("slope", 0)
    # boost_a steers position A: in original, position A has task X
    # To measure effect on steered task, boost_b in original measures P(B) increase
    # which is 1-P(A) decrease, so slope should be negated
    # Actually: boost_a_original slope is dP(A)/dcoef when boosting position A (contains task X)
    # boost_b_original slope is dP(A)/dcoef when boosting position B (contains task Y)
    # For task X: effect in position A = boost_a_original slope (positive = X chosen more)
    #             effect in position B = -(boost_b_swapped slope) (flip because X is in B)
    # For the decomposition described in spec:
    # slope_boost_A = boost_a slope (across orderings) = how much P(A) changes when A is boosted
    # slope_boost_B = -(boost_b slope) = how much P(steered task) changes when boosted task is in B

    # Simpler: just report the raw slopes and let the report interpret
    results["decomposition"] = {
        "boost_a_original_slope": boost_a_orig,
        "boost_b_original_slope": boost_b_orig,
        "note": "boost_a_original: dP(A)/dcoef when +probe on position A in original ordering. "
                "boost_b_original: dP(A)/dcoef when +probe on position B in original ordering. "
                "For steered task X in position B, the effect is -(boost_b slope) since P(X)=1-P(A).",
    }

    return results


def analyze_per_pair_slopes(trials: list[dict]) -> list[dict]:
    """Compute per-pair regression slopes for the boost condition."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for t in trials:
        key = (t["pair_id"], t["condition"], t["coefficient"], t["ordering"])
        grouped[key].append(t)

    pair_ids = sorted(set(t["pair_id"] for t in trials))
    coefs = sorted(set(t["coefficient"] for t in trials))

    pair_slopes = []
    for pair_id in pair_ids:
        xs, ys = [], []
        for coef in coefs:
            # Position-controlled P(pick steered) for task X
            t_posA = grouped.get((pair_id, "boost_a", coef, "original"), [])
            t_posB = grouped.get((pair_id, "boost_b", coef, "swapped"), [])
            p_a_posA = compute_p_a(t_posA)
            p_a_posB = compute_p_a(t_posB)
            if p_a_posA is not None and p_a_posB is not None:
                p_pick = (p_a_posA + (1 - p_a_posB)) / 2
                xs.append(coef)
                ys.append(p_pick)

        if len(xs) >= 3:
            slope, intercept, r, p, se = stats.linregress(xs, ys)
            pair_slopes.append({
                "pair_id": pair_id,
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r**2),
                "p_value": float(p),
                "n_coefs": len(xs),
            })

    return pair_slopes


def analyze_internal_consistency(trials: list[dict]) -> dict:
    """Check that negative end of boost_a mirrors positive end of suppress_a."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for t in trials:
        key = (t["condition"], t["coefficient"], t["ordering"])
        grouped[key].append(t)

    comparisons = []
    for ordering in ["original", "swapped"]:
        # boost_a at coef=-3000 should ≈ suppress_a at coef=+3000
        # Both apply -probe to position A's tokens
        boost_neg = grouped.get(("boost_a", -3000, ordering), [])
        suppress_pos = grouped.get(("suppress_a", 3000, ordering), [])

        p_a_boost_neg = compute_p_a(boost_neg)
        p_a_suppress_pos = compute_p_a(suppress_pos)

        if p_a_boost_neg is not None and p_a_suppress_pos is not None:
            comparisons.append({
                "ordering": ordering,
                "boost_a_at_neg3000_P_A": float(p_a_boost_neg),
                "suppress_a_at_pos3000_P_A": float(p_a_suppress_pos),
                "difference": float(abs(p_a_boost_neg - p_a_suppress_pos)),
            })

    return {"comparisons": comparisons}


def analyze_screening(screening_data: dict) -> dict:
    """Analyze screening results: P(A) distribution, borderline rates."""
    trials = screening_data["trials"]

    # Group by (pair_id, ordering)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for t in trials:
        grouped[(t["pair_id"], t["ordering"])].append(t)

    p_a_values = []
    for (pair_id, ordering), ts in grouped.items():
        p_a = compute_p_a(ts)
        if p_a is not None:
            p_a_values.append({
                "pair_id": pair_id,
                "ordering": ordering,
                "p_a": float(p_a),
                "n_valid": sum(1 for t in ts if t["choice"] != "parse_fail"),
            })

    parse_rate = compute_parse_rate(trials)

    return {
        "n_pairs": screening_data["n_pairs"],
        "n_borderline": screening_data["n_borderline"],
        "borderline_rate": screening_data["n_borderline"] / screening_data["n_pairs"],
        "parse_rate": float(parse_rate),
        "p_a_distribution": p_a_values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering-results", type=Path)
    parser.add_argument("--screening-results", type=Path)
    parser.add_argument("--output", type=Path, help="Output path for analysis JSON")
    args = parser.parse_args()

    analysis = {}

    if args.screening_results:
        print("Analyzing screening results...")
        screening_data = load_results(args.screening_results)
        analysis["screening"] = analyze_screening(screening_data)
        print(f"  Pairs: {analysis['screening']['n_pairs']}")
        print(f"  Borderline: {analysis['screening']['n_borderline']} ({analysis['screening']['borderline_rate']:.1%})")
        print(f"  Parse rate: {analysis['screening']['parse_rate']:.1%}")

    if args.steering_results:
        print("Analyzing steering results...")
        steering_data = load_results(args.steering_results)
        trials = steering_data["trials"]

        # Load pairs metadata for reference
        pairs_path = Path("scripts/single_task_steering/pairs.json")
        with open(pairs_path) as f:
            pairs = json.load(f)
        pairs_meta = {p["pair_id"]: p for p in pairs}

        print("  Computing position-controlled analysis...")
        analysis["position_controlled"] = analyze_position_controlled(trials, pairs_meta)

        for label, data in analysis["position_controlled"].items():
            reg = data.get("regression")
            if reg:
                print(f"  {label}: slope={reg['slope']:.6f}, p={reg['p_value']:.4f}, R²={reg['r_squared']:.4f}")

        print("  Computing per-ordering analysis...")
        analysis["per_ordering"] = analyze_per_ordering(trials)

        print("  Computing per-pair slopes...")
        analysis["per_pair_slopes"] = analyze_per_pair_slopes(trials)
        slopes = [p["slope"] for p in analysis["per_pair_slopes"]]
        if slopes:
            print(f"  Per-pair slopes: mean={np.mean(slopes):.6f}, median={np.median(slopes):.6f}")
            # t-test that mean slope > 0
            t_stat, p_val = stats.ttest_1samp(slopes, 0)
            print(f"  t-test (slope > 0): t={t_stat:.3f}, p={p_val:.4f}")

        print("  Checking internal consistency...")
        analysis["internal_consistency"] = analyze_internal_consistency(trials)

        # Parse rate
        parse_rate = compute_parse_rate(trials)
        analysis["overall_parse_rate"] = float(parse_rate)
        print(f"  Overall parse rate: {parse_rate:.1%}")

    # Save analysis
    output_path = args.output or Path("results/experiments/single_task_steering/analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()
