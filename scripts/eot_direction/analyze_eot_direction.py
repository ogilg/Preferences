"""Analysis for EOT vs prompt_last direction steering.

Combines EOT checkpoint with v2 followup checkpoint (baseline + mult=+/-0.03 prompt_last).
Computes ordering differences, derived steering effects, per-pair correlations, parse rates.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EOT_CHECKPOINT = Path("experiments/steering/eot_direction/checkpoint.jsonl")
V2_CHECKPOINT = Path("experiments/revealed_steering_v2/followup/checkpoint.jsonl")
PAIRS_PATH = Path("experiments/revealed_steering_v2/followup/pairs_500.json")
ASSETS_DIR = Path("experiments/steering/eot_direction/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_pairs() -> dict[str, dict]:
    with open(PAIRS_PATH) as f:
        pairs = json.load(f)
    return {p["pair_id"]: p for p in pairs}


def filter_valid_choices(records: list[dict]) -> list[dict]:
    """Keep only records with valid choices (a or b)."""
    return [r for r in records if r["choice_original"] in ("a", "b")]


def compute_pa_by_ordering(records: list[dict]) -> dict[str, dict[int, float]]:
    """Compute P(A) per pair per ordering. Returns {pair_id: {ordering: P(A)}}."""
    grouped: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["choice_original"] in ("a", "b"):
            grouped[r["pair_id"]][r["ordering"]].append(r["choice_original"] == "a")
    result = {}
    for pair_id, orderings in grouped.items():
        result[pair_id] = {}
        for ordering, choices in orderings.items():
            result[pair_id][ordering] = np.mean(choices)
    return result


def compute_ordering_diff(pa_by_ordering: dict[str, dict[int, float]]) -> dict[str, float]:
    """Compute P(A|AB) - P(A|BA) per pair."""
    diffs = {}
    for pair_id, orderings in pa_by_ordering.items():
        if 0 in orderings and 1 in orderings:
            diffs[pair_id] = orderings[0] - orderings[1]
    return diffs


def bootstrap_mean_ci(values: list[float], n_boot: int = 10000, ci: float = 0.95) -> tuple[float, float, float]:
    """Returns (mean, ci_lower, ci_upper)."""
    arr = np.array(values)
    mean = np.mean(arr)
    boot_means = np.array([
        np.mean(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    ci_lower = np.percentile(boot_means, alpha * 100)
    ci_upper = np.percentile(boot_means, (1 - alpha) * 100)
    return float(mean), float(ci_lower), float(ci_upper)


def compute_aggregate_steering_effect(
    steered_pa: dict[str, dict[int, float]],
    baseline_pa: dict[str, dict[int, float]],
) -> dict[str, float]:
    """Compute per-pair aggregate steering effect:
    effect = ((P(A|AB,steered) - P(A|AB,base)) + (P(A|BA,base) - P(A|BA,steered))) / 2
    """
    effects = {}
    for pair_id in steered_pa:
        if pair_id not in baseline_pa:
            continue
        s = steered_pa[pair_id]
        b = baseline_pa[pair_id]
        if 0 in s and 1 in s and 0 in b and 1 in b:
            ab_shift = s[0] - b[0]
            ba_shift = b[1] - s[1]
            effects[pair_id] = (ab_shift + ba_shift) / 2
    return effects


def main():
    np.random.seed(42)

    # Load data
    eot_records = load_jsonl(EOT_CHECKPOINT)
    v2_records = load_jsonl(V2_CHECKPOINT)
    pairs = load_pairs()

    print(f"EOT records: {len(eot_records)}")
    print(f"V2 records: {len(v2_records)}")

    # Filter v2 to baseline and mult=+/-0.03 (prompt_last condition)
    baseline_records = [r for r in v2_records if r["condition"] == "baseline"]
    prompt_last_pos = [r for r in v2_records if r["condition"] == "probe" and abs(r["multiplier"] - 0.03) < 1e-6]
    prompt_last_neg = [r for r in v2_records if r["condition"] == "probe" and abs(r["multiplier"] - (-0.03)) < 1e-6]

    # EOT records split by multiplier
    eot_pos = [r for r in eot_records if r["multiplier"] > 0]
    eot_neg = [r for r in eot_records if r["multiplier"] < 0]

    print(f"\nRecord counts:")
    print(f"  Baseline: {len(baseline_records)}")
    print(f"  Prompt_last +0.03: {len(prompt_last_pos)}")
    print(f"  Prompt_last -0.03: {len(prompt_last_neg)}")
    print(f"  EOT +0.03: {len(eot_pos)}")
    print(f"  EOT -0.03: {len(eot_neg)}")

    # Parse rates
    print("\n--- Parse rates ---")
    for label, recs in [("Baseline", baseline_records), ("Prompt_last +0.03", prompt_last_pos),
                         ("Prompt_last -0.03", prompt_last_neg), ("EOT +0.03", eot_pos), ("EOT -0.03", eot_neg)]:
        valid = sum(1 for r in recs if r["choice_original"] in ("a", "b"))
        total = len(recs)
        print(f"  {label}: {valid}/{total} = {valid/total:.4f}" if total > 0 else f"  {label}: no records")

    # Compute P(A) by ordering for each condition
    baseline_pa = compute_pa_by_ordering(filter_valid_choices(baseline_records))
    pl_pos_pa = compute_pa_by_ordering(filter_valid_choices(prompt_last_pos))
    pl_neg_pa = compute_pa_by_ordering(filter_valid_choices(prompt_last_neg))
    eot_pos_pa = compute_pa_by_ordering(filter_valid_choices(eot_pos))
    eot_neg_pa = compute_pa_by_ordering(filter_valid_choices(eot_neg))

    # 1. Ordering difference at +/-0.03 for both conditions
    print("\n--- Ordering difference: P(A|AB) - P(A|BA) ---")

    baseline_diffs = compute_ordering_diff(baseline_pa)
    pl_pos_diffs = compute_ordering_diff(pl_pos_pa)
    pl_neg_diffs = compute_ordering_diff(pl_neg_pa)
    eot_pos_diffs = compute_ordering_diff(eot_pos_pa)
    eot_neg_diffs = compute_ordering_diff(eot_neg_pa)

    for label, diffs in [("Baseline", baseline_diffs), ("Prompt_last +0.03", pl_pos_diffs),
                          ("Prompt_last -0.03", pl_neg_diffs), ("EOT +0.03", eot_pos_diffs),
                          ("EOT -0.03", eot_neg_diffs)]:
        vals = list(diffs.values())
        if vals:
            mean, ci_lo, ci_hi = bootstrap_mean_ci(vals)
            print(f"  {label}: mean={mean:.4f} 95% CI [{ci_lo:.4f}, {ci_hi:.4f}] (n={len(vals)})")

    # 2. Derived steering effect
    print("\n--- Aggregate steering effect: (ord_diff - baseline_diff) / 2 ---")

    # Per-pair effects
    pl_pos_effects = compute_aggregate_steering_effect(pl_pos_pa, baseline_pa)
    pl_neg_effects = compute_aggregate_steering_effect(pl_neg_pa, baseline_pa)
    eot_pos_effects = compute_aggregate_steering_effect(eot_pos_pa, baseline_pa)
    eot_neg_effects = compute_aggregate_steering_effect(eot_neg_pa, baseline_pa)

    for label, effects in [("Prompt_last +0.03", pl_pos_effects), ("Prompt_last -0.03", pl_neg_effects),
                           ("EOT +0.03", eot_pos_effects), ("EOT -0.03", eot_neg_effects)]:
        vals = list(effects.values())
        if vals:
            mean, ci_lo, ci_hi = bootstrap_mean_ci(vals)
            print(f"  {label}: mean={mean:+.4f} 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] (n={len(vals)})")

    # 3. Per-pair correlation: EOT vs prompt_last
    print("\n--- Per-pair correlation (EOT vs prompt_last) ---")

    # +0.03
    common_pos = sorted(set(eot_pos_effects.keys()) & set(pl_pos_effects.keys()))
    if len(common_pos) > 10:
        eot_vals = np.array([eot_pos_effects[p] for p in common_pos])
        pl_vals = np.array([pl_pos_effects[p] for p in common_pos])
        r = np.corrcoef(eot_vals, pl_vals)[0, 1]
        print(f"  +0.03: Pearson r = {r:.4f} (n={len(common_pos)})")

    # -0.03
    common_neg = sorted(set(eot_neg_effects.keys()) & set(pl_neg_effects.keys()))
    if len(common_neg) > 10:
        eot_vals = np.array([eot_neg_effects[p] for p in common_neg])
        pl_vals = np.array([pl_neg_effects[p] for p in common_neg])
        r = np.corrcoef(eot_vals, pl_vals)[0, 1]
        print(f"  -0.03: Pearson r = {r:.4f} (n={len(common_neg)})")

    # Combined: average across +/- for pairs with both
    common_both = sorted(set(common_pos) & set(common_neg))
    if len(common_both) > 10:
        eot_combined = np.array([(eot_pos_effects[p] - eot_neg_effects[p]) / 2 for p in common_both])
        pl_combined = np.array([(pl_pos_effects[p] - pl_neg_effects[p]) / 2 for p in common_both])
        r = np.corrcoef(eot_combined, pl_combined)[0, 1]
        print(f"  Combined (pos-neg)/2: Pearson r = {r:.4f} (n={len(common_both)})")

    # 4. Per-ordering P(A) summary
    print("\n--- Per-ordering P(A) means ---")
    for label, pa_data in [("Baseline", baseline_pa), ("Prompt_last +0.03", pl_pos_pa),
                           ("Prompt_last -0.03", pl_neg_pa), ("EOT +0.03", eot_pos_pa),
                           ("EOT -0.03", eot_neg_pa)]:
        ab_vals = [pa_data[p][0] for p in pa_data if 0 in pa_data[p]]
        ba_vals = [pa_data[p][1] for p in pa_data if 1 in pa_data[p]]
        if ab_vals and ba_vals:
            ab_mean, ab_lo, ab_hi = bootstrap_mean_ci(ab_vals)
            ba_mean, ba_lo, ba_hi = bootstrap_mean_ci(ba_vals)
            print(f"  {label}: P(A|AB)={ab_mean:.3f} [{ab_lo:.3f},{ab_hi:.3f}], P(A|BA)={ba_mean:.3f} [{ba_lo:.3f},{ba_hi:.3f}]")

    # 5. Steering fallback rate
    eot_fallbacks = sum(1 for r in eot_records if r.get("steering_fallback", False))
    print(f"\n--- Steering fallback (span detection) ---")
    print(f"  EOT: {eot_fallbacks}/{len(eot_records)} = {eot_fallbacks/len(eot_records):.4f}" if eot_records else "  No EOT records")

    # Save summary for report
    summary = {
        "parse_rates": {},
        "ordering_diffs": {},
        "steering_effects": {},
        "per_ordering_pa": {},
        "correlations": {},
    }

    for label, recs in [("baseline", baseline_records), ("prompt_last_pos", prompt_last_pos),
                         ("prompt_last_neg", prompt_last_neg), ("eot_pos", eot_pos), ("eot_neg", eot_neg)]:
        valid = sum(1 for r in recs if r["choice_original"] in ("a", "b"))
        summary["parse_rates"][label] = {"valid": valid, "total": len(recs), "rate": valid / len(recs) if recs else 0}

    for label, diffs in [("baseline", baseline_diffs), ("prompt_last_pos", pl_pos_diffs),
                          ("prompt_last_neg", pl_neg_diffs), ("eot_pos", eot_pos_diffs),
                          ("eot_neg", eot_neg_diffs)]:
        vals = list(diffs.values())
        if vals:
            mean, ci_lo, ci_hi = bootstrap_mean_ci(vals)
            summary["ordering_diffs"][label] = {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": len(vals)}

    for label, effects in [("prompt_last_pos", pl_pos_effects), ("prompt_last_neg", pl_neg_effects),
                           ("eot_pos", eot_pos_effects), ("eot_neg", eot_neg_effects)]:
        vals = list(effects.values())
        if vals:
            mean, ci_lo, ci_hi = bootstrap_mean_ci(vals)
            summary["steering_effects"][label] = {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": len(vals)}

    with open(ASSETS_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {ASSETS_DIR / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()
