"""Analysis script for revealed preference steering v2.

Reads checkpoint.jsonl, produces:
  1. Dose-response curve (P(choose A) vs coefficient)
  2. Steerability vs borderlineness scatter
  3. Random control comparison
  4. Ordering effects analysis
  5. Summary statistics for the report

Can run on partial data (only probe condition, no random yet).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP_DIR = Path("experiments/revealed_steering_v2")
CHECKPOINT_PATH = EXP_DIR / "checkpoint.jsonl"
RESULTS_PATH = EXP_DIR / "steering_results.json"
ANALYSIS_PATH = EXP_DIR / "analysis_results.json"


def load_records():
    records = []
    for line in CHECKPOINT_PATH.read_text().strip().split("\n"):
        if line.strip():
            records.append(json.loads(line))
    return records


def compute_dose_response(records: list[dict], condition: str = "probe") -> dict:
    """Compute P(choose A) at each multiplier."""
    cond_records = [r for r in records if r["condition"] == condition]
    by_mult = defaultdict(list)
    for r in cond_records:
        by_mult[r["multiplier"]].append(r)

    results = {}
    for mult in sorted(by_mult.keys()):
        recs = by_mult[mult]
        valid = [r for r in recs if r["choice_original"] is not None]
        n_a = sum(1 for r in valid if r["choice_original"] == "a")
        n_total = len(valid)
        n_unparseable = len(recs) - len(valid)
        results[f"{mult:.3f}"] = {
            "multiplier": mult,
            "n_a": n_a,
            "n_total": n_total,
            "n_unparseable": n_unparseable,
            "p_a": n_a / n_total if n_total > 0 else None,
            "parse_rate": n_total / len(recs) if recs else None,
        }
    return results


def compute_per_pair_stats(records: list[dict], condition: str = "probe") -> list[dict]:
    """Compute per-pair statistics at each multiplier."""
    cond_records = [r for r in records if r["condition"] == condition]
    by_pair_mult = defaultdict(list)
    for r in cond_records:
        by_pair_mult[(r["pair_id"], r["multiplier"])].append(r)

    # Also compute baseline stats (mult=0)
    baseline_by_pair = defaultdict(list)
    for r in cond_records:
        if r["multiplier"] == 0.0:
            baseline_by_pair[r["pair_id"]].append(r)

    # Per-pair across multipliers
    pair_ids = set(r["pair_id"] for r in cond_records)
    pair_stats = []
    for pair_id in sorted(pair_ids):
        # Baseline P(A)
        base_recs = baseline_by_pair.get(pair_id, [])
        base_valid = [r for r in base_recs if r["choice_original"] is not None]
        base_n_a = sum(1 for r in base_valid if r["choice_original"] == "a")
        base_p_a = base_n_a / len(base_valid) if base_valid else None
        base_borderlineness = 1.0 - abs(base_p_a - 0.5) * 2 if base_p_a is not None else None  # 1.0 = perfectly borderline

        delta_mu = base_recs[0].get("delta_mu") if base_recs else None

        # P(A) at each multiplier
        mult_p_a = {}
        for (pid, mult), recs in by_pair_mult.items():
            if pid != pair_id:
                continue
            valid = [r for r in recs if r["choice_original"] is not None]
            n_a = sum(1 for r in valid if r["choice_original"] == "a")
            mult_p_a[mult] = n_a / len(valid) if valid else None

        # Steerability: max shift from baseline
        max_shift = 0.0
        if base_p_a is not None:
            for mult, pa in mult_p_a.items():
                if pa is not None:
                    shift = abs(pa - base_p_a)
                    if shift > max_shift:
                        max_shift = shift

        pair_stats.append({
            "pair_id": pair_id,
            "delta_mu": delta_mu,
            "baseline_p_a": base_p_a,
            "borderlineness": base_borderlineness,
            "max_steerability": max_shift,
            "p_a_by_multiplier": {f"{m:.3f}": p for m, p in sorted(mult_p_a.items())},
        })

    return pair_stats


def compute_ordering_effects(records: list[dict], condition: str = "probe") -> dict:
    """Check whether steering effects are symmetric across AB/BA orderings."""
    cond_records = [r for r in records if r["condition"] == condition]
    by_mult_ordering = defaultdict(lambda: defaultdict(list))
    for r in cond_records:
        by_mult_ordering[r["multiplier"]][r["ordering"]].append(r)

    results = {}
    for mult in sorted(by_mult_ordering.keys()):
        for ordering in [0, 1]:
            recs = by_mult_ordering[mult][ordering]
            valid = [r for r in recs if r["choice_original"] is not None]
            n_a = sum(1 for r in valid if r["choice_original"] == "a")
            results[f"{mult:.3f}_ord{ordering}"] = {
                "multiplier": mult,
                "ordering": ordering,
                "n_a": n_a,
                "n_total": len(valid),
                "p_a": n_a / len(valid) if valid else None,
            }
    return results


def compute_random_comparison(records: list[dict]) -> dict:
    """Compare probe vs random direction effects."""
    probe_dr = compute_dose_response(records, "probe")
    random_dr = compute_dose_response(records, "random")

    # Per-pair comparison at each shared multiplier
    probe_recs = [r for r in records if r["condition"] == "probe"]
    random_recs = [r for r in records if r["condition"] == "random"]

    if not random_recs:
        return {"status": "no_random_data"}

    # Find shared multipliers
    probe_mults = set(r["multiplier"] for r in probe_recs)
    random_mults = set(r["multiplier"] for r in random_recs)
    shared_mults = probe_mults & random_mults

    comparison = {}
    for mult in sorted(shared_mults):
        p_key = f"{mult:.3f}"
        p_data = probe_dr.get(p_key, {})
        r_data = random_dr.get(p_key, {})
        comparison[p_key] = {
            "multiplier": mult,
            "probe_p_a": p_data.get("p_a"),
            "random_p_a": r_data.get("p_a"),
            "probe_n": p_data.get("n_total", 0),
            "random_n": r_data.get("n_total", 0),
        }

    return {"comparison": comparison, "probe_dose_response": probe_dr, "random_dose_response": random_dr}


def compute_fallback_stats(records: list[dict]) -> dict:
    """Count how many trials used steering fallback."""
    fallback = [r for r in records if r.get("steering_fallback", False)]
    non_fallback = [r for r in records if not r.get("steering_fallback", False)]
    return {
        "n_fallback": len(fallback),
        "n_differential": len(non_fallback),
        "n_total": len(records),
        "fallback_rate": len(fallback) / len(records) if records else 0,
    }


def main():
    records = load_records()
    print(f"Loaded {len(records)} records")

    conditions = set(r["condition"] for r in records)
    print(f"Conditions: {conditions}")

    # Dose-response
    print("\n=== Dose-Response (probe) ===")
    probe_dr = compute_dose_response(records, "probe")
    for key, data in sorted(probe_dr.items(), key=lambda x: x[1]["multiplier"]):
        pa = data["p_a"]
        print(f"  mult={data['multiplier']:+.3f}: P(A)={pa:.3f} ({data['n_a']}/{data['n_total']}), parse={data['parse_rate']:.3f}")

    # Per-pair stats
    print("\n=== Per-Pair Stats ===")
    pair_stats = compute_per_pair_stats(records, "probe")
    borderlineness_vals = [p["borderlineness"] for p in pair_stats if p["borderlineness"] is not None]
    steerability_vals = [p["max_steerability"] for p in pair_stats]
    print(f"  N pairs: {len(pair_stats)}")
    if borderlineness_vals:
        print(f"  Mean borderlineness: {np.mean(borderlineness_vals):.3f}")
        print(f"  Mean max steerability: {np.mean(steerability_vals):.3f}")

        # Correlation
        both = [(p["borderlineness"], p["max_steerability"]) for p in pair_stats
                if p["borderlineness"] is not None]
        if len(both) > 10:
            b_vals = np.array([x[0] for x in both])
            s_vals = np.array([x[1] for x in both])
            corr = np.corrcoef(b_vals, s_vals)[0, 1]
            print(f"  Correlation (borderlineness, steerability): r={corr:.3f}")

    # Ordering effects
    print("\n=== Ordering Effects ===")
    ordering_effects = compute_ordering_effects(records, "probe")
    mults_seen = sorted(set(float(k.split("_ord")[0]) for k in ordering_effects.keys()))
    for mult in mults_seen:
        ord0 = ordering_effects.get(f"{mult:.3f}_ord0", {})
        ord1 = ordering_effects.get(f"{mult:.3f}_ord1", {})
        p0 = ord0.get("p_a", "N/A")
        p1 = ord1.get("p_a", "N/A")
        if isinstance(p0, float) and isinstance(p1, float):
            print(f"  mult={mult:+.3f}: AB P(A)={p0:.3f}, BA P(A)={p1:.3f}, diff={p0-p1:+.3f}")

    # Random comparison
    if "random" in conditions:
        print("\n=== Random Direction Comparison ===")
        random_cmp = compute_random_comparison(records)
        if "comparison" in random_cmp:
            for key, data in sorted(random_cmp["comparison"].items(), key=lambda x: x[1]["multiplier"]):
                pp = data["probe_p_a"]
                rp = data["random_p_a"]
                if pp is not None and rp is not None:
                    print(f"  mult={data['multiplier']:+.3f}: probe={pp:.3f}, random={rp:.3f}, delta={pp-rp:+.3f}")
    else:
        random_cmp = {"status": "no_random_data"}

    # Fallback stats
    print("\n=== Fallback Stats ===")
    fb_stats = compute_fallback_stats(records)
    print(f"  Fallback: {fb_stats['n_fallback']}/{fb_stats['n_total']} ({fb_stats['fallback_rate']:.1%})")

    # Save analysis
    analysis = {
        "probe_dose_response": probe_dr,
        "pair_stats_summary": {
            "n_pairs": len(pair_stats),
            "mean_borderlineness": float(np.mean(borderlineness_vals)) if borderlineness_vals else None,
            "mean_steerability": float(np.mean(steerability_vals)) if steerability_vals else None,
        },
        "ordering_effects": ordering_effects,
        "random_comparison": random_cmp,
        "fallback_stats": fb_stats,
        "pair_stats": pair_stats,
    }
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2))
    print(f"\nAnalysis saved to {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
