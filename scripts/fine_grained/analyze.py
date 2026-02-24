"""
Fine-Grained Steering Analysis Script.

Usage:
  python scripts/fine_grained/analyze.py
  python scripts/fine_grained/analyze.py --phase phase1
  python scripts/fine_grained/analyze.py --all

Outputs:
  experiments/steering/replication/fine_grained/results/analysis.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "steering" / "replication" / "fine_grained"
RESULTS_DIR = EXP_DIR / "results"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_p_a(responses: list[str]) -> tuple[float, int]:
    """Returns (p_a, n_valid)."""
    valid = [r for r in responses if r != "parse_fail"]
    if not valid:
        return 0.5, 0
    n_a = sum(1 for r in valid if r == "a")
    return n_a / len(valid), len(valid)


def analyze_phase(records: list[dict], condition_name: str | None = None) -> dict:
    """
    Analyze a set of JSONL records. Returns per-condition, per-coefficient P(a) stats.

    condition_name: if provided, only analyze this condition.
    """
    # Group by (pair_id, ordering, condition, coefficient)
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        cond = r["condition"]
        coef = r["coefficient"]
        key = (r["pair_id"], r["ordering"], cond, coef)
        groups[key].extend(r["responses"])

    # Per-pair per-condition per-coef P(a)
    pair_orderings = sorted({(r["pair_id"], r["ordering"]) for r in records})
    conditions = sorted({r["condition"] for r in records})
    coefficients = sorted({r["coefficient"] for r in records})

    # Compute control P(a) per pair×ordering (from coef=0, condition=control or control_ml or control_rand)
    ctrl_pa: dict[tuple, float] = {}
    for (pair_id, ordering) in pair_orderings:
        for cond in ["control", "control_ml", "control_rand"]:
            key = (pair_id, ordering, cond, 0.0)
            if key in groups:
                p_a, n_v = compute_p_a(groups[key])
                if n_v > 0:
                    ctrl_pa[(pair_id, ordering)] = p_a
                break

    # Per-pair per-condition effects vs control
    # For each steered condition and coefficient:
    # effect_per_pair = P(a|steered) - P(a|control) for same pair×ordering
    analysis: dict[str, dict] = {}

    for cond in conditions:
        if cond in ("control", "control_ml", "control_rand"):
            continue
        if condition_name and cond != condition_name:
            continue

        cond_data: dict = {"by_coef": {}}

        for coef in coefficients:
            if coef == 0.0:
                continue
            per_pair_effects = []
            p_a_vals = []
            ctrl_p_a_vals = []

            for pair_id, ordering in pair_orderings:
                key = (pair_id, ordering, cond, coef)
                if key not in groups:
                    continue
                ctrl_key = (pair_id, ordering)
                if ctrl_key not in ctrl_pa:
                    continue

                p_a, n_v = compute_p_a(groups[key])
                if n_v == 0:
                    continue
                ctrl = ctrl_pa[ctrl_key]
                effect = p_a - ctrl

                # For boost_b, measure P(b) = 1-P(a)
                if cond == "boost_b":
                    effect = (1 - p_a) - (1 - ctrl)

                per_pair_effects.append(effect)
                p_a_vals.append(p_a)
                ctrl_p_a_vals.append(ctrl)

            if not per_pair_effects:
                continue

            n = len(per_pair_effects)
            mean_effect = np.mean(per_pair_effects)
            se = np.std(per_pair_effects, ddof=1) / np.sqrt(n)
            t_stat, p_val = stats.ttest_1samp(per_pair_effects, 0)
            pct_positive = sum(1 for e in per_pair_effects if e > 0) / n

            cond_data["by_coef"][coef] = {
                "n_pairs": n,
                "mean_p_a": float(np.mean(p_a_vals)),
                "mean_ctrl_p_a": float(np.mean(ctrl_p_a_vals)),
                "mean_effect_pp": float(mean_effect * 100),
                "se_pp": float(se * 100),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "pct_positive": float(pct_positive),
            }

        analysis[cond] = cond_data

    # Also compute control stats
    ctrl_vals = list(ctrl_pa.values())
    if ctrl_vals:
        analysis["_control_summary"] = {
            "n_pairs": len(ctrl_vals),
            "mean_p_a": float(np.mean(ctrl_vals)),
            "std_p_a": float(np.std(ctrl_vals)),
        }

    return analysis


def analyze_all() -> dict:
    """Analyze all phases and combine results."""
    results = {}

    # Phase 1: L31
    phase1_path = RESULTS_DIR / "phase1_L31.jsonl"
    if phase1_path.exists():
        records = load_jsonl(phase1_path)
        print(f"Phase 1 (L31): {len(records)} records")
        results["phase1_L31"] = analyze_phase(records)

    # Phase 2: L49, L55
    for layer in [49, 55]:
        path = RESULTS_DIR / f"phase2_L{layer}.jsonl"
        if path.exists():
            records = load_jsonl(path)
            print(f"Phase 2 (L{layer}): {len(records)} records")
            results[f"phase2_L{layer}"] = analyze_phase(records)

    # Phase 3: multi-layer
    phase3_path = RESULTS_DIR / "phase3_multilayer.jsonl"
    if phase3_path.exists():
        records = load_jsonl(phase3_path)
        print(f"Phase 3 (multi-layer): {len(records)} records")
        results["phase3_multilayer"] = analyze_phase(records)

    # Phase 4: random controls
    for layer in [49, 55]:
        path = RESULTS_DIR / f"phase4_random_L{layer}.jsonl"
        if path.exists():
            records = load_jsonl(path)
            print(f"Phase 4 random (L{layer}): {len(records)} records")
            results[f"phase4_random_L{layer}"] = analyze_phase(records)

    return results


def print_summary(results: dict) -> None:
    """Print a readable summary of key results."""
    for phase_key, analysis in results.items():
        print(f"\n=== {phase_key} ===")
        ctrl = analysis.get("_control_summary", {})
        if ctrl:
            print(f"  Control P(a): {ctrl['mean_p_a']:.3f} ± {ctrl['std_p_a']:.3f}")

        for cond, cond_data in analysis.items():
            if cond.startswith("_"):
                continue
            by_coef = cond_data.get("by_coef", {})
            if not by_coef:
                continue
            print(f"\n  Condition: {cond}")
            print(f"  {'Coef':>10} {'N':>5} {'P(a)':>6} {'Effect':>8} {'SE':>6} {'t':>6} {'p':>8} {'%pos':>5}")
            for coef in sorted(by_coef.keys()):
                d = by_coef[coef]
                print(f"  {coef:>10.0f} {d['n_pairs']:>5} {d['mean_p_a']:>6.3f} "
                      f"{d['mean_effect_pp']:>7.1f}pp {d['se_pp']:>5.1f} "
                      f"{d['t_stat']:>6.2f} {d['p_value']:>8.4f} {d['pct_positive']:>5.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all")
    args = parser.parse_args()

    results = analyze_all()
    print_summary(results)

    out_path = RESULTS_DIR / "analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved analysis → {out_path}")


if __name__ == "__main__":
    main()
