"""
Analysis for random direction control experiment.

Loads probe_rerun.json and random_seed{100,101,102}.json, computes:
1. Aggregate P(steered) at coef=+2641 for boost_a and diff_ab (each direction)
2. Per-pair effects (shift from within-experiment control or replication control)
3. Probe vs random comparison (t-tests, Cohen's d, rank)
4. KS test on per-pair slope distributions

Outputs plots to experiments/steering/replication/random_control/assets/
Prints summary table for report.

Usage:
  python scripts/random_control/analyze_random_control.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats as stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "steering" / "replication" / "random_control"
RESULTS_DIR = EXP_DIR / "results"
ASSETS_DIR = EXP_DIR / "assets"
REPLICATION_RESULTS = REPO_ROOT / "experiments" / "steering" / "replication" / "results"


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def compute_per_pair_effect(
    results: list[dict],
    condition: str,
    coef: float,
    control_results: list[dict] | None = None,
    control_coef: float = 0.0,
) -> dict[str, float]:
    """
    Compute per-pair effect: P(pick_a | condition=c, coef=coef) - P(pick_a | control).
    Averages over both orderings.

    For boost_a and diff_ab: "steered" response is 'a'.

    Returns dict: pair_id -> effect (pp)
    """
    # Which response is "steered" for each condition
    steered_choice = {"boost_a": "a", "diff_ab": "a"}[condition]

    # Compute P(steered) per pair×ordering for the steered condition
    pair_ordering_steered: dict[tuple[str, str], list[float]] = {}
    for trial in results:
        if trial["condition"] == condition and abs(trial["coefficient"] - coef) < 1e-3:
            key = (trial["pair_id"], trial["ordering"])
            responses = trial["responses"]
            valid = [r for r in responses if r != "parse_fail"]
            if not valid:
                continue
            p = sum(1 for r in valid if r == steered_choice) / len(valid)
            pair_ordering_steered[key] = pair_ordering_steered.get(key, [])
            pair_ordering_steered[key].append(p)

    # Compute P(pick_a) for control per pair×ordering
    if control_results is not None:
        src = control_results
    else:
        src = results

    pair_ordering_ctrl: dict[tuple[str, str], list[float]] = {}
    for trial in src:
        if trial["condition"] == "control" and abs(trial["coefficient"] - control_coef) < 1e-3:
            key = (trial["pair_id"], trial["ordering"])
            responses = trial["responses"]
            valid = [r for r in responses if r != "parse_fail"]
            if not valid:
                continue
            # For control, steered_choice for diff_ab and boost_a is 'a'
            p = sum(1 for r in valid if r == "a") / len(valid)
            pair_ordering_ctrl[key] = pair_ordering_ctrl.get(key, [])
            pair_ordering_ctrl[key].append(p)

    # Aggregate over orderings per pair
    all_pair_ids = set(k[0] for k in pair_ordering_steered) | set(k[0] for k in pair_ordering_ctrl)
    effects = {}
    for pid in sorted(all_pair_ids):
        s_vals = []
        c_vals = []
        for ordering in ["original", "swapped"]:
            key = (pid, ordering)
            if key in pair_ordering_steered:
                s_vals.extend(pair_ordering_steered[key])
            if key in pair_ordering_ctrl:
                c_vals.extend(pair_ordering_ctrl[key])
        if s_vals and c_vals:
            effects[pid] = float(np.mean(s_vals) - np.mean(c_vals))

    return effects


def compute_aggregate(results: list[dict], condition: str, coef: float) -> tuple[float, int]:
    """Return (mean P(steered), total valid responses) at given condition and coef."""
    steered_choice = {"boost_a": "a", "diff_ab": "a", "control": "a"}[condition]
    total, hits = 0, 0
    for trial in results:
        if trial["condition"] == condition and abs(trial["coefficient"] - coef) < 1e-3:
            for r in trial["responses"]:
                if r != "parse_fail":
                    total += 1
                    if r == steered_choice:
                        hits += 1
    if total == 0:
        return float("nan"), 0
    return hits / total, total


def compute_per_pair_at_coef(results: list[dict], condition: str, coef: float) -> dict[str, float]:
    """P(steered) per pair (averaged over orderings) at given condition+coef."""
    steered_choice = {"boost_a": "a", "diff_ab": "a"}[condition]
    pair_ordering: dict[tuple[str, str], list[float]] = {}
    for trial in results:
        if trial["condition"] == condition and abs(trial["coefficient"] - coef) < 1e-3:
            key = (trial["pair_id"], trial["ordering"])
            valid = [r for r in trial["responses"] if r != "parse_fail"]
            if valid:
                pair_ordering[key] = pair_ordering.get(key, [])
                pair_ordering[key].append(sum(1 for r in valid if r == steered_choice) / len(valid))

    pair_ids = sorted(set(k[0] for k in pair_ordering))
    out = {}
    for pid in pair_ids:
        vals = []
        for ordering in ["original", "swapped"]:
            key = (pid, ordering)
            if key in pair_ordering:
                vals.extend(pair_ordering[key])
        if vals:
            out[pid] = float(np.mean(vals))
    return out


def analyze_direction(
    results: list[dict],
    direction_label: str,
    control_results: list[dict],
) -> dict:
    """
    Compute full analysis for one direction (probe or random).

    Returns dict with per-condition metrics.
    """
    out = {"label": direction_label, "conditions": {}}

    for condition in ["boost_a", "diff_ab"]:
        cond_out = {}

        # Aggregate P(steered) at +2641
        agg_pos, n_pos = compute_aggregate(results, condition, 2641.0)
        agg_neg, n_neg = compute_aggregate(results, condition, -2641.0)
        ctrl_agg, n_ctrl = compute_aggregate(control_results, "control", 0.0)

        cond_out["agg_pos2641"] = agg_pos
        cond_out["agg_neg2641"] = agg_neg
        cond_out["agg_ctrl"] = ctrl_agg
        cond_out["shift_pp_pos"] = (agg_pos - ctrl_agg) * 100 if not np.isnan(agg_pos) and not np.isnan(ctrl_agg) else float("nan")
        cond_out["shift_pp_neg"] = (agg_neg - ctrl_agg) * 100 if not np.isnan(agg_neg) and not np.isnan(ctrl_agg) else float("nan")

        # Per-pair effects at +2641
        effects_pos = compute_per_pair_effect(results, condition, 2641.0, control_results)
        if effects_pos:
            vals = list(effects_pos.values())
            n = len(vals)
            mean_eff = np.mean(vals) * 100
            sem = stats.sem(vals) * 100
            t_stat, p_val = stats.ttest_1samp(vals, 0)
            frac_pos = sum(1 for v in vals if v > 0) / n

            cond_out["n_pairs"] = n
            cond_out["mean_effect_pp"] = float(mean_eff)
            cond_out["sem_pp"] = float(sem)
            cond_out["t_stat"] = float(t_stat)
            cond_out["p_value"] = float(p_val)
            cond_out["frac_positive"] = float(frac_pos)
            cond_out["per_pair_effects"] = {k: v * 100 for k, v in effects_pos.items()}
        else:
            cond_out["n_pairs"] = 0
            cond_out["mean_effect_pp"] = float("nan")

        out["conditions"][condition] = cond_out

    return out


def main():
    # Load probe re-run
    probe_path = RESULTS_DIR / "probe_rerun.json"
    if not probe_path.exists():
        print(f"ERROR: {probe_path} not found. Run probe mode first.")
        sys.exit(1)

    probe_data = load_json(probe_path)
    probe_results = probe_data["results"]

    # Load random directions
    random_data = {}
    for seed in [100, 101, 102, 103, 104]:
        path = RESULTS_DIR / f"random_seed{seed}.json"
        if path.exists():
            random_data[seed] = load_json(path)

    if not random_data:
        print("ERROR: No random direction results found.")
        sys.exit(1)

    print(f"Loaded probe + {len(random_data)} random direction(s): seeds {list(random_data)}")

    # Load replication Phase 1 results for reference
    phase1_path = REPLICATION_RESULTS / "steering_phase1.json"
    phase1_data = load_json(phase1_path) if phase1_path.exists() else None

    # Use within-experiment control from probe re-run
    control_results = probe_results

    # Analyze probe
    probe_analysis = analyze_direction(probe_results, "probe_L31", control_results)

    # Analyze random directions
    random_analyses = {}
    for seed, data in random_data.items():
        label = f"random_seed{seed}"
        analysis = analyze_direction(data["results"], label, control_results)
        random_analyses[seed] = analysis

    # ────────────────────────────────────────────────────────────
    # Summary table
    # ────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    for condition in ["boost_a", "diff_ab"]:
        print(f"\n--- {condition} (coef=+2641, vs within-experiment control) ---")
        print(f"{'Direction':<22} {'Mean effect (pp)':>18} {'t':>7} {'p':>8} {'%pos':>6} {'N':>4}")
        print("-" * 70)

        # Probe
        c = probe_analysis["conditions"][condition]
        print(f"{'probe_L31':<22} {c.get('mean_effect_pp', float('nan')):>+17.1f} "
              f"{c.get('t_stat', float('nan')):>7.2f} {c.get('p_value', float('nan')):>8.4f} "
              f"{c.get('frac_positive', float('nan')):>5.1%} {c.get('n_pairs', 0):>4}")

        random_effects = []
        for seed in sorted(random_analyses):
            rc = random_analyses[seed]["conditions"][condition]
            print(f"{'random_'+str(seed):<22} {rc.get('mean_effect_pp', float('nan')):>+17.1f} "
                  f"{rc.get('t_stat', float('nan')):>7.2f} {rc.get('p_value', float('nan')):>8.4f} "
                  f"{rc.get('frac_positive', float('nan')):>5.1%} {rc.get('n_pairs', 0):>4}")
            if not np.isnan(rc.get("mean_effect_pp", float("nan"))):
                random_effects.append(rc["mean_effect_pp"])

        if random_effects:
            print(f"\n  Random mean: {np.mean(random_effects):+.1f}pp  (SD={np.std(random_effects, ddof=1):.1f}pp)")
            probe_eff = c.get("mean_effect_pp", float("nan"))
            if not np.isnan(probe_eff):
                diff = probe_eff - np.mean(random_effects)
                print(f"  Probe - random mean: {diff:+.1f}pp")

    # ────────────────────────────────────────────────────────────
    # Probe vs random comparison (formal tests)
    # ────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("PROBE vs RANDOM COMPARISON")
    print("="*80)

    for condition in ["boost_a", "diff_ab"]:
        print(f"\n--- {condition} ---")

        probe_effects = list(probe_analysis["conditions"][condition].get("per_pair_effects", {}).values())
        if not probe_effects:
            print("  No probe per-pair effects")
            continue

        all_random_effects = []
        for seed in sorted(random_analyses):
            rc = random_analyses[seed]["conditions"][condition]
            effects = list(rc.get("per_pair_effects", {}).values())
            all_random_effects.extend(effects)

        if not all_random_effects:
            print("  No random per-pair effects")
            continue

        # Cohen's d
        probe_mean = np.mean(probe_effects)
        probe_std = np.std(probe_effects)
        random_mean = np.mean(all_random_effects)
        random_std = np.std(all_random_effects)
        pooled_std = np.sqrt((probe_std**2 + random_std**2) / 2)
        cohens_d = (probe_mean - random_mean) / pooled_std if pooled_std > 0 else float("nan")

        print(f"  Probe mean: {probe_mean:+.1f}pp  Random pooled mean: {random_mean:+.1f}pp")
        print(f"  Cohen's d (probe vs random): {cohens_d:.2f}")

        # Welch's t-test: probe vs all random pooled (unequal variances)
        t2, p2 = stats.ttest_ind(probe_effects, all_random_effects, equal_var=False)
        print(f"  Welch's t-test (probe vs pooled random): t={t2:.2f}, p={p2:.4f}")

        # KS test
        ks_stat, ks_p = stats.ks_2samp(probe_effects, all_random_effects)
        print(f"  KS test: D={ks_stat:.3f}, p={ks_p:.4f}")

        # Rank of probe among all directions
        random_means = []
        for seed in sorted(random_analyses):
            rc = random_analyses[seed]["conditions"][condition]
            random_means.append(rc.get("mean_effect_pp", float("nan")))

        all_means = [probe_analysis["conditions"][condition].get("mean_effect_pp", float("nan"))] + random_means
        all_means_sorted = sorted(all_means, reverse=True)
        probe_rank = all_means_sorted.index(probe_analysis["conditions"][condition].get("mean_effect_pp")) + 1
        n_dirs = len(all_means)
        print(f"  Probe rank: {probe_rank}/{n_dirs} (p_rank = {probe_rank/n_dirs:.3f})")

    # ────────────────────────────────────────────────────────────
    # Save analysis JSON
    # ────────────────────────────────────────────────────────────
    analysis_out = {
        "probe": probe_analysis,
        "random": {str(k): v for k, v in random_analyses.items()},
    }
    out_path = RESULTS_DIR / "analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis_out, f, indent=2)
    print(f"\nSaved analysis to {out_path}")


if __name__ == "__main__":
    main()
