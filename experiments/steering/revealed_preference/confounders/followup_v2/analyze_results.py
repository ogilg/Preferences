"""Analyze followup v2 results: position-adjusted effects, Δmu sensitivity, condition comparisons."""

import json
from pathlib import Path

import numpy as np
from scipy import stats

OUTPUT_DIR = Path("experiments/steering/revealed_preference/confounders/followup_v2")
PAIRS_FILE = OUTPUT_DIR / "utility_matched_pairs.json"


def load_results(name: str) -> list[dict]:
    path = OUTPUT_DIR / f"{name}_results.json"
    with open(path) as f:
        return json.load(f)


def compute_pa(results: list[dict], filters: dict | None = None) -> float:
    filtered = results
    if filters:
        for k, v in filters.items():
            filtered = [r for r in filtered if r.get(k) == v]
    valid = [r for r in filtered if r.get("choice") is not None]
    if not valid:
        return float("nan")
    return sum(1 for r in valid if r["choice"] == "a") / len(valid)


def compute_slope(results: list[dict], filters: dict | None = None):
    """Compute slope of P(A) vs coefficient using linear regression."""
    by_coef = {}
    filtered = results
    if filters:
        for k, v in filters.items():
            filtered = [r for r in filtered if r.get(k) == v]

    for r in filtered:
        if r.get("choice") is None:
            continue
        coef = r["coefficient"]
        if coef not in by_coef:
            by_coef[coef] = []
        by_coef[coef].append(1 if r["choice"] == "a" else 0)

    if len(by_coef) < 2:
        return float("nan"), float("nan"), float("nan")

    coefs = sorted(by_coef.keys())
    x = np.array(coefs)
    y = np.array([np.mean(by_coef[c]) for c in coefs])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, p_value, r_value


def analyze_probe_differential():
    """Main analysis: position-adjusted effects, Δmu sensitivity."""
    print("=" * 70)
    print("PROBE DIFFERENTIAL ANALYSIS")
    print("=" * 70)

    results = load_results("probe_differential")
    with open(PAIRS_FILE) as f:
        pairs = json.load(f)
    pair_info = {p["pair_idx"]: p for p in pairs}

    # Overall slopes by ordering
    for ordering in ["original", "swapped"]:
        slope, p, r = compute_slope(results, {"ordering": ordering})
        print(f"\n{ordering.upper()}: slope={slope:.2e}, p={p:.2e}, r={r:.3f}")

    # Position-adjusted decomposition (aggregate)
    slope_orig, _, _ = compute_slope(results, {"ordering": "original"})
    slope_swap, _, _ = compute_slope(results, {"ordering": "swapped"})
    position_component = (slope_orig + slope_swap) / 2
    content_component = (slope_orig - slope_swap) / 2

    print(f"\nAggregate decomposition:")
    print(f"  Position component: {position_component:.2e}")
    print(f"  Content component:  {content_component:.2e}")
    print(f"  Position %: {abs(position_component) / (abs(position_component) + abs(content_component)) * 100:.1f}%")

    # Per Δmu bin analysis
    print(f"\n{'Δmu bin':<10} {'N':>4} {'slope_orig':>12} {'slope_swap':>12} {'position':>12} {'content':>12}")
    print("-" * 70)

    bins = ["0-1", "1-2", "2-3", "3-5", "5-20"]
    bin_results = {}
    for b in bins:
        bin_data = [r for r in results if r.get("delta_mu_bin") == b]
        if not bin_data:
            continue
        n_pairs = len(set(r["pair_idx"] for r in bin_data))
        s_orig, _, _ = compute_slope(bin_data, {"ordering": "original"})
        s_swap, _, _ = compute_slope(bin_data, {"ordering": "swapped"})
        pos = (s_orig + s_swap) / 2
        con = (s_orig - s_swap) / 2
        print(f"{b:<10} {n_pairs:>4} {s_orig:>12.2e} {s_swap:>12.2e} {pos:>12.2e} {con:>12.2e}")
        bin_results[b] = {"slope_orig": s_orig, "slope_swap": s_swap, "position": pos, "content": con}

    # Per-pair analysis for borderline
    print(f"\n\nPER-PAIR ANALYSIS (borderline, Δmu 0-1)")
    print(f"{'Pair':>6} {'Δmu':>6} {'slope_orig':>12} {'slope_swap':>12} {'position':>12} {'content':>12}")
    print("-" * 70)

    borderline_data = [r for r in results if r.get("delta_mu_bin") == "0-1"]
    pair_indices = sorted(set(r["pair_idx"] for r in borderline_data))

    per_pair_effects = []
    for pidx in pair_indices:
        pair_data = [r for r in borderline_data if r["pair_idx"] == pidx]
        info = pair_info[pidx]
        s_orig, _, _ = compute_slope(pair_data, {"ordering": "original"})
        s_swap, _, _ = compute_slope(pair_data, {"ordering": "swapped"})
        pos = (s_orig + s_swap) / 2
        con = (s_orig - s_swap) / 2
        per_pair_effects.append({
            "pair_idx": pidx, "delta_mu": info["delta_mu"],
            "slope_orig": s_orig, "slope_swap": s_swap,
            "position": pos, "content": con,
        })
        print(f"{pidx:>6} {info['delta_mu']:>6.2f} {s_orig:>12.2e} {s_swap:>12.2e} {pos:>12.2e} {con:>12.2e}")

    # Test content component against zero
    content_effects = [p["content"] for p in per_pair_effects if not np.isnan(p["content"])]
    if content_effects:
        t_stat, p_val = stats.ttest_1samp(content_effects, 0)
        print(f"\nContent component t-test: t={t_stat:.3f}, p={p_val:.4f}")
        print(f"  Mean content effect: {np.mean(content_effects):.2e}")
        print(f"  95% CI: [{np.mean(content_effects) - 1.96*np.std(content_effects)/np.sqrt(len(content_effects)):.2e}, "
              f"{np.mean(content_effects) + 1.96*np.std(content_effects)/np.sqrt(len(content_effects)):.2e}]")

    return bin_results, per_pair_effects


def analyze_same_task():
    """Analyze same-task condition."""
    print("\n" + "=" * 70)
    print("SAME-TASK ANALYSIS")
    print("=" * 70)

    results = load_results("same_task")
    slope, p, r = compute_slope(results)
    print(f"Slope: {slope:.2e}, p={p:.2e}, r={r:.3f}")

    # Baseline P(A) at coef=0
    pa_0 = compute_pa(results, {"coefficient": 0.0})
    print(f"Baseline P(A) at coef=0: {pa_0:.3f}")

    # P(A) range
    pa_neg = compute_pa(results, {"coefficient": -3000.0})
    pa_pos = compute_pa(results, {"coefficient": 3000.0})
    print(f"P(A) range: {pa_neg:.3f} → {pa_pos:.3f} (Δ={pa_pos - pa_neg:+.3f})")


def analyze_header_only():
    """Analyze header-only condition."""
    print("\n" + "=" * 70)
    print("HEADER-ONLY ANALYSIS")
    print("=" * 70)

    results = load_results("header_only")

    for ordering in ["original", "swapped"]:
        slope, p, r = compute_slope(results, {"ordering": ordering})
        print(f"\n{ordering.upper()}: slope={slope:.2e}, p={p:.2e}, r={r:.3f}")

    slope_orig, _, _ = compute_slope(results, {"ordering": "original"})
    slope_swap, _, _ = compute_slope(results, {"ordering": "swapped"})
    position_component = (slope_orig + slope_swap) / 2
    content_component = (slope_orig - slope_swap) / 2

    print(f"\nDecomposition:")
    print(f"  Position component: {position_component:.2e}")
    print(f"  Content component:  {content_component:.2e}")


def analyze_random_directions():
    """Analyze random direction control."""
    print("\n" + "=" * 70)
    print("RANDOM DIRECTIONS ANALYSIS")
    print("=" * 70)

    results = load_results("random_directions")

    # Per-direction ΔP(A)
    dir_effects = []
    for dir_idx in range(20):
        dir_data = [r for r in results if r["dir_idx"] == dir_idx]
        pa_neg = compute_pa(dir_data, {"coefficient": -3000.0})
        pa_pos = compute_pa(dir_data, {"coefficient": 3000.0})
        delta = pa_pos - pa_neg
        dir_effects.append(delta)
        if dir_idx < 5:
            print(f"  Dir {dir_idx}: P(A) {pa_neg:.3f} → {pa_pos:.3f} (Δ={delta:+.3f})")

    print(f"\nRandom directions (N={len(dir_effects)}):")
    print(f"  Mean |Δ|: {np.mean(np.abs(dir_effects)):.3f}")
    print(f"  Mean Δ: {np.mean(dir_effects):+.3f}")
    print(f"  Std Δ: {np.std(dir_effects):.3f}")

    # Compare to probe
    probe_results = load_results("probe_differential")
    borderline_probe = [r for r in probe_results
                       if r.get("delta_mu_bin") == "0-1" and r.get("ordering") == "original"]
    probe_pa_neg = compute_pa(borderline_probe, {"coefficient": -3000.0})
    probe_pa_pos = compute_pa(borderline_probe, {"coefficient": 3000.0})
    probe_delta = probe_pa_pos - probe_pa_neg

    print(f"\nProbe Δ (borderline, original): {probe_delta:+.3f}")
    print(f"Probe |Δ|: {abs(probe_delta):.3f}")

    # Z-score
    z = (abs(probe_delta) - np.mean(np.abs(dir_effects))) / np.std(np.abs(dir_effects))
    print(f"Z-score (|Δ| vs random): {z:.2f}")
    rank = sum(1 for d in dir_effects if abs(d) >= abs(probe_delta))
    print(f"Rank p-value: {rank}/{len(dir_effects)} ({rank / len(dir_effects):.3f})")


def analyze_delta_mu_sensitivity():
    """E7: Position-adjusted steering effect vs Δmu."""
    print("\n" + "=" * 70)
    print("DELTA-MU SENSITIVITY ANALYSIS (E7)")
    print("=" * 70)

    results = load_results("probe_differential")
    with open(PAIRS_FILE) as f:
        pairs = json.load(f)
    pair_info = {p["pair_idx"]: p for p in pairs}

    # Per-pair position-adjusted effect
    pair_indices = sorted(set(r["pair_idx"] for r in results))
    effects = []
    for pidx in pair_indices:
        pair_data = [r for r in results if r["pair_idx"] == pidx]
        info = pair_info[pidx]
        s_orig, _, _ = compute_slope(pair_data, {"ordering": "original"})
        s_swap, _, _ = compute_slope(pair_data, {"ordering": "swapped"})

        if np.isnan(s_orig) or np.isnan(s_swap):
            continue

        content = (s_orig - s_swap) / 2
        position = (s_orig + s_swap) / 2
        effects.append({
            "pair_idx": pidx,
            "delta_mu": info["delta_mu"],
            "content_effect": content,
            "position_effect": position,
            "slope_orig": s_orig,
            "slope_swap": s_swap,
        })

    # Correlation: Δmu vs content effect
    delta_mus = np.array([e["delta_mu"] for e in effects])
    content_effects = np.array([e["content_effect"] for e in effects])
    position_effects = np.array([e["position_effect"] for e in effects])

    r_content, p_content = stats.pearsonr(delta_mus, content_effects)
    r_position, p_position = stats.pearsonr(delta_mus, position_effects)

    print(f"\nCorrelation with Δmu:")
    print(f"  Content effect: r={r_content:.3f}, p={p_content:.4f}")
    print(f"  Position effect: r={r_position:.3f}, p={p_position:.4f}")

    # By Δmu bin
    print(f"\n{'Δmu bin':<10} {'N':>4} {'Mean content':>14} {'Mean position':>14} {'Mean |content|':>14}")
    print("-" * 70)
    for b in ["0-1", "1-2", "2-3", "3-5", "5-20"]:
        bin_effects = [e for e in effects if pair_info[e["pair_idx"]]["delta_mu_bin"] == b]
        if not bin_effects:
            continue
        ce = [e["content_effect"] for e in bin_effects]
        pe = [e["position_effect"] for e in bin_effects]
        print(f"{b:<10} {len(bin_effects):>4} {np.mean(ce):>14.2e} {np.mean(pe):>14.2e} {np.mean(np.abs(ce)):>14.2e}")

    # Save for plotting
    output_path = OUTPUT_DIR / "delta_mu_effects.json"
    with open(output_path, "w") as f:
        json.dump(effects, f, indent=2)
    print(f"\nSaved per-pair effects to {output_path}")

    return effects


def main():
    print("FOLLOWUP V2 ANALYSIS")
    print("=" * 70)

    bin_results, per_pair = analyze_probe_differential()
    analyze_same_task()
    analyze_header_only()

    try:
        analyze_random_directions()
    except FileNotFoundError:
        print("\nRandom directions not yet available.")

    effects = analyze_delta_mu_sensitivity()

    # Summary comparison
    print("\n" + "=" * 70)
    print("CONDITION COMPARISON SUMMARY")
    print("=" * 70)

    probe_results = load_results("probe_differential")
    same_results = load_results("same_task")
    header_results = load_results("header_only")

    probe_slope, probe_p, _ = compute_slope(probe_results, {"ordering": "original"})
    same_slope, same_p, _ = compute_slope(same_results)
    header_slope, header_p, _ = compute_slope(header_results, {"ordering": "original"})

    print(f"\n{'Condition':<25} {'Slope':>12} {'p-value':>12} {'Δ P(A)':>10}")
    print("-" * 65)

    for name, slope, p, results_sub in [
        ("Probe (original)", probe_slope, probe_p,
         [r for r in probe_results if r["ordering"] == "original"]),
        ("Probe (swapped)", *compute_slope(probe_results, {"ordering": "swapped"})[:2],
         [r for r in probe_results if r["ordering"] == "swapped"]),
        ("Same-task", same_slope, same_p, same_results),
        ("Header (original)", header_slope, header_p,
         [r for r in header_results if r["ordering"] == "original"]),
        ("Header (swapped)", *compute_slope(header_results, {"ordering": "swapped"})[:2],
         [r for r in header_results if r["ordering"] == "swapped"]),
    ]:
        pa_neg = compute_pa(results_sub, {"coefficient": -3000.0})
        pa_pos = compute_pa(results_sub, {"coefficient": 3000.0})
        delta = pa_pos - pa_neg
        print(f"{name:<25} {slope:>12.2e} {p:>12.2e} {delta:>+10.3f}")


if __name__ == "__main__":
    main()
