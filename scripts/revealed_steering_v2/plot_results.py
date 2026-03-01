"""Plotting script for revealed preference steering v2.

Generates four key plots:
1. Dose-response curve (ordering difference vs multiplier)
2. Steerability vs borderlineness scatter
3. Probe vs random comparison (if random data available)
4. P(choose first-listed) dose-response

Run from repo root: python scripts/revealed_steering_v2/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path("experiments/revealed_steering_v2")
ANALYSIS_PATH = EXP_DIR / "analysis_results.json"
ASSETS_DIR = EXP_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def load_analysis():
    return json.loads(ANALYSIS_PATH.read_text())


def plot_ordering_diff_dose_response(analysis: dict):
    """Plot ordering difference (AB P(A) - BA P(A)) as steering effect measure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Get ordering effects for probe condition
    ordering = analysis["ordering_effects"]
    mults = sorted(set(float(k.split("_ord")[0]) for k in ordering.keys()))

    ab_pa = []
    ba_pa = []
    diffs = []
    for mult in mults:
        key0 = f"{mult:.3f}_ord0"
        key1 = f"{mult:.3f}_ord1"
        if key0 in ordering and key1 in ordering:
            ab = ordering[key0]["p_a"]
            ba = ordering[key1]["p_a"]
            ab_pa.append(ab)
            ba_pa.append(ba)
            diffs.append(ab - ba)
        else:
            ab_pa.append(None)
            ba_pa.append(None)
            diffs.append(None)

    valid = [(m, d) for m, d in zip(mults, diffs) if d is not None]
    valid_mults = [x[0] for x in valid]
    valid_diffs = [x[1] for x in valid]

    # Left panel: AB and BA P(A) by multiplier
    valid_ab = [(m, a) for m, a in zip(mults, ab_pa) if a is not None]
    valid_ba = [(m, b) for m, b in zip(mults, ba_pa) if b is not None]

    ax1.plot([x[0] for x in valid_ab], [x[1] for x in valid_ab], 'o-', color='#2196F3', label='AB ordering (A first)', markersize=8)
    ax1.plot([x[0] for x in valid_ba], [x[1] for x in valid_ba], 's-', color='#FF5722', label='BA ordering (B first)', markersize=8)
    ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(0.0, color='gray', linestyle=':', alpha=0.3)
    ax1.set_xlabel('Steering multiplier')
    ax1.set_ylabel('P(choose A, original)')
    ax1.set_title('Steering effect by ordering')
    ax1.legend(fontsize=9)
    ax1.set_ylim(0.1, 0.9)

    # Right panel: ordering difference (steering effect measure)
    ax2.plot(valid_mults, valid_diffs, 'D-', color='#4CAF50', markersize=8, linewidth=2)
    # Add baseline reference
    baseline_diff = None
    for m, d in zip(valid_mults, valid_diffs):
        if abs(m) < 0.001:
            baseline_diff = d
            break
    if baseline_diff is not None:
        ax2.axhline(baseline_diff, color='gray', linestyle='--', alpha=0.7, label=f'Baseline position bias ({baseline_diff:.3f})')
    ax2.axvline(0.0, color='gray', linestyle=':', alpha=0.3)
    ax2.set_xlabel('Steering multiplier')
    ax2.set_ylabel('Ordering difference: P(A|AB) - P(A|BA)')
    ax2.set_title('Steering effect (ordering difference)')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = ASSETS_DIR / "plot_022826_ordering_dose_response.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def plot_p_choose_first(analysis: dict):
    """Plot P(choose first-listed task) which combines position bias + steering effect."""
    ordering = analysis["ordering_effects"]
    mults = sorted(set(float(k.split("_ord")[0]) for k in ordering.keys()))

    p_first = []
    for mult in mults:
        key0 = f"{mult:.3f}_ord0"
        key1 = f"{mult:.3f}_ord1"
        if key0 in ordering and key1 in ordering:
            ab_pa = ordering[key0]["p_a"]  # In AB, first-listed is A, so P(first) = P(A)
            ba_pa = ordering[key1]["p_a"]  # In BA, first-listed is B, so P(first) = 1 - P(A_original)
            avg = (ab_pa + (1 - ba_pa)) / 2
            p_first.append(avg)
        else:
            p_first.append(None)

    valid = [(m, p) for m, p in zip(mults, p_first) if p is not None]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([x[0] for x in valid], [x[1] for x in valid], 'o-', color='#9C27B0', markersize=8, linewidth=2)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Chance')
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Steering multiplier')
    ax.set_ylabel('P(choose first-listed task)')
    ax.set_title('Position bias + steering effect')
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 0.85)

    plt.tight_layout()
    out = ASSETS_DIR / "plot_022826_p_choose_first.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def plot_steerability_vs_borderlineness(analysis: dict):
    """Scatter: per-pair steerability vs borderlineness."""
    pair_stats = analysis["pair_stats"]

    borderlineness = [p["borderlineness"] for p in pair_stats if p["borderlineness"] is not None]
    steerability = [p["max_steerability"] for p in pair_stats if p["borderlineness"] is not None]

    if len(borderlineness) < 10:
        print("Not enough data for steerability scatter")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(borderlineness, steerability, alpha=0.4, s=20, color='#2196F3')

    # Fit line
    b = np.array(borderlineness)
    s = np.array(steerability)
    r = np.corrcoef(b, s)[0, 1]

    # Add trend line
    z = np.polyfit(b, s, 1)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.7, label=f'r = {r:.3f}')

    ax.set_xlabel('Borderlineness (1 = perfectly 50/50 at baseline)')
    ax.set_ylabel('Max steerability (max |shift in P(A)|)')
    ax.set_title('Steerability vs Borderlineness')
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    out = ASSETS_DIR / "plot_022826_steerability_vs_borderlineness.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def plot_probe_vs_random(analysis: dict):
    """Compare probe vs random direction dose-response."""
    random_cmp = analysis.get("random_comparison", {})
    if random_cmp.get("status") == "no_random_data":
        print("No random direction data available yet")
        return

    if "comparison" not in random_cmp:
        print("No comparison data available")
        return

    comparison = random_cmp["comparison"]
    mults = sorted(comparison.keys(), key=lambda k: float(k))

    probe_pa = []
    random_pa = []
    mult_vals = []
    for k in mults:
        d = comparison[k]
        if d["probe_p_a"] is not None and d["random_p_a"] is not None:
            mult_vals.append(d["multiplier"])
            probe_pa.append(d["probe_p_a"])
            random_pa.append(d["random_p_a"])

    if not mult_vals:
        print("No overlapping data between probe and random")
        return

    # Also compute ordering differences for both conditions
    # Need to load checkpoint data directly for this
    import sys
    sys.path.insert(0, str(Path("scripts/revealed_steering_v2")))
    from analyze_results import load_records, compute_ordering_effects

    records = load_records()
    probe_ordering = compute_ordering_effects(records, "probe")
    random_ordering = compute_ordering_effects(records, "random")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall P(A)
    ax1.plot(mult_vals, probe_pa, 'o-', color='#2196F3', label='Probe direction', markersize=8)
    ax1.plot(mult_vals, random_pa, 's-', color='#FF9800', label='Random direction', markersize=8)
    ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(0.0, color='gray', linestyle=':', alpha=0.3)
    ax1.set_xlabel('Steering multiplier')
    ax1.set_ylabel('P(choose A, original)')
    ax1.set_title('Overall P(A): Probe vs Random')
    ax1.legend(fontsize=9)

    # Right: ordering difference
    shared_mults = sorted(set(float(k.split("_ord")[0]) for k in probe_ordering.keys()) &
                          set(float(k.split("_ord")[0]) for k in random_ordering.keys()))

    probe_diffs = []
    random_diffs = []
    for mult in shared_mults:
        pk0 = f"{mult:.3f}_ord0"
        pk1 = f"{mult:.3f}_ord1"
        rk0 = f"{mult:.3f}_ord0"
        rk1 = f"{mult:.3f}_ord1"
        if pk0 in probe_ordering and pk1 in probe_ordering:
            probe_diffs.append(probe_ordering[pk0]["p_a"] - probe_ordering[pk1]["p_a"])
        else:
            probe_diffs.append(None)
        if rk0 in random_ordering and rk1 in random_ordering:
            random_diffs.append(random_ordering[rk0]["p_a"] - random_ordering[rk1]["p_a"])
        else:
            random_diffs.append(None)

    valid_probe = [(m, d) for m, d in zip(shared_mults, probe_diffs) if d is not None]
    valid_random = [(m, d) for m, d in zip(shared_mults, random_diffs) if d is not None]

    if valid_probe:
        ax2.plot([x[0] for x in valid_probe], [x[1] for x in valid_probe], 'o-', color='#2196F3', label='Probe direction', markersize=8)
    if valid_random:
        ax2.plot([x[0] for x in valid_random], [x[1] for x in valid_random], 's-', color='#FF9800', label='Random direction', markersize=8)
    ax2.axvline(0.0, color='gray', linestyle=':', alpha=0.3)
    ax2.set_xlabel('Steering multiplier')
    ax2.set_ylabel('Ordering difference: P(A|AB) - P(A|BA)')
    ax2.set_title('Steering effect: Probe vs Random')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = ASSETS_DIR / "plot_022826_probe_vs_random.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def plot_steering_effect_derived(analysis: dict):
    """Plot the derived steering effect: (ordering_diff - baseline_diff) / 2."""
    ordering = analysis["ordering_effects"]
    mults = sorted(set(float(k.split("_ord")[0]) for k in ordering.keys()))

    # Get baseline ordering diff
    baseline_diff = None
    for mult in mults:
        if abs(mult) < 0.001:
            key0 = f"{mult:.3f}_ord0"
            key1 = f"{mult:.3f}_ord1"
            if key0 in ordering and key1 in ordering:
                baseline_diff = ordering[key0]["p_a"] - ordering[key1]["p_a"]
            break

    if baseline_diff is None:
        print("No baseline data for derived steering effect")
        return

    steering_effects = []
    for mult in mults:
        key0 = f"{mult:.3f}_ord0"
        key1 = f"{mult:.3f}_ord1"
        if key0 in ordering and key1 in ordering:
            diff = ordering[key0]["p_a"] - ordering[key1]["p_a"]
            effect = (diff - baseline_diff) / 2
            steering_effects.append(effect)
        else:
            steering_effects.append(None)

    valid = [(m, e) for m, e in zip(mults, steering_effects) if e is not None]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([x[0] for x in valid], [x[1] for x in valid], 'D-', color='#E91E63', markersize=8, linewidth=2)
    ax.axhline(0.0, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.3)

    # Shade expected positive region
    ax.fill_between([-0.15, 0], [-0.2, -0.2], [0, 0], alpha=0.05, color='red', label='Expected negative')
    ax.fill_between([0, 0.15], [0, 0], [0.2, 0.2], alpha=0.05, color='green', label='Expected positive')

    ax.set_xlabel('Steering multiplier')
    ax.set_ylabel('Steering effect: (ordering_diff - baseline) / 2')
    ax.set_title('Derived steering effect (corrected for position bias)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = ASSETS_DIR / "plot_022826_steering_effect.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def main():
    analysis = load_analysis()

    plot_ordering_diff_dose_response(analysis)
    plot_p_choose_first(analysis)
    plot_steerability_vs_borderlineness(analysis)
    plot_steering_effect_derived(analysis)
    plot_probe_vs_random(analysis)

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
