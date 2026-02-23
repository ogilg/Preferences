#!/usr/bin/env python3
"""Phase 3 analysis: multi-layer steering comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "steering" / "replication"
RESULTS_DIR = EXP_DIR / "results"
ASSETS_DIR = EXP_DIR / "assets"

PHASE3_PATH = RESULTS_DIR / "steering_phase3.json"
PHASE1_PATH = RESULTS_DIR / "steering_phase1.json"

CONTROL_P_A = 0.595  # global fallback

CONDITIONS = [
    "L31_only",
    "L31_L37_same_split",
    "L31_L37_same_full",
    "L31_L37_layer_split",
    "L31_L43_layer_split",
    "L31_L37_L43_layer_split",
]

CONDITION_LABELS = {
    "L31_only": "L31",
    "L31_L37_same_split": "L31+L37 (same/split)",
    "L31_L37_same_full": "L31+L37 (same/full)",
    "L31_L37_layer_split": "L31+L37 (layer/split)",
    "L31_L43_layer_split": "L31+L43 (layer/split)",
    "L31_L37_L43_layer_split": "L31+L37+L43 (layer/split)",
}

CONDITION_COLORS = {
    "L31_only": "#2171b5",
    "L31_L37_same_split": "#f16913",
    "L31_L37_same_full": "#d94801",
    "L31_L37_layer_split": "#fd8d3c",
    "L31_L43_layer_split": "#e6550d",
    "L31_L37_L43_layer_split": "#a63603",
}

CONDITION_LINESTYLES = {
    "L31_only": "-",
    "L31_L37_same_split": "--",
    "L31_L37_same_full": "-.",
    "L31_L37_layer_split": ":",
    "L31_L43_layer_split": (0, (3, 1, 1, 1)),
    "L31_L37_L43_layer_split": (0, (5, 2)),
}


def p_a(responses: list[str]) -> float | None:
    valid = [r for r in responses if r in ("a", "b")]
    if not valid:
        return None
    return sum(1 for r in valid if r == "a") / len(valid)


def bootstrap_mean_ci(
    values: list[float], n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    point = float(arr.mean())
    boot = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def load_per_pair_controls() -> dict[str, float]:
    """Load per-pair control P(a) from Phase 1 results."""
    if not PHASE1_PATH.exists():
        print(f"Warning: Phase 1 results not found at {PHASE1_PATH}. Using global baseline.")
        return {}

    data = json.loads(PHASE1_PATH.read_text())
    pair_responses: dict[str, list[float]] = {}
    for r in data["results"]:
        if r["condition"] != "control":
            continue
        vals = [1.0 if resp == "a" else 0.0 for resp in r["responses"] if resp in ("a", "b")]
        pair_responses.setdefault(r["pair_id"], []).extend(vals)

    return {pid: float(np.mean(vals)) for pid, vals in pair_responses.items() if vals}


def collect_condition_coef_pa(
    results: list[dict], condition: str, coef: float
) -> list[float]:
    """Collect P(a) values across all entries for a given condition and coefficient."""
    entries = [
        r for r in results
        if r["condition"] == condition and abs(r["coefficient"] - coef) < 1.0
    ]
    out = []
    for r in entries:
        val = p_a(r["responses"])
        if val is not None:
            out.append(val)
    return out


def collect_per_pair_pa(
    results: list[dict], condition: str, coef: float
) -> dict[str, list[float]]:
    """Collect per-pair P(a) lists for a given condition and coefficient."""
    pair_data: dict[str, list[float]] = {}
    for r in results:
        if r["condition"] != condition:
            continue
        if abs(r["coefficient"] - coef) >= 1.0:
            continue
        val = p_a(r["responses"])
        if val is None:
            continue
        pair_data.setdefault(r["pair_id"], []).append(val)
    return pair_data


def compute_per_pair_effect(
    results: list[dict],
    condition: str,
    coef: float,
    per_pair_controls: dict[str, float],
) -> list[float]:
    """Per-pair effect = P(a) at coef - per-pair control P(a) from Phase 1.
    Falls back to global CONTROL_P_A if pair has no Phase 1 control."""
    pair_data = collect_per_pair_pa(results, condition, coef)
    effects = []
    for pair_id, vals in pair_data.items():
        mean_pa = float(np.mean(vals))
        ctrl = per_pair_controls.get(pair_id, CONTROL_P_A)
        effects.append(mean_pa - ctrl)
    return effects


# ── Plot 1: dose-response comparison ──────────────────────────────────────────

def plot_dose_response(results: list[dict], pos_coefs: list[float], save_path: Path) -> None:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Control baseline
    ax.axhline(
        CONTROL_P_A,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Control P(a) = {CONTROL_P_A:.3f}",
        zorder=2,
    )

    present_conditions = sorted(set(r["condition"] for r in results))

    for condition in CONDITIONS:
        if condition not in present_conditions:
            continue

        color = CONDITION_COLORS[condition]
        ls = CONDITION_LINESTYLES[condition]
        label = CONDITION_LABELS[condition]

        xs, ys, y_los, y_his = [], [], [], []
        for coef in pos_coefs:
            vals = collect_condition_coef_pa(results, condition, coef)
            if not vals:
                continue
            mean, lo, hi = bootstrap_mean_ci(vals)
            xs.append(coef)
            ys.append(mean)
            y_los.append(lo)
            y_his.append(hi)

        if not xs:
            continue

        xs_arr = np.array(xs)
        ys_arr = np.array(ys)
        y_los_arr = np.array(y_los)
        y_his_arr = np.array(y_his)

        ax.plot(
            xs_arr, ys_arr,
            color=color,
            linestyle=ls,
            linewidth=2.0,
            marker="o",
            markersize=6,
            label=label,
            zorder=4,
        )
        ax.fill_between(xs_arr, y_los_arr, y_his_arr, color=color, alpha=0.12, zorder=3)

    ax.set_xlabel("Steering coefficient", fontsize=12)
    ax.set_ylabel("P(a)", fontsize=12)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(left=min(pos_coefs) * 0.85, right=max(pos_coefs) * 1.1)
    ax.set_title("Phase 3: Multi-Layer Steering Comparison", fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Plot 2: effect-size bar chart ──────────────────────────────────────────────

def plot_slope_bars(
    results: list[dict],
    coef_low: float,
    per_pair_controls: dict[str, float],
    save_path: Path,
) -> None:
    sns.set_style("whitegrid")

    present_conditions = set(r["condition"] for r in results)
    ordered = [c for c in CONDITIONS if c in present_conditions]

    bar_means = []
    bar_los = []
    bar_his = []
    bar_colors = []
    bar_labels = []

    for condition in ordered:
        effects = compute_per_pair_effect(results, condition, coef_low, per_pair_controls)
        if not effects:
            bar_means.append(float("nan"))
            bar_los.append(float("nan"))
            bar_his.append(float("nan"))
        else:
            effects_pp = [e * 100 for e in effects]
            mean, lo, hi = bootstrap_mean_ci(effects_pp)
            bar_means.append(mean)
            bar_los.append(mean - lo)
            bar_his.append(hi - mean)
        bar_colors.append(CONDITION_COLORS[condition])
        bar_labels.append(CONDITION_LABELS[condition])

    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, (mean, lo_err, hi_err, color, label) in enumerate(
        zip(bar_means, bar_los, bar_his, bar_colors, bar_labels)
    ):
        if np.isnan(mean):
            continue
        ax.bar(
            x[i], mean,
            color=color,
            width=0.6,
            alpha=0.85,
            zorder=3,
        )
        ax.errorbar(
            x[i], mean,
            yerr=[[lo_err], [hi_err]],
            fmt="none",
            color="black",
            capsize=5,
            linewidth=1.5,
            zorder=4,
        )

    # Reference line at L31_only effect
    if "L31_only" in present_conditions:
        l31_effects = compute_per_pair_effect(results, "L31_only", coef_low, per_pair_controls)
        if l31_effects:
            l31_pp = float(np.mean([e * 100 for e in l31_effects]))
            ax.axhline(
                l31_pp,
                color=CONDITION_COLORS["L31_only"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                zorder=2,
                label=f"L31 effect = {l31_pp:.1f}pp",
            )
            ax.legend(fontsize=10, framealpha=0.9)

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Effect vs Phase 1 control (pp)", fontsize=12)
    ax.set_ylim(-5, 25)
    ax.set_title(
        f"Phase 3: Effect Size by Condition (at coef=+{int(coef_low):,})",
        fontsize=13,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Statistics ─────────────────────────────────────────────────────────────────

def print_statistics(
    results: list[dict],
    pos_coefs: list[float],
    per_pair_controls: dict[str, float],
) -> None:
    if len(pos_coefs) < 2:
        print("Warning: fewer than 2 positive coefficients found.")
        return

    coef_low = pos_coefs[0]
    coef_high = pos_coefs[-1]

    n_with_ctrl = sum(1 for pid in set(r["pair_id"] for r in results) if pid in per_pair_controls)
    print(f"\n=== Phase 3 Statistics ===")
    print(f"Control P(a) baseline: {CONTROL_P_A:.3f} (global fallback)")
    print(f"Per-pair controls from Phase 1: {n_with_ctrl}/{len(set(r['pair_id'] for r in results))} pairs matched")
    print(f"Positive coefficients analysed: {[int(c) for c in pos_coefs]}")

    present_conditions = sorted(set(r["condition"] for r in results))
    ordered = [c for c in CONDITIONS if c in present_conditions]

    print(f"\nConditions present: {present_conditions}")

    # Table header
    print(f"\n{'Condition':<30} {'P(a) @+{:.0f}'.format(coef_low):>14} {'Effect @low (pp)':>18} "
          f"{'P(a) @+{:.0f}'.format(coef_high):>14} {'Effect @high (pp)':>18}")
    print("-" * 100)

    l31_effects_low: list[float] = []
    condition_stats: dict[str, dict] = {}

    for condition in ordered:
        vals_low = collect_condition_coef_pa(results, condition, coef_low)
        vals_high = collect_condition_coef_pa(results, condition, coef_high)
        effects_low = compute_per_pair_effect(results, condition, coef_low, per_pair_controls)
        effects_high = compute_per_pair_effect(results, condition, coef_high, per_pair_controls)

        mean_low = float(np.mean(vals_low)) if vals_low else float("nan")
        mean_high = float(np.mean(vals_high)) if vals_high else float("nan")
        effect_low_pp = (mean_low - CONTROL_P_A) * 100 if not np.isnan(mean_low) else float("nan")
        effect_high_pp = (mean_high - CONTROL_P_A) * 100 if not np.isnan(mean_high) else float("nan")

        if condition == "L31_only":
            l31_effects_low = [e * 100 for e in effects_low]

        condition_stats[condition] = {
            "mean_low": mean_low,
            "mean_high": mean_high,
            "effect_low_pp": effect_low_pp,
            "effect_high_pp": effect_high_pp,
            "effects_low_pp": [e * 100 for e in effects_low],
            "effects_high_pp": [e * 100 for e in effects_high],
        }

        label = CONDITION_LABELS.get(condition, condition)
        print(
            f"{label:<30} {mean_low:>14.4f} {effect_low_pp:>+18.2f} "
            f"{mean_high:>14.4f} {effect_high_pp:>+18.2f}"
        )

    # T-tests per condition vs zero shift (using per-pair controls)
    print(f"\nT-test: does each condition produce a significant shift at coef=+{int(coef_low):,}?")
    print(f"  (One-sample t-test of per-pair effects vs Phase 1 control baseline)")
    print(f"  {'Condition':<30} {'N pairs':>8} {'Mean effect (pp)':>18} {'t-stat':>8} {'p-value':>10} {'% pos':>7}")
    print("  " + "-" * 90)

    for condition in ordered:
        label = CONDITION_LABELS.get(condition, condition)
        effects_pp = condition_stats[condition]["effects_low_pp"]
        n = len(effects_pp)
        if n < 2:
            print(f"  {label:<30} {n:>8}  (insufficient data)")
            continue
        t_stat, p_val = stats.ttest_1samp(effects_pp, 0)
        mean_eff = float(np.mean(effects_pp))
        pct_pos = 100.0 * sum(1 for e in effects_pp if e > 0) / n
        print(
            f"  {label:<30} {n:>8} {mean_eff:>+18.2f} {t_stat:>8.3f} {p_val:>10.4f} {pct_pos:>6.1f}%"
        )

    # Compare multi-layer vs L31_only (Welch's t-test)
    print(f"\nDo multi-layer conditions significantly exceed L31_only at coef=+{int(coef_low):,}?")
    if l31_effects_low:
        print(f"  L31_only mean effect: {float(np.mean(l31_effects_low)):+.2f}pp (n={len(l31_effects_low)})")
        for condition in ordered:
            if condition == "L31_only":
                continue
            label = CONDITION_LABELS.get(condition, condition)
            effects_pp = condition_stats[condition]["effects_low_pp"]
            if len(effects_pp) < 2 or len(l31_effects_low) < 2:
                print(f"  {label:<30}: insufficient data")
                continue
            t_stat, p_val = stats.ttest_ind(effects_pp, l31_effects_low, equal_var=False)
            diff = float(np.mean(effects_pp)) - float(np.mean(l31_effects_low))
            sig = "**" if p_val < 0.05 else ("*" if p_val < 0.10 else "ns")
            print(
                f"  {label:<30}: diff={diff:+.2f}pp  t={t_stat:.3f}  p={p_val:.4f}  {sig}"
            )
    else:
        print("  L31_only data not available for comparison.")


def main() -> None:
    if not PHASE3_PATH.exists():
        print(f"Error: {PHASE3_PATH} not found. Run the Phase 3 experiment first.")
        sys.exit(1)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    data = json.loads(PHASE3_PATH.read_text())
    results = data["results"]
    all_coefs = data["coefficients"]

    pos_coefs = sorted(c for c in all_coefs if c > 0)
    per_pair_controls = load_per_pair_controls()

    print(f"N borderline pairs: {data['n_borderline_pairs']}")
    print(f"All coefficients: {all_coefs}")
    print(f"Positive coefficients: {pos_coefs}")
    print(f"N results: {len(results)}")

    present_conditions = sorted(set(r["condition"] for r in results))
    print(f"Conditions: {present_conditions}")

    coef_low = pos_coefs[0] if pos_coefs else 2641.14

    # Plot 1: dose-response
    plot1_path = ASSETS_DIR / "plot_022226_phase3_comparison.png"
    plot_dose_response(results, pos_coefs, plot1_path)

    # Plot 2: effect-size bar chart
    plot2_path = ASSETS_DIR / "plot_022226_phase3_slope_comparison.png"
    plot_slope_bars(results, coef_low, per_pair_controls, plot2_path)

    # Statistics
    print_statistics(results, pos_coefs, per_pair_controls)


if __name__ == "__main__":
    main()
