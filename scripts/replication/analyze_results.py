#!/usr/bin/env python3
"""Analysis script for steering replication experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "steering" / "replication"
RESULTS_DIR = EXP_DIR / "results"
ASSETS_DIR = EXP_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

PAIRS_PATH = RESULTS_DIR / "pairs.json"
SCREENING_PATH = RESULTS_DIR / "screening.json"
PHASE1_PATH = RESULTS_DIR / "steering_phase1.json"
PHASE2_PATH = RESULTS_DIR / "steering_phase2.json"
PHASE3_PATH = RESULTS_DIR / "steering_phase3.json"
STATISTICS_PATH = RESULTS_DIR / "statistics.json"


# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────

def parse_rate(responses: list[str]) -> float:
    """Fraction of responses that are 'a' or 'b' (not parse_fail)."""
    valid = [r for r in responses if r in ("a", "b")]
    if not len(responses):
        return 0.0
    return len(valid) / len(responses)


def p_choice(responses: list[str], choice: str) -> float:
    """P(choice) among valid responses."""
    valid = [r for r in responses if r in ("a", "b")]
    if not valid:
        return 0.5
    return sum(1 for r in valid if r == choice) / len(valid)


def bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if not values:
        return (float("nan"), float("nan"))
    arr = np.array(values)
    boot_means = [np.mean(arr[np.random.choice(len(arr), len(arr), replace=True)]) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return (float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1 - alpha))))


def linear_regression(x: list[float], y: list[float]) -> tuple[float, float, float, float]:
    """Returns (slope, intercept, slope_ci_low, slope_ci_high)."""
    if len(x) < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    result = stats.linregress(x, y)
    slope = float(result.slope)
    intercept = float(result.intercept)
    se = float(result.stderr)
    ci_low = slope - 1.96 * se
    ci_high = slope + 1.96 * se
    return slope, intercept, ci_low, ci_high


def add_error_band(ax, x_vals, y_vals_by_x, color, label, marker="o"):
    """Plot mean ± 95% CI across x values."""
    xs, ys, yerr_lo, yerr_hi = [], [], [], []
    for x in sorted(x_vals):
        vals = y_vals_by_x[x]
        if not vals:
            continue
        mean = np.mean(vals)
        lo, hi = bootstrap_ci(vals)
        xs.append(x)
        ys.append(mean)
        yerr_lo.append(mean - lo)
        yerr_hi.append(hi - mean)
    xs = np.array(xs)
    ys = np.array(ys)
    ax.plot(xs, ys, marker=marker, color=color, label=label)
    ax.fill_between(xs, ys - np.array(yerr_lo), ys + np.array(yerr_hi), alpha=0.2, color=color)
    return xs, ys


# ────────────────────────────────────────────────────────────────────────────────
# Screening analysis
# ────────────────────────────────────────────────────────────────────────────────

def analyze_screening() -> dict:
    if not PAIRS_PATH.exists() or not SCREENING_PATH.exists():
        print("[screening] Skipping: pairs.json or screening.json not found.")
        return {}

    pairs = json.loads(PAIRS_PATH.read_text())
    screening = json.loads(SCREENING_PATH.read_text())

    n_pairs = screening["n_pairs"]
    n_borderline = screening["n_borderline"]
    borderline_ids = set(screening["borderline_pair_ids"])

    pair_bins = {p["pair_id"]: p["bin"] for p in pairs}

    # Collect p_a values per pair (average across orderings)
    # We use p_a from screening results — this is P(response=='a') for that ordering
    # For borderline classification we use the pair-level flag in borderline_pair_ids
    results = screening["results"]

    # Build: pair_id -> list of p_a (across orderings, for histogram)
    pair_p_a: dict[str, list[float]] = {}
    for r in results:
        pid = r["pair_id"]
        if pid not in pair_p_a:
            pair_p_a[pid] = []
        if r["n_valid"] > 0:
            pair_p_a[pid].append(r["p_a"])

    # All p_a values for histogram
    all_p_a_borderline = []
    all_p_a_non_borderline = []
    for pid, vals in pair_p_a.items():
        for v in vals:
            if pid in borderline_ids:
                all_p_a_borderline.append(v)
            else:
                all_p_a_non_borderline.append(v)

    # Borderline rate per bin
    bin_order = sorted(set(pair_bins.values()))
    bin_borderline: dict[str, list[bool]] = {b: [] for b in bin_order}
    for pid, b in pair_bins.items():
        bin_borderline[b].append(pid in borderline_ids)

    bin_rates = {b: np.mean(vals) for b, vals in bin_borderline.items() if vals}
    bin_counts = {b: len(vals) for b, vals in bin_borderline.items()}

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Screening Overview", fontsize=13)

    # Panel 1: histogram of p_a values
    bins = np.linspace(0, 1, 21)
    ax1.hist(all_p_a_borderline, bins=bins, color="#2196F3", alpha=0.7, label=f"Borderline (n={n_borderline})")
    ax1.hist(all_p_a_non_borderline, bins=bins, color="#E0E0E0", alpha=0.7, label=f"Non-borderline (n={n_pairs - n_borderline})")
    ax1.set_xlabel("P(A) per ordering")
    ax1.set_ylabel("Count")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, None)
    ax1.legend(fontsize=9)
    pct = 100 * n_borderline / n_pairs
    ax1.text(0.05, 0.92, f"n_borderline = {n_borderline}/{n_pairs} ({pct:.1f}%)",
             transform=ax1.transAxes, fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
    ax1.set_title("P(A) distribution across orderings")

    # Panel 2: borderline rate per bin
    bins_sorted = sorted(bin_rates.keys())
    rates = [bin_rates[b] for b in bins_sorted]
    counts = [bin_counts[b] for b in bins_sorted]
    xs = np.arange(len(bins_sorted))
    bars = ax2.bar(xs, rates, color="#2196F3", alpha=0.8)
    for i, (bar, cnt) in enumerate(zip(bars, counts)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"n={cnt}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(bins_sorted, rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("Mu bin")
    ax2.set_ylabel("Borderline rate")
    ax2.set_ylim(0, 1)
    ax2.set_title("Borderline rate per mu-bin")

    plt.tight_layout()
    out_path = ASSETS_DIR / "plot_022226_screening_overview.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[screening] Saved {out_path}")

    stats_out = {
        "n_pairs": n_pairs,
        "n_borderline": n_borderline,
        "borderline_rate": n_borderline / n_pairs,
        "borderline_rate_by_bin": {b: bin_rates[b] for b in bins_sorted},
    }
    print(f"[screening] n_borderline={n_borderline}/{n_pairs} ({pct:.1f}%)")
    return stats_out


# ────────────────────────────────────────────────────────────────────────────────
# Phase 1 analysis
# ────────────────────────────────────────────────────────────────────────────────

def compute_p_steered_boost_a(result: dict) -> float | None:
    """
    For boost_a: steered task is always task in position A.
    P(steered) = P(response == 'a') regardless of ordering.
    """
    responses = result["responses"]
    valid = [r for r in responses if r in ("a", "b")]
    if not valid:
        return None
    return sum(1 for r in valid if r == "a") / len(valid)


def compute_p_steered_for_condition(result: dict) -> float | None:
    """
    Compute P(steered task picked) for a steering result.

    Steered task logic:
    - boost_a: steer task in position A → steered='a' → P(response=='a')
    - boost_b: steer task in position B → steered='b' → P(response=='b')
    - suppress_a: suppress task A → avoiding it helps B → steered='b' → P(response=='b')
    - suppress_b: suppress task B → avoiding it helps A → steered='a' → P(response=='a')
    - diff_ab: +direction on A tokens → steered='a' → P(response=='a')
    - diff_ba: +direction on B tokens → steered='b' → P(response=='b')
    """
    condition = result["condition"]
    responses = result["responses"]
    valid = [r for r in responses if r in ("a", "b")]
    if not valid:
        return None

    if condition in ("boost_a", "suppress_b", "diff_ab"):
        target = "a"
    elif condition in ("boost_b", "suppress_a", "diff_ba"):
        target = "b"
    else:
        return None

    return sum(1 for r in valid if r == target) / len(valid)


def analyze_phase1() -> dict:
    if not PHASE1_PATH.exists():
        print("[phase1] Skipping: steering_phase1.json not found.")
        return {}

    data = json.loads(PHASE1_PATH.read_text())
    results = data["results"]
    coefficients = sorted(set(r["coefficient"] for r in results))

    all_conditions = sorted(set(r["condition"] for r in results))
    print(f"[phase1] Conditions: {all_conditions}")
    print(f"[phase1] Coefficients: {coefficients}")
    print(f"[phase1] n_results: {len(results)}")

    # ── Plot 2: Combined boost_a + diff_ab dose-response ──────────────────────
    primary_conditions = {"boost_a", "diff_ab"}
    p_by_coef: dict[int, list[float]] = {c: [] for c in coefficients}
    for r in results:
        if r["condition"] not in primary_conditions:
            continue
        p = compute_p_steered_for_condition(r)
        if p is not None:
            p_by_coef[r["coefficient"]].append(p)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs_arr, ys_arr = add_error_band(ax, coefficients, p_by_coef, color="#1565C0",
                                    label="boost_a + diff_ab (mean ± 95% CI)")

    # Linear regression
    all_x, all_y = [], []
    for coef, vals in p_by_coef.items():
        all_x.extend([coef] * len(vals))
        all_y.extend(vals)

    if len(all_x) >= 2:
        slope, intercept, ci_lo, ci_hi = linear_regression(all_x, all_y)
        x_line = np.array([min(coefficients), max(coefficients)])
        ax.plot(x_line, slope * x_line + intercept, "--", color="#1565C0", alpha=0.6,
                label=f"Linear fit: slope={slope*1e4:.2f}e-4 [{ci_lo*1e4:.2f}, {ci_hi*1e4:.2f}]e-4 pp/unit")

        total_shift = (slope * max(coefficients) + intercept) - (slope * min(coefficients) + intercept)
        n_pairs_used = len(set(r["pair_id"] for r in results if r["condition"] in primary_conditions))
        textstr = f"n_pairs={n_pairs_used}, slope={slope*1e4:.2f}e-4 pp/unit\ntotal_shift={total_shift*100:.1f} pp (max−min)"
        ax.text(0.03, 0.95, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85))
    else:
        slope, ci_lo, ci_hi, total_shift = float("nan"), float("nan"), float("nan"), float("nan")

    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel("P(steered task picked)")
    ax.set_ylim(0, 1)
    ax.set_title("L31 Replication: P(steered task picked) vs steering coefficient")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.tight_layout()
    out_path = ASSETS_DIR / "plot_022226_phase1_dose_response.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[phase1] Saved {out_path}")

    # ── Plot 3: Per-condition dose-response ────────────────────────────────────
    colors = {
        "boost_a": "#1565C0", "boost_b": "#0288D1",
        "suppress_a": "#E65100", "suppress_b": "#FF8F00",
        "diff_ab": "#4CAF50", "diff_ba": "#2E7D32",
    }
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in all_conditions:
        p_by_coef_cond: dict[int, list[float]] = {c: [] for c in coefficients}
        for r in results:
            if r["condition"] != cond:
                continue
            p = compute_p_steered_for_condition(r)
            if p is not None:
                p_by_coef_cond[r["coefficient"]].append(p)

        color = colors.get(cond, "#888888")
        add_error_band(ax, coefficients, p_by_coef_cond, color=color, label=cond)

    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel("P(steered task picked)")
    ax.set_ylim(0, 1)
    ax.set_title("Phase 1: Per-condition dose-response")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(fontsize=9, ncol=2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.tight_layout()
    out_path = ASSETS_DIR / "plot_022226_phase1_by_condition.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[phase1] Saved {out_path}")

    # ── Plot 4: Per-pair slopes histogram ─────────────────────────────────────
    # For each pair, combine both orderings and compute slope of P(steered) vs coef
    # Using boost_a + diff_ab as primary
    pair_coef_p: dict[str, dict[int, list[float]]] = {}
    for r in results:
        if r["condition"] not in primary_conditions:
            continue
        pid = r["pair_id"]
        coef = r["coefficient"]
        p = compute_p_steered_for_condition(r)
        if p is None:
            continue
        if pid not in pair_coef_p:
            pair_coef_p[pid] = {c: [] for c in coefficients}
        pair_coef_p[pid][coef].append(p)

    per_pair_slopes = []
    for pid, coef_p in pair_coef_p.items():
        xs, ys = [], []
        for coef, vals in coef_p.items():
            if vals:
                xs.append(coef)
                ys.extend(vals)
        # Use mean per coefficient for the regression
        xs_mean, ys_mean = [], []
        for coef in sorted(coef_p.keys()):
            vals = coef_p[coef]
            if vals:
                xs_mean.append(coef)
                ys_mean.append(np.mean(vals))
        if len(xs_mean) >= 2:
            slope_pp, *_ = linear_regression(xs_mean, ys_mean)
            per_pair_slopes.append(slope_pp)

    if per_pair_slopes:
        slopes_arr = np.array(per_pair_slopes)
        t_stat, p_val = stats.ttest_1samp(slopes_arr, 0)
        mean_slope = float(np.mean(slopes_arr))
        pct_positive = float(np.mean(slopes_arr > 0)) * 100

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(slopes_arr, bins=20, color="#1565C0", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1.2, label="zero")
        ax.axvline(mean_slope, color="#E65100", linewidth=1.5, linestyle="--",
                   label=f"mean slope = {mean_slope*1e4:.2f}e-4")
        ax.set_xlabel("Per-pair slope (P(steered)/coefficient)")
        ax.set_ylabel("Count")
        ax.set_title("Phase 1: Per-pair slopes (boost_a + diff_ab)")
        ax.text(0.03, 0.95,
                f"n_pairs={len(per_pair_slopes)}\n"
                f"{pct_positive:.0f}% positive\n"
                f"t-test vs 0: t={t_stat:.2f}, p={p_val:.3e}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85))
        ax.legend(fontsize=9)
        plt.tight_layout()
        out_path = ASSETS_DIR / "plot_022226_phase1_per_pair_slopes.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[phase1] Saved {out_path}")
    else:
        t_stat, p_val, mean_slope, pct_positive = float("nan"), float("nan"), float("nan"), float("nan")

    stats_out = {
        "overall_slope_pp_per_unit": slope * 100 if not np.isnan(slope) else float("nan"),
        "total_shift_pp": total_shift * 100 if not np.isnan(total_shift) else float("nan"),
        "pct_pairs_positive_slope": pct_positive,
        "ttest_vs_0": {"t": float(t_stat), "p": float(p_val)},
        "slope_ci": [ci_lo, ci_hi],
    }
    print(f"[phase1] slope={slope:.3e}, total_shift={total_shift*100:.1f}pp, "
          f"{pct_positive:.0f}% pairs positive slope, t={t_stat:.2f}, p={p_val:.3e}")
    return stats_out


# ────────────────────────────────────────────────────────────────────────────────
# Phase 2 analysis
# ────────────────────────────────────────────────────────────────────────────────

def analyze_phase2() -> dict:
    if not PHASE2_PATH.exists():
        print("[phase2] Skipping: steering_phase2.json not found.")
        return {}

    data = json.loads(PHASE2_PATH.read_text())
    results = data["results"]
    coefficients = sorted(set(r["coefficient"] for r in results))

    # Add delta_mu tercile labels
    delta_mus = [r["delta_mu"] for r in results if "delta_mu" in r]
    if not delta_mus:
        print("[phase2] No delta_mu field found in results. Skipping.")
        return {}

    tercile_thresholds = np.percentile(delta_mus, [33.3, 66.7])
    print(f"[phase2] delta_mu tercile thresholds: {tercile_thresholds}")

    def tercile_label(dm: float) -> str:
        if dm <= tercile_thresholds[0]:
            return "small"
        elif dm <= tercile_thresholds[1]:
            return "medium"
        return "large"

    # Build P(boost_a pick) = P(response=='a') vs coefficient, by tercile
    # Using boost_a condition only for cleanliness
    tercile_coef_p: dict[str, dict[int, list[float]]] = {
        "small": {c: [] for c in coefficients},
        "medium": {c: [] for c in coefficients},
        "large": {c: [] for c in coefficients},
    }
    for r in results:
        if r.get("condition") != "boost_a":
            continue
        dm = r.get("delta_mu")
        if dm is None:
            continue
        label = tercile_label(dm)
        responses = r["responses"]
        valid = [x for x in responses if x in ("a", "b")]
        if not valid:
            continue
        p_a = sum(1 for x in valid if x == "a") / len(valid)
        tercile_coef_p[label][r["coefficient"]].append(p_a)

    colors_tercile = {"small": "#4CAF50", "medium": "#FF9800", "large": "#F44336"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))

    # Top panel: dose-response by tercile
    tercile_slopes: dict[str, float] = {}
    for label in ["small", "medium", "large"]:
        coef_p = tercile_coef_p[label]
        color = colors_tercile[label]
        xs_arr, ys_arr = add_error_band(ax1, coefficients, coef_p, color=color, label=f"|Δmu| {label}")
        # slope for this tercile
        all_x = [c for c in sorted(coef_p.keys()) for _ in coef_p[c]]
        all_y = [v for c in sorted(coef_p.keys()) for v in coef_p[c]]
        if len(all_x) >= 2:
            slope, *_ = linear_regression(all_x, all_y)
            tercile_slopes[label] = slope
        else:
            tercile_slopes[label] = float("nan")

    ax1.set_xlabel("Steering coefficient")
    ax1.set_ylabel("P(task A picked)")
    ax1.set_ylim(0, 1)
    ax1.set_title("Phase 2: P(boost_a pick) vs coefficient by |Δmu| tercile")
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Bottom panel: slope by tercile with bootstrap CI
    tercile_order = ["small", "medium", "large"]

    # Bootstrap slope CIs per tercile via resampling over pairs
    slope_means, slope_los, slope_his = [], [], []
    for label in tercile_order:
        # Use the slope from the full dataset per tercile
        slope_val = tercile_slopes[label]
        # Bootstrap CI by resampling pairs within tercile
        coef_p = tercile_coef_p[label]
        all_coef_val_pairs = [(c, v) for c in coef_p for v in coef_p[c]]
        if len(all_coef_val_pairs) >= 4:
            boot_slopes = []
            for _ in range(1000):
                sample = [all_coef_val_pairs[i] for i in
                          np.random.choice(len(all_coef_val_pairs), len(all_coef_val_pairs), replace=True)]
                bx = [s[0] for s in sample]
                by = [s[1] for s in sample]
                if len(set(bx)) >= 2:
                    bs, *_ = linear_regression(bx, by)
                    boot_slopes.append(bs)
            if boot_slopes:
                lo = np.percentile(boot_slopes, 2.5)
                hi = np.percentile(boot_slopes, 97.5)
            else:
                lo, hi = float("nan"), float("nan")
        else:
            lo, hi = float("nan"), float("nan")
        slope_means.append(slope_val * 100)  # convert to pp/unit * 100 = pp per 100 units
        slope_los.append((slope_val - lo) * 100 if not np.isnan(lo) else 0)
        slope_his.append((hi - slope_val) * 100 if not np.isnan(hi) else 0)

    xs_bar = np.arange(len(tercile_order))
    bars = ax2.bar(xs_bar, slope_means, color=[colors_tercile[t] for t in tercile_order],
                   alpha=0.8, yerr=[slope_los, slope_his], capsize=5)
    ax2.set_xticks(xs_bar)
    ax2.set_xticklabels([f"|Δmu| {t}" for t in tercile_order])
    ax2.set_ylabel("Slope (pp/unit × 100)")
    ax2.set_title("Phase 2: Steering slope by |Δmu| tercile")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylim(min(0, min(s - e for s, e in zip(slope_means, slope_los))) * 1.2 if slope_means else -0.1,
                 max(0, max(s + e for s, e in zip(slope_means, slope_his))) * 1.2 if slope_means else 0.1)

    plt.tight_layout()
    out_path = ASSETS_DIR / "plot_022226_phase2_utility_bin.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[phase2] Saved {out_path}")

    # Kruskal-Wallis test for tercile effect on P(boost_a pick)
    groups = []
    for label in tercile_order:
        coef_p = tercile_coef_p[label]
        group_vals = [v for coef in coef_p for v in coef_p[coef]]
        groups.append(group_vals)
    kw_stat, kw_p = stats.kruskal(*[g for g in groups if g]) if any(groups) else (float("nan"), float("nan"))

    stats_out = {
        "slope_by_tercile_pp_per_unit": {t: tercile_slopes[t] * 100 for t in tercile_order},
        "kruskal_wallis": {"stat": float(kw_stat), "p": float(kw_p)},
    }
    print(f"[phase2] Slopes by tercile: { {t: f'{tercile_slopes[t]*1e4:.2f}e-4' for t in tercile_order} }")
    print(f"[phase2] Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.3e}")
    return stats_out


# ────────────────────────────────────────────────────────────────────────────────
# Phase 3 analysis
# ────────────────────────────────────────────────────────────────────────────────

def analyze_phase3() -> dict:
    if not PHASE3_PATH.exists():
        print("[phase3] Skipping: steering_phase3.json not found.")
        return {}

    data = json.loads(PHASE3_PATH.read_text())
    results = data["results"]
    coefficients = sorted(set(r["coefficient"] for r in results))

    all_conditions = sorted(set(r["condition"] for r in results))
    print(f"[phase3] Conditions found: {all_conditions}")

    colors_p3 = {
        "L31_only": "#1565C0",
        "L31_L37_same_split": "#42A5F5",
        "L31_L37_same_full": "#0288D1",
        "L31_L37_layer_split": "#4CAF50",
        "L31_L43_layer_split": "#FF9800",
        "L31_L37_L43_layer_split": "#F44336",
    }

    fig, ax = plt.subplots(figsize=(11, 6))

    condition_slopes: dict[str, float] = {}
    for cond in all_conditions:
        p_by_coef: dict[int, list[float]] = {c: [] for c in coefficients}
        for r in results:
            if r["condition"] != cond:
                continue
            responses = r["responses"]
            valid = [x for x in responses if x in ("a", "b")]
            if not valid:
                continue
            # Treat steered direction as 'a' for all conditions (boost_a logic)
            p_a = sum(1 for x in valid if x == "a") / len(valid)
            p_by_coef[r["coefficient"]].append(p_a)

        color = colors_p3.get(cond, "#888888")
        add_error_band(ax, coefficients, p_by_coef, color=color, label=cond)

        all_x = [c for c in sorted(p_by_coef.keys()) for _ in p_by_coef[c]]
        all_y = [v for c in sorted(p_by_coef.keys()) for v in p_by_coef[c]]
        if len(all_x) >= 2:
            slope, *_ = linear_regression(all_x, all_y)
            condition_slopes[cond] = slope
        else:
            condition_slopes[cond] = float("nan")

    # Slope comparison textbox
    slope_lines = [f"{cond}: {condition_slopes.get(cond, float('nan'))*1e4:.2f}e-4"
                   for cond in all_conditions]
    textstr = "Slopes (pp/unit):\n" + "\n".join(slope_lines)
    ax.text(1.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85))

    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel("P(task A steered picked)")
    ax.set_ylim(0, 1)
    ax.set_title("Phase 3: Multi-layer steering comparison")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.tight_layout()
    out_path = ASSETS_DIR / "plot_022226_phase3_multi_layer.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[phase3] Saved {out_path}")

    stats_out = {
        "slope_by_condition_pp_per_unit": {c: condition_slopes.get(c, float("nan")) * 100
                                            for c in all_conditions},
    }
    slope_strs = {c: f"{condition_slopes.get(c, float('nan'))*1e4:.2f}e-4" for c in all_conditions}
    print(f"[phase3] Slopes: {slope_strs}")
    return stats_out


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Steering Replication Analysis ===")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Assets dir:  {ASSETS_DIR}\n")

    all_stats: dict = {}

    screening_stats = analyze_screening()
    if screening_stats:
        all_stats["screening"] = screening_stats

    phase1_stats = analyze_phase1()
    if phase1_stats:
        all_stats["phase1"] = phase1_stats

    phase2_stats = analyze_phase2()
    if phase2_stats:
        all_stats["phase2"] = phase2_stats

    phase3_stats = analyze_phase3()
    if phase3_stats:
        all_stats["phase3"] = phase3_stats

    if all_stats:
        STATISTICS_PATH.write_text(json.dumps(all_stats, indent=2))
        print(f"\n[main] Statistics saved to {STATISTICS_PATH}")

    print("\n=== Summary ===")
    phases_run = list(all_stats.keys())
    if phases_run:
        print(f"Analyzed: {', '.join(phases_run)}")
    else:
        print("No data files found. Nothing analyzed.")
    skipped = []
    if not SCREENING_PATH.exists():
        skipped.append("screening")
    if not PHASE1_PATH.exists():
        skipped.append("phase1")
    if not PHASE2_PATH.exists():
        skipped.append("phase2")
    if not PHASE3_PATH.exists():
        skipped.append("phase3")
    if skipped:
        print(f"Skipped (missing files): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
