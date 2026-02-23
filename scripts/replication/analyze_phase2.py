#!/usr/bin/env python3
"""Phase 2 analysis: decisive pairs steering by |delta_mu| tercile."""

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

PHASE2_PATH = RESULTS_DIR / "steering_phase2.json"

TERCILE_COLORS = {
    "small": "#2171b5",
    "medium": "#fd8d3c",
    "large": "#d7191c",
}
TERCILE_ORDER = ["small", "medium", "large"]


def p_a(responses: list[str]) -> float | None:
    valid = [r for r in responses if r in ("a", "b")]
    if not valid:
        return None
    return sum(1 for r in valid if r == "a") / len(valid)


def bootstrap_mean_ci(
    values: list[float], n_boot: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    point = float(arr.mean())
    boot = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def linreg(x: list[float], y: list[float]) -> tuple[float, float, float, float]:
    """Returns (slope, intercept, ci_lo, ci_hi) with 95% CI on slope."""
    res = stats.linregress(x, y)
    slope = float(res.slope)
    intercept = float(res.intercept)
    ci_lo = slope - 1.96 * float(res.stderr)
    ci_hi = slope + 1.96 * float(res.stderr)
    return slope, intercept, ci_lo, ci_hi


def assign_tercile(delta_mu: float, t1: float, t2: float) -> str:
    if delta_mu <= t1:
        return "small"
    if delta_mu <= t2:
        return "medium"
    return "large"


def build_tercile_data(
    results: list[dict], t1: float, t2: float
) -> dict[str, dict[str, list]]:
    """
    Returns nested dict:
      tercile -> {
        "control_p_a": [floats],          # P(a) from control rows
        "boost_coefs": [float],            # one per boost_a row
        "boost_p_a": [float],             # paired with boost_coefs
        "pair_ids": set[str],
      }
    """
    out: dict[str, dict] = {
        t: {"control_p_a": [], "boost_coefs": [], "boost_p_a": [], "pair_ids": set()}
        for t in TERCILE_ORDER
    }
    for r in results:
        dm = r.get("delta_mu")
        if dm is None:
            continue
        label = assign_tercile(abs(dm), t1, t2)
        cond = r.get("condition")
        pa = p_a(r["responses"])
        if pa is None:
            continue
        out[label]["pair_ids"].add(r["pair_id"])
        if cond == "control":
            out[label]["control_p_a"].append(pa)
        elif cond == "boost_a":
            out[label]["boost_coefs"].append(float(r["coefficient"]))
            out[label]["boost_p_a"].append(pa)
    return out


def compute_per_pair_slopes(
    results: list[dict], t1: float, t2: float
) -> list[dict]:
    """
    For each pair, aggregate P(a) across orderings (both original and swapped)
    at each positive coefficient.  Return list of dicts with pair stats.
    """
    # pair_id -> coef -> [p_a values]
    pair_data: dict[str, dict[float, list[float]]] = {}
    pair_meta: dict[str, dict] = {}

    for r in results:
        if r.get("condition") != "boost_a":
            continue
        coef = float(r["coefficient"])
        if coef <= 0:
            continue
        pid = r["pair_id"]
        pa = p_a(r["responses"])
        if pa is None:
            continue
        if pid not in pair_data:
            pair_data[pid] = {}
            pair_meta[pid] = {
                "delta_mu": abs(r.get("delta_mu", float("nan"))),
                "delta_mu_bin": r.get("delta_mu_bin", ""),
            }
        pair_data[pid].setdefault(coef, []).append(pa)

    rows = []
    for pid, coef_map in pair_data.items():
        xs, ys = [], []
        for coef, vals in coef_map.items():
            xs.append(coef)
            ys.append(float(np.mean(vals)))
        if len(xs) < 2:
            continue
        slope, _, _, _ = linreg(xs, ys)
        dm = pair_meta[pid]["delta_mu"]
        label = assign_tercile(dm, t1, t2)
        rows.append(
            {
                "pair_id": pid,
                "delta_mu": dm,
                "tercile": label,
                "slope": slope,
            }
        )
    return rows


# ── Plot 1: dose-response by tercile ──────────────────────────────────────────

def plot_dose_response(
    results: list[dict], t1: float, t2: float, save_path: Path
) -> None:
    sns.set_style("whitegrid")

    # Collect P(a) per tercile per coefficient (boost_a only + control at coef=0)
    # tercile -> coef -> [p_a]
    coef_pa: dict[str, dict[float, list[float]]] = {
        t: {} for t in TERCILE_ORDER
    }
    ctrl_pa: dict[str, list[float]] = {t: [] for t in TERCILE_ORDER}

    for r in results:
        dm = r.get("delta_mu")
        if dm is None:
            continue
        label = assign_tercile(abs(dm), t1, t2)
        cond = r.get("condition")
        pa = p_a(r["responses"])
        if pa is None:
            continue

        if cond == "control":
            ctrl_pa[label].append(pa)
        elif cond == "boost_a":
            coef = float(r["coefficient"])
            coef_pa[label].setdefault(coef, []).append(pa)

    # Build x axis: sorted unique positive coefficients + 0 (control point)
    all_pos_coefs = sorted(
        {coef for t in TERCILE_ORDER for coef in coef_pa[t]}
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for label in TERCILE_ORDER:
        color = TERCILE_COLORS[label]

        # Control as separate dashed horizontal marker
        if ctrl_pa[label]:
            ctrl_mean, ctrl_lo, ctrl_hi = bootstrap_mean_ci(ctrl_pa[label])
            ax.axhline(
                ctrl_mean,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.6,
                label=f"|Δmu| {label} control P(a)={ctrl_mean:.3f}",
            )

        # Dose-response line at positive coefficients
        xs, ys, y_los, y_his = [], [], [], []
        for coef in all_pos_coefs:
            vals = coef_pa[label].get(coef, [])
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

        ax.plot(xs_arr, ys_arr, color=color, linewidth=2.0, marker="o", markersize=5,
                label=f"|Δmu| {label} (boost_a)")
        ax.fill_between(xs_arr, y_los_arr, y_his_arr, color=color, alpha=0.15)

    ax.set_xlabel("Steering coefficient", fontsize=11)
    ax.set_ylabel("P(steered = 'a')", fontsize=11)
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Phase 2: Steering Effect by Utility Gap (boost_a)", fontsize=12, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Plot 2: per-pair slope vs |delta_mu| scatter ──────────────────────────────

def plot_slope_scatter(
    pair_rows: list[dict], t1: float, t2: float, save_path: Path
) -> None:
    sns.set_style("whitegrid")

    dm_vals = [row["delta_mu"] for row in pair_rows]
    slope_vals = [row["slope"] for row in pair_rows]
    tercile_labels = [row["tercile"] for row in pair_rows]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Scatter colored by tercile
    for label in TERCILE_ORDER:
        idxs = [i for i, t in enumerate(tercile_labels) if t == label]
        x = [dm_vals[i] for i in idxs]
        y = [slope_vals[i] for i in idxs]
        ax.scatter(x, y, color=TERCILE_COLORS[label], label=f"|Δmu| {label}",
                   s=50, alpha=0.8, zorder=3)

    # Regression line over all pairs
    if len(dm_vals) >= 3:
        slope_reg, intercept_reg, ci_lo_reg, ci_hi_reg = linreg(dm_vals, slope_vals)
        x_line = np.linspace(min(dm_vals), max(dm_vals), 200)
        y_line = slope_reg * x_line + intercept_reg
        # CI band via bootstrapping
        rng = np.random.default_rng(0)
        dm_arr = np.array(dm_vals)
        sl_arr = np.array(slope_vals)
        boot_lines = []
        for _ in range(1000):
            idx = rng.integers(0, len(dm_arr), size=len(dm_arr))
            if len(set(idx)) < 2:
                continue
            res = stats.linregress(dm_arr[idx], sl_arr[idx])
            boot_lines.append(res.slope * x_line + res.intercept)
        if boot_lines:
            boot_arr = np.array(boot_lines)
            ci_lo_band = np.percentile(boot_arr, 2.5, axis=0)
            ci_hi_band = np.percentile(boot_arr, 97.5, axis=0)
            ax.fill_between(x_line, ci_lo_band, ci_hi_band, color="gray", alpha=0.2, zorder=2)
        ax.plot(x_line, y_line, color="black", linewidth=1.5, zorder=4,
                label=f"Regression (slope={slope_reg*1e4:.2f}e-4/unit)")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlabel("|Δmu| (utility gap)", fontsize=11)
    ax.set_ylabel("Per-pair slope of P(a) ~ coefficient", fontsize=11)
    ax.set_title("Per-pair slope vs |Δmu| (Phase 2 decisive pairs)", fontsize=12, fontweight="bold")
    ax.set_xlim(left=0)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tercile boundary lines
    ax.axvline(t1, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(t2, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_summary(
    results: list[dict], pair_rows: list[dict], t1: float, t2: float
) -> dict:
    # Control P(a) per tercile
    ctrl_pa: dict[str, list[float]] = {t: [] for t in TERCILE_ORDER}
    boost_pa_at_max: dict[str, list[float]] = {t: [] for t in TERCILE_ORDER}

    max_coef = max(
        (float(r["coefficient"]) for r in results if r.get("condition") == "boost_a"),
        default=float("nan"),
    )

    for r in results:
        dm = r.get("delta_mu")
        if dm is None:
            continue
        label = assign_tercile(abs(dm), t1, t2)
        pa = p_a(r["responses"])
        if pa is None:
            continue
        cond = r.get("condition")
        if cond == "control":
            ctrl_pa[label].append(pa)
        elif cond == "boost_a" and abs(float(r["coefficient"]) - max_coef) < 1.0:
            boost_pa_at_max[label].append(pa)

    # Per-pair slopes per tercile
    slopes_by_tercile: dict[str, list[float]] = {t: [] for t in TERCILE_ORDER}
    for row in pair_rows:
        slopes_by_tercile[row["tercile"]].append(row["slope"])

    summary: dict = {
        "tercile_thresholds": [t1, t2],
        "control_p_a": {},
        f"boost_a_at_{int(max_coef)}": {},
        "slope_pp_per_1k": {},
        "ttest_p": {},
    }
    boost_key = f"boost_a_at_{int(max_coef)}"

    for label in TERCILE_ORDER:
        ctrl = float(np.mean(ctrl_pa[label])) if ctrl_pa[label] else float("nan")
        boost_mean = (
            float(np.mean(boost_pa_at_max[label])) if boost_pa_at_max[label] else float("nan")
        )
        slopes = slopes_by_tercile[label]
        if slopes:
            slope_mean = float(np.mean(slopes))
            t_stat, p_val = stats.ttest_1samp(slopes, 0)
        else:
            slope_mean = float("nan")
            t_stat, p_val = float("nan"), float("nan")

        summary["control_p_a"][label] = round(ctrl, 4)
        summary[boost_key][label] = round(boost_mean, 4)
        summary["slope_pp_per_1k"][label] = round(slope_mean * 1000 * 100, 4)  # pp per 1k units
        summary["ttest_p"][label] = round(float(p_val), 4)

    return summary


def print_statistics(pair_rows: list[dict], summary: dict) -> None:
    print("\n=== Phase 2 Statistics ===")
    print(f"Tercile thresholds: {summary['tercile_thresholds']}")
    print(f"N pairs with slopes: {len(pair_rows)}")

    boost_key = [k for k in summary if k.startswith("boost_a_at_")][0]

    print("\nControl P(a) per tercile:")
    for label in TERCILE_ORDER:
        ctrl = summary["control_p_a"][label]
        print(f"  {label:8s}: {ctrl:.4f}")

    print(f"\nBoost P(a) at max coef ({boost_key}) per tercile:")
    for label in TERCILE_ORDER:
        val = summary[boost_key][label]
        ctrl = summary["control_p_a"][label]
        delta = val - ctrl if not (np.isnan(val) or np.isnan(ctrl)) else float("nan")
        print(f"  {label:8s}: {val:.4f}  (delta vs control: {delta:+.4f})")

    print("\nMean slope per tercile (pp per 1k coefficient):")
    slopes_list = []
    for label in TERCILE_ORDER:
        sl = summary["slope_pp_per_1k"][label]
        p = summary["ttest_p"][label]
        slopes_list.append(sl)
        print(f"  {label:8s}: {sl:.3f} pp/1k  (t-test vs 0: p={p:.4f})")

    # Check if slope decreases with |delta_mu|
    if all(not np.isnan(s) for s in slopes_list):
        decreasing = slopes_list[0] >= slopes_list[1] >= slopes_list[2]
        monotone_str = "YES (expected)" if decreasing else "NO (unexpected)"
        print(f"\nDoes slope decrease with |Δmu|? {monotone_str}")
        print(f"  small={slopes_list[0]:.3f} -> medium={slopes_list[1]:.3f} -> large={slopes_list[2]:.3f}")
    else:
        print("\nCannot assess monotonicity (missing slope data).")

    print("\nSummary JSON:")
    print(json.dumps(summary, indent=2))


def main() -> None:
    if not PHASE2_PATH.exists():
        print(f"Error: {PHASE2_PATH} not found. Run the Phase 2 experiment first.")
        sys.exit(1)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    data = json.loads(PHASE2_PATH.read_text())
    results = data["results"]

    # Use stored thresholds if available, otherwise compute from delta_mu distribution
    if "tercile_thresholds" in data:
        t1, t2 = data["tercile_thresholds"]
    else:
        delta_mus = [abs(r["delta_mu"]) for r in results if "delta_mu" in r]
        t1, t2 = float(np.percentile(delta_mus, 33.3)), float(np.percentile(delta_mus, 66.7))

    print(f"Tercile thresholds: t1={t1:.4f}, t2={t2:.4f}")
    print(f"N results: {len(results)}")

    conditions = sorted(set(r.get("condition", "") for r in results))
    coefs = sorted(set(r.get("coefficient", 0) for r in results))
    print(f"Conditions: {conditions}")
    print(f"Coefficients: {coefs}")

    pair_rows = compute_per_pair_slopes(results, t1, t2)
    print(f"N pairs with slope estimates: {len(pair_rows)}")

    # Plot 1: dose-response by tercile
    plot1_path = ASSETS_DIR / "plot_022226_phase2_by_tercile.png"
    plot_dose_response(results, t1, t2, plot1_path)

    # Plot 2: per-pair slope vs |delta_mu|
    plot2_path = ASSETS_DIR / "plot_022226_phase2_slope_vs_delta_mu.png"
    plot_slope_scatter(pair_rows, t1, t2, plot2_path)

    # Summary statistics
    summary = compute_summary(results, pair_rows, t1, t2)
    print_statistics(pair_rows, summary)


if __name__ == "__main__":
    main()
