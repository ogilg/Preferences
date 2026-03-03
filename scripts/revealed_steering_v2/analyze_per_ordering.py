"""Per-ordering analysis of revealed steering v2 follow-up.

Separates AB and BA orderings to disentangle position bias from steering.
Produces:
  1. Dose-response: P(A) vs multiplier for each ordering separately + overlay + aggregate
  2. Steerability vs decidedness: with decidedness = |P(A|ordering) - 0.5|
     computed within the same ordering used for steering
  3. Per-ordering baseline distributions
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FOLLOWUP_DIR = Path("experiments/revealed_steering_v2/followup")
CHECKPOINT_PATH = FOLLOWUP_DIR / "checkpoint.jsonl"
ASSETS_DIR = FOLLOWUP_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

COHERENCE_BOUNDARY = 0.05


def load_records():
    records = []
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def bootstrap_ci(values, n_boot=10000, ci=0.95):
    rng = np.random.default_rng(42)
    arr = np.array(values)
    n = len(arr)
    boot = [np.mean(rng.choice(arr, size=n, replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.percentile(boot, 100 * alpha)), float(np.percentile(boot, 100 * (1 - alpha)))


def compute_per_ordering_dose_response(records, pair_subset=None):
    """P(A) vs multiplier, separated by ordering."""
    steering = [r for r in records if r["condition"] == "probe" and r["choice_original"] is not None]
    if pair_subset is not None:
        steering = [r for r in steering if r["pair_id"] in pair_subset]

    by_ord_mult = {0: defaultdict(list), 1: defaultdict(list)}
    for r in steering:
        chose_a = 1 if r["choice_original"] == "a" else 0
        by_ord_mult[r["ordering"]][r["multiplier"]].append(chose_a)

    results = {}
    for ordering in [0, 1]:
        rows = []
        for mult in sorted(by_ord_mult[ordering].keys()):
            choices = by_ord_mult[ordering][mult]
            pa = np.mean(choices)
            ci_lo, ci_hi = bootstrap_ci(choices)
            rows.append({
                "multiplier": mult,
                "p_a": round(float(pa), 4),
                "ci_lo": round(float(ci_lo), 4),
                "ci_hi": round(float(ci_hi), 4),
                "n": len(choices),
            })
        results[ordering] = rows
    return results


def compute_aggregate_dose_response(dose_by_ord):
    """Aggregate steering effect: average P(A)-shift across orderings.

    For AB ordering, steering toward A means P(A) increases.
    For BA ordering, steering toward A means P(A) decreases.
    Aggregate effect = (P(A|AB,mult) - P(A|AB,0) - (P(A|BA,mult) - P(A|BA,0))) / 2
    This is equivalent to the ordering-difference metric / 2.
    """
    ab_baseline = next(r["p_a"] for r in dose_by_ord[0] if r["multiplier"] == 0.0)
    ba_baseline = next(r["p_a"] for r in dose_by_ord[1] if r["multiplier"] == 0.0)

    rows = []
    for ab_row, ba_row in zip(dose_by_ord[0], dose_by_ord[1]):
        mult = ab_row["multiplier"]
        ab_shift = ab_row["p_a"] - ab_baseline  # positive = toward A
        ba_shift = ba_baseline - ba_row["p_a"]   # positive = toward A (BA baseline is low)
        avg_shift = (ab_shift + ba_shift) / 2
        rows.append({
            "multiplier": mult,
            "ab_shift": round(ab_shift, 4),
            "ba_shift": round(ba_shift, 4),
            "avg_shift": round(avg_shift, 4),
        })
    return rows


def classify_pairs(pair_baselines):
    """Classify pairs as decided (both orderings agree) or borderline (anything else).

    Decided: P(A|AB) and P(A|BA) are both 0 or both 1 — the model picks the same
    task regardless of presentation order.
    Borderline: orderings disagree, or at least one ordering is non-extreme.
    """
    decided, borderline = set(), set()
    for pid, bl in pair_baselines.items():
        if 0 not in bl or 1 not in bl:
            continue
        pa_ab, pa_ba = bl[0], bl[1]
        both_a = pa_ab == 1.0 and pa_ba == 1.0
        both_b = pa_ab == 0.0 and pa_ba == 0.0
        if both_a or both_b:
            decided.add(pid)
        else:
            borderline.add(pid)
    return decided, borderline


def compute_per_ordering_baseline(records):
    """Per-pair, per-ordering baseline P(A)."""
    baseline = [r for r in records if r["condition"] == "baseline" and r["choice_original"] is not None]

    by_pair_ord = defaultdict(lambda: {0: [], 1: []})
    for r in baseline:
        chose_a = 1 if r["choice_original"] == "a" else 0
        by_pair_ord[r["pair_id"]][r["ordering"]].append(chose_a)

    pair_baselines = {}
    for pid, ord_data in by_pair_ord.items():
        pair_baselines[pid] = {}
        for ordering in [0, 1]:
            if ord_data[ordering]:
                pair_baselines[pid][ordering] = np.mean(ord_data[ordering])
    return pair_baselines


def compute_per_ordering_steerability(records, pair_baselines, mult=0.02):
    """Steerability and decidedness computed per-ordering."""
    steering = [r for r in records if r["condition"] == "probe"
                and r["choice_original"] is not None
                and r["multiplier"] == mult]

    steered_by_pair_ord = defaultdict(lambda: {0: [], 1: []})
    for r in steering:
        chose_a = 1 if r["choice_original"] == "a" else 0
        steered_by_pair_ord[r["pair_id"]][r["ordering"]].append(chose_a)

    results = {0: [], 1: []}
    for pid, bl in pair_baselines.items():
        for ordering in [0, 1]:
            if ordering not in bl:
                continue
            pa_bl = bl[ordering]
            steered_trials = steered_by_pair_ord[pid][ordering]
            if not steered_trials:
                continue
            pa_steered = np.mean(steered_trials)
            decidedness = abs(pa_bl - 0.5)
            shift = pa_steered - pa_bl
            results[ordering].append({
                "pair_id": pid,
                "pa_baseline": pa_bl,
                "pa_steered": pa_steered,
                "decidedness": decidedness,
                "shift": shift,
                "abs_shift": abs(shift),
                "n_steered": len(steered_trials),
            })
    return results


def _add_coherence_boundary(ax):
    ax.axvline(x=-COHERENCE_BOUNDARY, color="black", linestyle="--", alpha=0.3)
    ax.axvline(x=COHERENCE_BOUNDARY, color="black", linestyle="--", alpha=0.3)


def _split_coherent(rows):
    """Split rows into coherent (|mult| <= boundary) and incoherent."""
    coherent = [r for r in rows if abs(r["multiplier"]) <= COHERENCE_BOUNDARY]
    incoherent = [r for r in rows if abs(r["multiplier"]) > COHERENCE_BOUNDARY]
    return coherent, incoherent


def _plot_dose_line(ax, rows, color, label=None, marker="o"):
    """Plot dose-response with faded points outside coherence boundary."""
    coherent, incoherent = _split_coherent(rows)
    left = [r for r in incoherent if r["multiplier"] < -COHERENCE_BOUNDARY]
    right = [r for r in incoherent if r["multiplier"] > COHERENCE_BOUNDARY]

    # Coherent: full opacity
    mults_c = [r["multiplier"] for r in coherent]
    pa_c = [r["p_a"] for r in coherent]
    ci_lo_c = [r["ci_lo"] for r in coherent]
    ci_hi_c = [r["ci_hi"] for r in coherent]
    ax.fill_between(mults_c, ci_lo_c, ci_hi_c, alpha=0.2, color=color)
    ax.plot(mults_c, pa_c, f"{marker}-", color=color, label=label)

    # Incoherent: faded, split left/right to avoid cross-middle band
    for segment in [left, right]:
        if not segment:
            continue
        mults_s = [r["multiplier"] for r in segment]
        pa_s = [r["p_a"] for r in segment]
        ci_lo_s = [r["ci_lo"] for r in segment]
        ci_hi_s = [r["ci_hi"] for r in segment]
        ax.fill_between(mults_s, ci_lo_s, ci_hi_s, alpha=0.06, color=color)
        ax.plot(mults_s, pa_s, f"{marker}-", color=color, alpha=0.25, markersize=4)


def plot_dose_response(dose_by_ord):
    ord_labels = {0: "AB ordering (A first)", 1: "BA ordering (B first)"}
    colors = {0: "C0", 1: "C3"}

    # Side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ordering, ax in zip([0, 1], axes):
        _plot_dose_line(ax, dose_by_ord[ordering], colors[ordering])
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        _add_coherence_boundary(ax)
        ax.set_xlabel("Steering multiplier")
        ax.set_ylabel("P(choose A)")
        ax.set_title(ord_labels[ordering])
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.16, 0.16)

    fig.suptitle("Dose-response by ordering (separated)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_030326_dose_response_per_ordering.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved dose_response_per_ordering plot")

    # Overlay
    fig, ax = plt.subplots(figsize=(8, 5))
    for ordering in [0, 1]:
        _plot_dose_line(ax, dose_by_ord[ordering], colors[ordering],
                        label=ord_labels[ordering])
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    _add_coherence_boundary(ax)
    ax.set_xlabel("Steering multiplier")
    ax.set_ylabel("P(choose A)")
    ax.set_title("P(A) by multiplier — AB vs BA ordering")
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.16, 0.16)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_030326_dose_response_overlay.png", dpi=150)
    plt.close()
    print("Saved dose_response_overlay plot")


def plot_aggregate_dose_response(agg, dose_by_ord):
    """Aggregate steering effect: avg shift toward A across both orderings."""
    fig, ax = plt.subplots(figsize=(8, 5))

    all_mults = [r["multiplier"] for r in agg]

    # Per-ordering shifts: full line faded, coherent region highlighted
    for ordering, label, color in [
        (0, "AB ordering", "C0"),
        (1, "BA ordering", "C3"),
    ]:
        key = "ab_shift" if ordering == 0 else "ba_shift"
        all_vals = [r[key] for r in agg]
        coherent = [r for r in agg if abs(r["multiplier"]) <= COHERENCE_BOUNDARY]
        mults_c = [r["multiplier"] for r in coherent]
        vals_c = [r[key] for r in coherent]

        # Full line at low opacity (provides connecting segments)
        ax.plot(all_mults, all_vals, "o--", color=color, alpha=0.15, markersize=3)
        # Coherent region highlighted
        ax.plot(mults_c, vals_c, "o--", color=color, alpha=0.5, label=label, markersize=4)

    # Mean: full line faded, coherent region solid
    all_avg = [r["avg_shift"] for r in agg]
    coherent = [r for r in agg if abs(r["multiplier"]) <= COHERENCE_BOUNDARY]
    mults_c = [r["multiplier"] for r in coherent]
    avg_c = [r["avg_shift"] for r in coherent]

    ax.plot(all_mults, all_avg, "s-", color="black", alpha=0.2, markersize=4, zorder=5)
    ax.plot(mults_c, avg_c, "s-", color="black", label="Mean (both orderings)", zorder=5)

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    _add_coherence_boundary(ax)
    ax.set_xlabel("Steering coefficient (fraction of mean L31 norm)")
    ax.set_ylabel(r"$\Delta P(\mathrm{choose\ A})$ from baseline")
    ax.set_title("Steering revealed preferences in pairwise choices")
    ax.set_ylim(-0.25, 0.25)
    ax.set_xlim(-0.16, 0.16)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_030326_aggregate_steering.png", dpi=150)
    plt.close()
    print("Saved aggregate_steering plot")


def plot_decided_vs_borderline(agg_decided, agg_borderline, n_decided, n_borderline):
    """Aggregate steering effect for decided vs borderline pairs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for agg, color, label in [
        (agg_decided, "C2", f"Decided (n={n_decided})"),
        (agg_borderline, "C1", f"Borderline (n={n_borderline})"),
    ]:
        coherent = [r for r in agg if abs(r["multiplier"]) <= COHERENCE_BOUNDARY]
        left = [r for r in agg if r["multiplier"] < -COHERENCE_BOUNDARY]
        right = [r for r in agg if r["multiplier"] > COHERENCE_BOUNDARY]

        mults_c = [r["multiplier"] for r in coherent]
        avg_c = [r["avg_shift"] for r in coherent]
        ax.plot(mults_c, avg_c, "o-", color=color, label=label)

        for segment in [left, right]:
            if not segment:
                continue
            mults_s = [r["multiplier"] for r in segment]
            avg_s = [r["avg_shift"] for r in segment]
            ax.plot(mults_s, avg_s, "o-", color=color, alpha=0.25, markersize=4)

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    _add_coherence_boundary(ax)
    ax.set_xlabel("Steering multiplier")
    ax.set_ylabel("Mean steering effect (ΔP toward A)")
    ax.set_title("Aggregate steering: decided vs borderline pairs")
    ax.set_ylim(-0.25, 0.25)
    ax.set_xlim(-0.16, 0.16)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_030326_decided_vs_borderline.png", dpi=150)
    plt.close()
    print("Saved decided_vs_borderline plot")


def plot_steerability_vs_decidedness(steer_by_ord):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ord_labels = {0: "AB ordering (A first)", 1: "BA ordering (B first)"}
    colors = {0: "C0", 1: "C3"}

    for ordering, ax in zip([0, 1], axes):
        data = steer_by_ord[ordering]
        decidedness = np.array([d["decidedness"] for d in data])
        abs_shift = np.array([d["abs_shift"] for d in data])

        ax.scatter(decidedness, abs_shift, alpha=0.2, s=12, color=colors[ordering])

        bins = np.linspace(0, 0.5, 6)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_means = []
        bin_sems = []
        bin_ns = []
        for i in range(len(bins) - 1):
            in_bin = (decidedness >= bins[i]) & (decidedness < bins[i + 1])
            n = in_bin.sum()
            bin_ns.append(n)
            if n > 0:
                bin_means.append(abs_shift[in_bin].mean())
                bin_sems.append(abs_shift[in_bin].std() / np.sqrt(n))
            else:
                bin_means.append(float("nan"))
                bin_sems.append(float("nan"))

        ax.errorbar(bin_centers, bin_means, yerr=bin_sems,
                     fmt="s-", color="black", capsize=3, zorder=5, label="Bin mean ± SEM")

        r = np.corrcoef(decidedness, abs_shift)[0, 1]
        ax.set_xlabel("Decidedness = |P(A|same ordering) − 0.5|")
        ax.set_ylabel("|ΔP(A)| at mult=+0.02")
        ax.set_title(f"{ord_labels[ordering]}\nr = {r:.3f}, n = {len(data)}")
        ax.set_xlim(0, 0.55)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)

        print(f"\n{ord_labels[ordering]} — steerability by decidedness bin:")
        for i in range(len(bins) - 1):
            print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): n={bin_ns[i]}, "
                  f"mean |shift|={bin_means[i]:.3f}")

    fig.suptitle("Steerability vs decidedness (per-ordering)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_030326_steerability_per_ordering.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved steerability_per_ordering plot")


def plot_baseline_per_ordering(pair_baselines):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ord_labels = {0: "AB ordering (A first)", 1: "BA ordering (B first)"}
    colors = {0: "C0", 1: "C3"}

    for ordering, ax in zip([0, 1], axes):
        pa_vals = [bl[ordering] for bl in pair_baselines.values() if ordering in bl]
        pa_arr = np.array(pa_vals)
        at_zero = np.sum(pa_arr == 0.0)
        at_one = np.sum(pa_arr == 1.0)
        at_extreme = at_zero + at_one

        ax.hist(pa_vals, bins=11, range=(0, 1), edgecolor="black", alpha=0.7, color=colors[ordering])
        ax.set_xlabel(f"P(A | {ord_labels[ordering]})")
        ax.set_ylabel("Number of pairs")
        ax.set_title(f"{ord_labels[ordering]}\n"
                     f"n={len(pa_vals)}, P(A)=0: {at_zero}, P(A)=1: {at_one} "
                     f"({100*at_extreme/len(pa_vals):.0f}% extreme)")
        ax.set_xlim(0, 1)

    fig.suptitle("Baseline P(A) distribution by ordering (10 trials each)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_030326_baseline_per_ordering.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved baseline_per_ordering plot")


def main():
    print("Loading data...")
    records = load_records()
    print(f"Total records: {len(records)}")

    # Use original 300 pairs for all primary plots
    old_pids = {f"pair_{i:04d}" for i in range(300)}
    print(f"Using original 300 pairs for primary analysis")

    # 1. Per-ordering dose-response (original 300)
    print("\n=== Per-ordering dose-response (original 300) ===")
    dose_by_ord = compute_per_ordering_dose_response(records, pair_subset=old_pids)
    for ordering in [0, 1]:
        label = "AB" if ordering == 0 else "BA"
        print(f"\n{label} ordering:")
        print(f"  {'mult':>7} | {'P(A)':>6} | {'95% CI':>16} | {'N':>5}")
        print(f"  {'-'*45}")
        for r in dose_by_ord[ordering]:
            ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            print(f"  {r['multiplier']:>+7.3f} | {r['p_a']:>6.3f} | {ci:>16} | {r['n']:>5}")

    # 2. Aggregate (original 300)
    print("\n=== Aggregate steering effect (original 300) ===")
    agg = compute_aggregate_dose_response(dose_by_ord)
    print(f"  {'mult':>7} | {'AB shift':>8} | {'BA shift':>8} | {'avg':>6}")
    print(f"  {'-'*40}")
    for r in agg:
        print(f"  {r['multiplier']:>+7.3f} | {r['ab_shift']:>+8.4f} | {r['ba_shift']:>+8.4f} | {r['avg_shift']:>+6.4f}")

    # 3. Per-ordering baseline (original 300)
    print("\n=== Per-ordering baseline (original 300) ===")
    pair_baselines_all = compute_per_ordering_baseline(records)
    pair_baselines = {pid: bl for pid, bl in pair_baselines_all.items() if pid in old_pids}
    for ordering in [0, 1]:
        label = "AB" if ordering == 0 else "BA"
        pa_vals = [bl[ordering] for bl in pair_baselines.values() if ordering in bl]
        pa_arr = np.array(pa_vals)
        print(f"\n{label} ordering: n={len(pa_vals)}, mean P(A)={pa_arr.mean():.3f}, "
              f"P(A)=0: {(pa_arr==0).sum()}, P(A)=0.5: {(pa_arr==0.5).sum()}, P(A)=1: {(pa_arr==1).sum()}")

    # 4. Decided vs borderline (within original 300)
    print("\n=== Decided vs borderline (original 300) ===")
    decided_pids, borderline_pids = classify_pairs(pair_baselines)
    print(f"Decided (both orderings agree): {len(decided_pids)}")
    print(f"Borderline (orderings disagree or non-extreme): {len(borderline_pids)}")

    dose_decided = compute_per_ordering_dose_response(records, pair_subset=decided_pids)
    dose_borderline = compute_per_ordering_dose_response(records, pair_subset=borderline_pids)
    agg_decided = compute_aggregate_dose_response(dose_decided)
    agg_borderline = compute_aggregate_dose_response(dose_borderline)

    print("\nDecided pairs — aggregate steering:")
    for r in agg_decided:
        if abs(r["multiplier"]) <= COHERENCE_BOUNDARY:
            print(f"  mult={r['multiplier']:+.2f}: avg_shift={r['avg_shift']:+.4f}")
    print("\nBorderline pairs — aggregate steering:")
    for r in agg_borderline:
        if abs(r["multiplier"]) <= COHERENCE_BOUNDARY:
            print(f"  mult={r['multiplier']:+.2f}: avg_shift={r['avg_shift']:+.4f}")

    # 5. Per-ordering steerability (original 300)
    print("\n=== Per-ordering steerability vs decidedness (mult=+0.02) ===")
    steer_by_ord = compute_per_ordering_steerability(records, pair_baselines, mult=0.02)
    for ordering in [0, 1]:
        label = "AB" if ordering == 0 else "BA"
        data = steer_by_ord[ordering]
        decidedness = np.array([d["decidedness"] for d in data])
        abs_shift = np.array([d["abs_shift"] for d in data])
        r = np.corrcoef(decidedness, abs_shift)[0, 1]
        print(f"\n{label}: n={len(data)}, r(decidedness, |shift|)={r:.3f}, "
              f"mean |shift|={abs_shift.mean():.3f}")

    # 6. Plots
    print("\n=== Generating plots ===")
    plot_dose_response(dose_by_ord)
    plot_aggregate_dose_response(agg, dose_by_ord)
    plot_decided_vs_borderline(agg_decided, agg_borderline,
                               len(decided_pids), len(borderline_pids))
    plot_steerability_vs_decidedness(steer_by_ord)
    plot_baseline_per_ordering(pair_baselines)


if __name__ == "__main__":
    main()
