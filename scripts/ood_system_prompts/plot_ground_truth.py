"""Plot ground truth analysis results for OOD system prompts report.

Produces:
1. Overview bar chart: key metrics across all experiments
2. Per-experiment detail plots showing beh↔probe scatter with ground truth coloring

Usage: python -m scripts.ood_system_prompts.plot_ground_truth
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "ood_system_prompts"
ASSETS_DIR = RESULTS_DIR / "assets"

# Experiment display config
EXPERIMENTS = [
    ("1a: Known\ncategories", "exp1a"),
    ("1b: Novel\ntopics", "exp1b"),
    ("1c: Topic in\nwrong shell", "exp1c"),
    ("1d: Competing\nvalence", "exp1d"),
    ("2: Broad\nroles", "exp2"),
    ("3: Single\nsentence", "exp3"),
]

COLORS = {
    "beh_probe": "#2196F3",       # blue
    "beh_gt": "#4CAF50",          # green
    "probe_gt": "#FF9800",        # orange
}


def load_data() -> tuple[dict, dict]:
    gt_results = json.load(open(RESULTS_DIR / "ground_truth_results.json"))
    analysis_results = json.load(open(RESULTS_DIR / "analysis_results.json"))
    return gt_results, analysis_results


def plot_overview_bars(gt: dict, ar: dict) -> None:
    """Bar chart comparing key metrics across experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    exp_labels = [label for label, _ in EXPERIMENTS]

    # --- Panel 1: Pearson r (beh↔probe) ---
    ax = axes[0]
    r_all = []
    r_on = []
    for _, key in EXPERIMENTS:
        if key in gt:
            r_all.append(gt[key]["beh_probe_r_all"])
            r_on.append(gt[key]["beh_probe_r_on_target"])
        elif key == "exp2":
            r_all.append(ar["exp2"]["L31"]["pearson_r"])
            r_on.append(float("nan"))

    x = np.arange(len(EXPERIMENTS))
    width = 0.35
    bars1 = ax.bar(x - width / 2, r_all, width, label="All pairs", color="#90CAF9", edgecolor="#1565C0", linewidth=0.8)
    r_on_plot = [v if not np.isnan(v) else 0 for v in r_on]
    r_on_mask = [not np.isnan(v) for v in r_on]
    bars2 = ax.bar(
        x[r_on_mask] + width / 2,
        [v for v, m in zip(r_on_plot, r_on_mask) if m],
        width,
        label="Targeted pairs",
        color="#2196F3",
        edgecolor="#1565C0",
        linewidth=0.8,
    )
    ax.set_ylabel("Pearson r")
    ax.set_title("Behavioral ↔ Probe correlation", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.legend(fontsize=8, loc="upper right")

    # --- Panel 2: Sign agreement (beh↔probe) ---
    ax = axes[1]
    sign_all = []
    sign_on = []
    for _, key in EXPERIMENTS:
        if key in gt:
            sign_all.append(gt[key]["beh_probe_sign_all"])
            sign_on.append(gt[key]["beh_probe_sign_on_target"])
        elif key == "exp2":
            sign_all.append(ar["exp2"]["L31"]["sign_agreement"])
            sign_on.append(float("nan"))

    bars1 = ax.bar(x - width / 2, [v * 100 for v in sign_all], width, label="All pairs", color="#A5D6A7", edgecolor="#2E7D32", linewidth=0.8)
    sign_on_plot = [v * 100 if not np.isnan(v) else 0 for v in sign_on]
    sign_on_mask = [not np.isnan(v) for v in sign_on]
    bars2 = ax.bar(
        x[sign_on_mask] + width / 2,
        [v for v, m in zip(sign_on_plot, sign_on_mask) if m],
        width,
        label="Targeted pairs",
        color="#4CAF50",
        edgecolor="#2E7D32",
        linewidth=0.8,
    )
    ax.axhline(50, color="red", linewidth=1, linestyle="--", label="Chance (50%)")
    ax.set_ylabel("Sign agreement (%)")
    ax.set_title("Behavioral ↔ Probe sign agreement", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower right")

    # --- Panel 3: Ground truth sign agreement (on-target) ---
    ax = axes[2]
    beh_gt_sign = []
    probe_gt_sign = []
    gt_labels = []
    gt_x_indices = []
    for i, (label, key) in enumerate(EXPERIMENTS):
        if key in gt and not np.isnan(gt[key]["behavioral_sign_agreement"]):
            beh_gt_sign.append(gt[key]["behavioral_sign_agreement"] * 100)
            probe_gt_sign.append(gt[key]["probe_sign_agreement"] * 100)
            gt_labels.append(label)
            gt_x_indices.append(i)

    gx = np.arange(len(gt_labels))
    bars1 = ax.bar(gx - width / 2, beh_gt_sign, width, label="Behavior", color="#FFCC80", edgecolor="#E65100", linewidth=0.8)
    probe_gt_plot = [v if not np.isnan(v) else 0 for v in probe_gt_sign]
    probe_gt_mask = [not np.isnan(v) for v in probe_gt_sign]
    bars2 = ax.bar(
        gx[probe_gt_mask] + width / 2,
        [v for v, m in zip(probe_gt_plot, probe_gt_mask) if m],
        width,
        label="Probe",
        color="#FF9800",
        edgecolor="#E65100",
        linewidth=0.8,
    )
    ax.axhline(50, color="red", linewidth=1, linestyle="--", label="Chance (50%)")
    ax.set_ylabel("Sign agreement (%)")
    ax.set_title("Directional accuracy\n(targeted pairs)", fontsize=11, fontweight="bold")
    ax.set_xticks(gx)
    ax.set_xticklabels(gt_labels, fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out = ASSETS_DIR / "plot_022126_overview_metrics.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_per_experiment_results(gt: dict, ar: dict) -> None:
    """One figure per experiment showing beh↔probe scatter colored by ground truth."""
    exp_configs = [
        ("exp1a", "1a: Known categories"),
        ("exp1b", "1b: Novel topics"),
        ("exp1c", "1c: Topic in wrong shell"),
        ("exp1d", "1d: Competing valence"),
        ("exp3", "3: Single-sentence interest"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    for idx, (key, title) in enumerate(exp_configs):
        ax = axes[idx]
        beh, probe, labels, per_point_gt = _recompute_experiment(key)
        gt_data = gt[key]

        off_target = per_point_gt == 0
        gt_pos = per_point_gt > 0
        gt_neg = per_point_gt < 0

        ax.scatter(beh[off_target], probe[off_target], alpha=0.4, s=15,
                   color="#BDBDBD", edgecolors="none", label="Other", zorder=1)
        ax.scatter(beh[gt_pos], probe[gt_pos], alpha=0.7, s=22,
                   color="#4CAF50", edgecolors="none", label="Targeted (expected +)", zorder=2)
        ax.scatter(beh[gt_neg], probe[gt_neg], alpha=0.7, s=22,
                   color="#E53935", edgecolors="none", label="Targeted (expected −)", zorder=2)
        ax.legend(fontsize=7, loc="lower right")

        # Fit line
        mask = np.isfinite(beh) & np.isfinite(probe)
        if mask.sum() > 2:
            slope, intercept, r, p, se = scipy_stats.linregress(beh[mask], probe[mask])
            x_fit = np.linspace(beh[mask].min(), beh[mask].max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, color="red", linewidth=1.5, alpha=0.8)

        ax.axhline(0, color="grey", linewidth=0.5, linestyle="-")
        ax.axvline(0, color="grey", linewidth=0.5, linestyle="-")

        ax.set_xlabel("Behavioral delta", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Probe delta", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Stats text box
        stats_lines = [
            f"Beh↔Probe r: {gt_data['beh_probe_r_all']:.2f} (all), {gt_data['beh_probe_r_on_target']:.2f} (targeted)",
            f"Sign agree: {gt_data['beh_probe_sign_all']:.0%} (all), {gt_data['beh_probe_sign_on_target']:.0%} (targeted)",
            f"Directional: {gt_data['behavioral_sign_agreement']:.0%} beh, {gt_data['probe_sign_agreement']:.0%} probe (n={gt_data['n_on_target']})",
        ]

        stats_text = "\n".join(stats_lines)
        ax.text(
            0.05, 0.95, stats_text,
            transform=ax.transAxes, fontsize=7.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="grey"),
        )

    fig.tight_layout()
    out = ASSETS_DIR / "plot_022126_per_experiment_scatter.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_ground_truth_detail(gt: dict) -> None:
    """Grouped bar chart: Pearson r with ground truth for behavior vs probe, on-target."""
    exp_configs = [
        ("exp1a", "1a: Known\ncategories"),
        ("exp1b", "1b: Novel\ntopics"),
        ("exp1c", "1c: Topic in\nwrong shell"),
        ("exp1d", "1d: Competing\nvalence"),
        ("exp3", "3: Single\nsentence"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(exp_configs))
    width = 0.25

    beh_gt_r = [gt[k]["beh_gt_r_on_target"] for _, (k, _) in enumerate(exp_configs)]
    probe_gt_r = [gt[k]["probe_gt_r_on_target"] for _, (k, _) in enumerate(exp_configs)]
    beh_probe_r = [gt[k]["beh_probe_r_on_target"] for _, (k, _) in enumerate(exp_configs)]

    labels_x = [label for _, label in exp_configs]

    ax.bar(x - width, beh_gt_r, width, label="Behavior ↔ Ground truth", color="#4CAF50", edgecolor="#2E7D32", linewidth=0.8)
    ax.bar(x, probe_gt_r, width, label="Probe ↔ Ground truth", color="#FF9800", edgecolor="#E65100", linewidth=0.8)
    ax.bar(x + width, beh_probe_r, width, label="Behavior ↔ Probe", color="#2196F3", edgecolor="#1565C0", linewidth=0.8)

    ax.set_ylabel("Pearson r (on-target pairs)")
    ax.set_title("On-target correlations: behavior, probe, and ground truth", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower left")

    # Add value labels on bars
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    fig.tight_layout()
    out = ASSETS_DIR / "plot_022126_ground_truth_correlation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _recompute_experiment(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recompute beh, probe, condition_labels, and per-point ground truth in one pass.

    Returns aligned arrays — no ordering mismatch possible.
    """
    from scripts.ood_system_prompts.analyze_ground_truth import (
        RESULTS_OOD, ACTS_DIR, CONFIGS, probe_path,
        _recompute_with_task_ids,
        _ground_truth_exp1a,
        _ground_truth_exp1b,
        _ground_truth_exp1c,
        _ground_truth_exp1d,
        _ground_truth_exp3,
    )
    from src.ood.analysis import compute_p_choose_from_pairwise

    gt_fns = {
        "exp1a": _ground_truth_exp1a,
        "exp1b": _ground_truth_exp1b,
        "exp1c": _ground_truth_exp1c,
        "exp1d": _ground_truth_exp1d,
        "exp3": _ground_truth_exp3,
    }

    if key == "exp1a":
        pairwise = json.load(open(RESULTS_OOD / "category_preference" / "pairwise.json"))
        rates = compute_p_choose_from_pairwise(pairwise["results"])
        acts_dir = ACTS_DIR / "exp1_category"
    elif key == "exp1b":
        pairwise = json.load(open(RESULTS_OOD / "hidden_preference" / "pairwise.json"))
        rates = compute_p_choose_from_pairwise(pairwise["results"])
        rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}
        rates = {k: {tid: v for tid, v in rd.items() if tid.startswith("hidden_")} for k, rd in rates.items()}
        acts_dir = ACTS_DIR / "exp1_prompts"
    elif key == "exp1c":
        pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
        rates = compute_p_choose_from_pairwise(pairwise["results"])
        rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}
        rates = {k: {tid: v for tid, v in rd.items() if tid.startswith("crossed_")} for k, rd in rates.items()}
        acts_dir = ACTS_DIR / "exp1_prompts"
    elif key == "exp1d":
        pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
        rates = compute_p_choose_from_pairwise(pairwise["results"])
        rates = {k: v for k, v in rates.items() if k.startswith("compete_") or k == "baseline"}
        rates = {k: {tid: v for tid, v in rd.items() if tid.startswith("crossed_")} for k, rd in rates.items()}
        acts_dir = ACTS_DIR / "exp1_prompts"
    elif key == "exp3":
        mp_cfg = json.load(open(CONFIGS / "prompts" / "minimal_pairs_v7.json"))
        selected_roles = {"midwest", "brooklyn"}
        selected_versions = {"A", "B", "C"}
        selected_cids = {
            c["condition_id"] for c in mp_cfg["conditions"]
            if c["base_role"] in selected_roles and c["version"] in selected_versions
        }
        selected_cids.add("baseline")
        beh_data = json.load(open(RESULTS_OOD / "minimal_pairs_v7" / "behavioral.json"))
        rates = {}
        for cid, cond_data in beh_data["conditions"].items():
            if cid not in selected_cids:
                continue
            rates[cid] = {tid: v["p_choose"] for tid, v in cond_data["task_rates"].items()}
        acts_dir = ACTS_DIR / "exp3_minimal_pairs"
    else:
        raise ValueError(f"Unknown experiment key: {key}")

    beh, probe, labels, task_ids = _recompute_with_task_ids(rates, acts_dir, 31)
    per_point_gt = gt_fns[key](labels, task_ids)
    return beh, probe, labels, per_point_gt


def _sign_agreement_filtered(a: np.ndarray, b: np.ndarray, beh: np.ndarray, probe: np.ndarray) -> float:
    """Sign agreement between a and b, excluding pairs where |beh| or |probe| is near zero."""
    beh_thresh = 0.05
    probe_thresh = np.median(np.abs(probe)) / 10
    keep = (np.abs(beh) >= beh_thresh) & (np.abs(probe) >= probe_thresh)
    if keep.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(a[keep]) == np.sign(b[keep])) * 100)


def plot_individual_experiments(gt: dict, ar: dict) -> None:
    """Per-experiment figure: scatter + beh↔probe bars + sign agreement vs expected."""
    full_results = json.load(open(RESULTS_DIR / "analysis_results_full.json"))

    exp_configs = [
        ("exp1a", "1a: Known categories"),
        ("exp1b", "1b: Novel topics"),
        ("exp1c", "1c: Topic in wrong shell"),
        ("exp1d", "1d: Competing valence"),
        ("exp2", "2: Broad roles"),
        ("exp3", "3: Single-sentence interest"),
    ]

    for key, title in exp_configs:
        has_gt = key in gt

        if has_gt:
            fig, (ax_scatter, ax_bp, ax_expected) = plt.subplots(
                1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1.4, 1, 1]}
            )
        else:
            fig, ax_scatter = plt.subplots(figsize=(7, 5))

        # --- Scatter plot ---
        if has_gt:
            beh, probe, labels, per_point_gt = _recompute_experiment(key)
            off_target = per_point_gt == 0
            gt_pos = per_point_gt > 0
            gt_neg = per_point_gt < 0

            ax_scatter.scatter(beh[off_target], probe[off_target], alpha=0.4, s=15,
                               color="#BDBDBD", edgecolors="none", label="Other", zorder=1)
            ax_scatter.scatter(beh[gt_pos], probe[gt_pos], alpha=0.7, s=22,
                               color="#4CAF50", edgecolors="none", label="Targeted (expected +)", zorder=2)
            ax_scatter.scatter(beh[gt_neg], probe[gt_neg], alpha=0.7, s=22,
                               color="#E53935", edgecolors="none", label="Targeted (expected −)", zorder=2)
            ax_scatter.legend(fontsize=8, loc="lower right")
        else:
            res = full_results[key]["L31"]
            beh = np.array(res["behavioral_deltas"])
            probe = np.array(res["probe_deltas"])
            ax_scatter.scatter(beh, probe, alpha=0.5, s=18, color="#2196F3", edgecolors="none")

        fin = np.isfinite(beh) & np.isfinite(probe)
        if fin.sum() > 2:
            slope, intercept, r, p, se = scipy_stats.linregress(beh[fin], probe[fin])
            x_fit = np.linspace(beh[fin].min(), beh[fin].max(), 100)
            ax_scatter.plot(x_fit, slope * x_fit + intercept, color="red", linewidth=1.5, alpha=0.8)

        ax_scatter.axhline(0, color="grey", linewidth=0.5)
        ax_scatter.axvline(0, color="grey", linewidth=0.5)
        ax_scatter.set_xlabel("Behavioral delta", fontsize=10)
        ax_scatter.set_ylabel("Probe delta", fontsize=10)
        ax_scatter.set_title(f"{title}", fontsize=11, fontweight="bold")

        if has_gt:
            # --- Panel 2: Beh ↔ Probe (Pearson r + sign agreement, all & targeted) ---
            targeted = per_point_gt != 0
            beh_t = beh[targeted]
            probe_t = probe[targeted]

            r_all = scipy_stats.pearsonr(beh, probe)[0]
            r_targeted = scipy_stats.pearsonr(beh_t, probe_t)[0]
            sign_all = _sign_agreement_filtered(beh, probe, beh, probe)
            sign_targeted = _sign_agreement_filtered(beh_t, probe_t, beh_t, probe_t)

            _plot_beh_probe_bars(ax_bp, r_all, r_targeted, sign_all, sign_targeted,
                                int(len(beh)), int(targeted.sum()))

            # --- Panel 3: Sign agreement vs expected (targeted only) ---
            gt_t = per_point_gt[targeted]
            _plot_expected_bars(ax_expected, beh_t, probe_t, gt_t)
        else:
            pass  # no extra panels for exp2

        fig.tight_layout()
        out = ASSETS_DIR / f"plot_022126_{key}_detail.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


def _plot_beh_probe_bars(
    ax: plt.Axes, r_all: float, r_targeted: float,
    sign_all: float, sign_targeted: float,
    n_all: int, n_targeted: int,
) -> None:
    """Panel 2: Pearson r and sign agreement between beh and probe, all vs targeted."""
    x = np.arange(2)
    width = 0.35

    bars_all = ax.bar(x - width / 2, [r_all, sign_all / 100], width,
                      label=f"All (n={n_all})", color="#90CAF9", edgecolor="#1565C0", linewidth=0.8)
    bars_tgt = ax.bar(x + width / 2, [r_targeted, sign_targeted / 100], width,
                      label=f"Targeted (n={n_targeted})", color="#2196F3", edgecolor="#1565C0", linewidth=0.8)

    for bars in [bars_all, bars_tgt]:
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ax.set_ylabel("Value")
    ax.set_title("Beh ↔ Probe", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Pearson r", "Sign\nagreement"], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def _plot_expected_bars(ax: plt.Axes, beh: np.ndarray, probe: np.ndarray, gt: np.ndarray) -> None:
    """Panel 3: Sign agreement of beh and probe vs expected direction (targeted pairs only)."""
    n = len(gt)
    beh_vs_expected = float(np.mean(np.sign(beh) == np.sign(gt)) * 100)
    probe_vs_expected = float(np.mean(np.sign(probe) == np.sign(gt)) * 100)

    labels = ["Beh vs\nexpected", "Probe vs\nexpected"]
    vals = [beh_vs_expected, probe_vs_expected]
    colors = ["#4CAF50", "#FF9800"]
    edge_colors = ["#2E7D32", "#E65100"]

    x = np.arange(len(labels))
    bars = ax.bar(x, vals, 0.45, color=colors, edgecolor=edge_colors, linewidth=0.8)
    ax.bar_label(bars, fmt="%.0f%%", fontsize=9, padding=2)

    ax.axhline(50, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="Chance")
    ax.set_ylabel("Sign agreement (%)")
    ax.set_title(f"Sign agreement vs expected\n(targeted pairs, n={n})", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=8, loc="upper right")


def main() -> None:
    gt, ar = load_data()
    plot_overview_bars(gt, ar)
    plot_per_experiment_results(gt, ar)
    plot_ground_truth_detail(gt)
    plot_individual_experiments(gt, ar)


if __name__ == "__main__":
    main()
