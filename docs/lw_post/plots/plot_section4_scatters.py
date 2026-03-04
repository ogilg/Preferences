"""Section 4 scatter plots: probe delta vs behavioral delta for OOD experiments.

Each plot shows:
- Grey dots for off-target tasks, coloured dots for targeted tasks
- Two trend lines: all tasks (grey) and targeted only (coloured)
- Pearson r annotated for both

Usage:
    cd docs/lw_post && python plot_section4_scatters.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "experiments" / "ood_system_prompts"
ASSETS_DIR = Path(__file__).parent / "assets"

GREEN = "#4CAF50"
RED = "#E53935"
GREY = "#BDBDBD"


def load_data():
    gt = json.load(open(RESULTS_DIR / "ground_truth_results.json"))
    ar = json.load(open(RESULTS_DIR / "analysis_results.json"))
    return gt, ar


def recompute_experiment(key: str):
    from scripts.ood_system_prompts.plot_ground_truth import _recompute_experiment
    return _recompute_experiment(key)


def plot_scatter(key: str, title: str, filename: str, gt_data: dict) -> None:
    beh, probe, labels, per_point_gt = recompute_experiment(key)

    fig, ax = plt.subplots(figsize=(6, 5))

    off_target = per_point_gt == 0
    gt_pos = per_point_gt > 0
    gt_neg = per_point_gt < 0
    targeted = per_point_gt != 0

    # Scatter points
    ax.scatter(beh[off_target], probe[off_target], alpha=0.3, s=12,
               color=GREY, edgecolors="none", label="Off-target", zorder=1)
    ax.scatter(beh[gt_pos], probe[gt_pos], alpha=0.7, s=20,
               color=GREEN, edgecolors="none", label="Targeted (+)", zorder=2)
    ax.scatter(beh[gt_neg], probe[gt_neg], alpha=0.7, s=20,
               color=RED, edgecolors="none", label="Targeted (−)", zorder=2)

    # Trend line: all tasks
    fin = np.isfinite(beh) & np.isfinite(probe)
    slope, intercept, r_all, _, _ = scipy_stats.linregress(beh[fin], probe[fin])
    x_fit = np.linspace(beh[fin].min(), beh[fin].max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept,
            color="#888", linewidth=1.5, linestyle="--", alpha=0.7,
            label=f"All (r = {r_all:.2f})")

    # Trend line: targeted only
    beh_t, probe_t = beh[targeted], probe[targeted]
    fin_t = np.isfinite(beh_t) & np.isfinite(probe_t)
    slope_t, intercept_t, r_tgt, _, _ = scipy_stats.linregress(beh_t[fin_t], probe_t[fin_t])
    x_fit_t = np.linspace(beh_t[fin_t].min(), beh_t[fin_t].max(), 100)
    ax.plot(x_fit_t, slope_t * x_fit_t + intercept_t,
            color=RED, linewidth=2, alpha=0.8,
            label=f"Targeted (r = {r_tgt:.2f})")

    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel(
        "Behavioral shift  ($\\Delta \\mathbf{P}$)\n"
        "$\\Delta \\mathbf{P} = \\mathbf{P}(\\mathrm{choose\\;task} \\mid \\mathrm{sysprompt})"
        " - \\mathbf{P}(\\mathrm{choose\\;task} \\mid \\mathrm{baseline})$",
        fontsize=9.5, linespacing=1.6,
    )
    ax.set_ylabel(
        "Probe shift  ($\\Delta\\mathbf{probe}$)\n"
        "$\\Delta\\mathbf{probe} = \\mathbf{probe}(\\mathrm{sysprompt} + \\mathrm{task})"
        " - \\mathbf{probe}(\\mathrm{task})$",
        fontsize=9.5, linespacing=1.6,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    out = ASSETS_DIR / filename
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _scatter_panel(ax: plt.Axes, key: str, title: str) -> None:
    beh, probe, labels, per_point_gt = recompute_experiment(key)

    off_target = per_point_gt == 0
    gt_pos = per_point_gt > 0
    gt_neg = per_point_gt < 0
    targeted = per_point_gt != 0

    ax.scatter(beh[off_target], probe[off_target], alpha=0.3, s=12,
               color=GREY, edgecolors="none", label="Off-target", zorder=1)
    ax.scatter(beh[gt_pos], probe[gt_pos], alpha=0.7, s=20,
               color=GREEN, edgecolors="none", label="Targeted (+)", zorder=2)
    ax.scatter(beh[gt_neg], probe[gt_neg], alpha=0.7, s=20,
               color=RED, edgecolors="none", label="Targeted (−)", zorder=2)

    fin = np.isfinite(beh) & np.isfinite(probe)
    slope, intercept, r_all, _, _ = scipy_stats.linregress(beh[fin], probe[fin])
    x_fit = np.linspace(beh[fin].min(), beh[fin].max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept,
            color="#888", linewidth=1.5, linestyle="--", alpha=0.7,
            label=f"All (r = {r_all:.2f})")

    beh_t, probe_t = beh[targeted], probe[targeted]
    fin_t = np.isfinite(beh_t) & np.isfinite(probe_t)
    slope_t, intercept_t, r_tgt, _, _ = scipy_stats.linregress(beh_t[fin_t], probe_t[fin_t])
    x_fit_t = np.linspace(beh_t[fin_t].min(), beh_t[fin_t].max(), 100)
    ax.plot(x_fit_t, slope_t * x_fit_t + intercept_t,
            color=RED, linewidth=2, alpha=0.8,
            label=f"Targeted (r = {r_tgt:.2f})")

    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")


def plot_scatter_sidebyside(filename: str) -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))

    _scatter_panel(ax_l, "exp1c", "One-sided conflict")
    _scatter_panel(ax_r, "exp1d", "Opposing prompts")

    ax_l.set_xlabel(
        "Behavioral shift  ($\\Delta \\mathbf{P}$)\n"
        "$\\Delta \\mathbf{P} = \\mathbf{P}(\\mathrm{choose\\;task} \\mid \\mathrm{sysprompt})"
        " - \\mathbf{P}(\\mathrm{choose\\;task} \\mid \\mathrm{baseline})$",
        fontsize=9.5, linespacing=1.6,
    )
    ax_r.set_xlabel(
        "Behavioral shift  ($\\Delta \\mathbf{P}$)\n"
        "$\\Delta \\mathbf{P} = \\mathbf{P}(\\mathrm{choose\\;task} \\mid \\mathrm{sysprompt})"
        " - \\mathbf{P}(\\mathrm{choose\\;task} \\mid \\mathrm{baseline})$",
        fontsize=9.5, linespacing=1.6,
    )
    ax_l.set_ylabel(
        "Probe shift  ($\\Delta\\mathbf{probe}$)\n"
        "$\\Delta\\mathbf{probe} = \\mathbf{probe}(\\mathrm{sysprompt} + \\mathrm{task})"
        " - \\mathbf{probe}(\\mathrm{task})$",
        fontsize=9.5, linespacing=1.6,
    )

    fig.tight_layout()
    out = ASSETS_DIR / filename
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_exp3_version_pairs(filename: str) -> None:
    """3-panel scatter: A vs B, B vs C, A vs C for fine-grained single-sentence experiment."""
    from scripts.ood_system_prompts.analyze_exp3_versions import (
        ACTS_DIR as EXP3_ACTS_DIR,
        PROBE_DIR as EXP3_PROBE_DIR,
        BEH_PATH as EXP3_BEH_PATH,
        CFG_PATH as EXP3_CFG_PATH,
        SELECTED_ROLES,
        VERSION_PAIRS,
        EXP3_TASK_TARGETS,
        EXCLUDED_TASKS,
        load_probe,
        score_activations,
        compute_condition_deltas,
        LAYER,
    )

    beh_data = json.load(open(EXP3_BEH_PATH))
    cfg = json.load(open(EXP3_CFG_PATH))
    cond_info = {c["condition_id"]: c for c in cfg["conditions"]}

    weights, bias = load_probe(LAYER)
    baseline_scores = score_activations(
        EXP3_ACTS_DIR / "baseline" / "activations_prompt_last.npz", LAYER, weights, bias
    )
    baseline_rates = {
        tid: v["p_choose"] for tid, v in beh_data["conditions"]["baseline"]["task_rates"].items()
    }
    tasks = sorted(k for k in baseline_rates.keys() if k not in EXCLUDED_TASKS)

    # Compute per-condition deltas
    condition_deltas: dict[str, tuple[dict, dict]] = {}
    for cid in cond_info:
        if cond_info[cid]["base_role"] not in SELECTED_ROLES:
            continue
        beh_d, probe_d = compute_condition_deltas(
            cid, baseline_rates, baseline_scores, beh_data, weights, bias, tasks
        )
        if beh_d:
            condition_deltas[cid] = (beh_d, probe_d)

    pair_labels = {
        "A_vs_B": "A vs B\n(pro vs neutral)",
        "B_vs_C": "B vs C\n(neutral vs anti)",
        "A_vs_C": "A vs C\n(pro vs anti)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for idx, (v_x, v_y) in enumerate(VERSION_PAIRS):
        ax = axes[idx]
        pair_key = f"{v_x}_vs_{v_y}"

        # Group conditions by (base_role, target)
        groups: dict[tuple[str, str], dict[str, str]] = {}
        for cid, info in cond_info.items():
            if info["base_role"] not in SELECTED_ROLES:
                continue
            key = (info["base_role"], info["target"])
            if key not in groups:
                groups[key] = {}
            groups[key][info["version"]] = cid

        all_beh, all_probe, all_is_target, all_probe_rank1 = [], [], [], []
        n_target_total, n_probe_rank1, n_beh_rank1 = 0, 0, 0
        probe_ranks, beh_ranks = [], []

        for (base_role, target), version_cids in sorted(groups.items()):
            if v_x not in version_cids or v_y not in version_cids:
                continue
            cid_x, cid_y = version_cids[v_x], version_cids[v_y]
            if cid_x not in condition_deltas or cid_y not in condition_deltas:
                continue
            beh_x, probe_x = condition_deltas[cid_x]
            beh_y, probe_y = condition_deltas[cid_y]
            common = sorted(set(beh_x) & set(beh_y))

            beh_vp = {tid: beh_x[tid] - beh_y[tid] for tid in common}
            probe_vp = {tid: probe_x[tid] - probe_y[tid] for tid in common}

            # Compute ranks (descending)
            beh_sorted = sorted(common, key=lambda t: beh_vp[t], reverse=True)
            probe_sorted = sorted(common, key=lambda t: probe_vp[t], reverse=True)

            target_tids = {tid for tid in common if target in EXP3_TASK_TARGETS.get(tid, set())}

            for tid in common:
                is_target = tid in target_tids
                all_beh.append(beh_vp[tid])
                all_probe.append(probe_vp[tid])
                all_is_target.append(is_target)
                if is_target:
                    p_rank = probe_sorted.index(tid) + 1
                    b_rank = beh_sorted.index(tid) + 1
                    all_probe_rank1.append(p_rank == 1)
                    probe_ranks.append(p_rank)
                    beh_ranks.append(b_rank)
                    n_target_total += 1
                    if p_rank == 1:
                        n_probe_rank1 += 1
                    if b_rank == 1:
                        n_beh_rank1 += 1
                else:
                    all_probe_rank1.append(False)

        beh_arr = np.array(all_beh)
        probe_arr = np.array(all_probe)
        target_mask = np.array(all_is_target)
        rank1_mask = np.array(all_probe_rank1)
        target_not_rank1 = target_mask & ~rank1_mask

        # Scatter: off-target
        ax.scatter(beh_arr[~target_mask], probe_arr[~target_mask],
                   alpha=0.3, s=12, color=GREY, edgecolors="none", zorder=1,
                   label="Other tasks")
        # Target rank 1: filled star
        if rank1_mask.any():
            ax.scatter(beh_arr[rank1_mask], probe_arr[rank1_mask],
                       s=70, color="#e41a1c", marker="*", edgecolors="black",
                       linewidths=0.3, zorder=3,
                       label=f"Target (probe rank 1/{n_target_total})")
        # Target not rank 1: open star
        if target_not_rank1.any():
            ax.scatter(beh_arr[target_not_rank1], probe_arr[target_not_rank1],
                       s=70, facecolors="none", marker="*", edgecolors="#e41a1c",
                       linewidths=1.0, zorder=3,
                       label=f"Target (not rank 1/{n_target_total})")

        # Trend line (all) — dashed, de-emphasized
        r_all = scipy_stats.pearsonr(beh_arr, probe_arr)[0]
        slope, intercept, _, _, _ = scipy_stats.linregress(beh_arr, probe_arr)
        x_fit = np.linspace(beh_arr.min(), beh_arr.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept,
                color="#2196F3", linewidth=1.2, linestyle="--", alpha=0.5,
                label=f"All (r = {r_all:.2f})")

        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5)

        # Axis labels
        version_label = f"{v_x} − {v_y}"
        ax.set_xlabel(
            f"Behavioral shift  ($\\Delta \\mathbf{{P}}$, {version_label})\n"
            f"$\\Delta \\mathbf{{P}} = \\mathbf{{P}}_{{\\mathrm{{{v_x}}}}} - \\mathbf{{P}}_{{\\mathrm{{{v_y}}}}}$",
            fontsize=9.5, linespacing=1.6,
        )
        if idx == 0:
            ax.set_ylabel(
                f"Probe shift  ($\\Delta\\mathbf{{probe}}$, {version_label})\n"
                f"$\\Delta\\mathbf{{probe}} = \\mathbf{{probe}}_{{\\mathrm{{{v_x}}}}} - \\mathbf{{probe}}_{{\\mathrm{{{v_y}}}}}$",
                fontsize=9.5, linespacing=1.6,
            )

        ax.set_title(pair_labels[pair_key], fontsize=12, fontweight="bold")

        # Stats
        stats_text = (
            f"r = {r_all:.2f} (n={len(beh_arr)})\n"
            f"Probe rank 1: {n_probe_rank1}/{n_target_total}"
        )
        ax.text(
            0.05, 0.95, stats_text,
            transform=ax.transAxes, fontsize=8.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="grey"),
        )

        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out = ASSETS_DIR / filename
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_exp3_avc(filename: str) -> None:
    """Single-panel A vs C scatter for fine-grained experiment."""
    from scripts.ood_system_prompts.analyze_exp3_versions import (
        ACTS_DIR as EXP3_ACTS_DIR,
        BEH_PATH as EXP3_BEH_PATH,
        CFG_PATH as EXP3_CFG_PATH,
        SELECTED_ROLES,
        EXP3_TASK_TARGETS,
        EXCLUDED_TASKS,
        load_probe,
        score_activations,
        compute_condition_deltas,
        LAYER,
    )

    beh_data = json.load(open(EXP3_BEH_PATH))
    cfg = json.load(open(EXP3_CFG_PATH))
    cond_info = {c["condition_id"]: c for c in cfg["conditions"]}

    weights, bias = load_probe(LAYER)
    baseline_scores = score_activations(
        EXP3_ACTS_DIR / "baseline" / "activations_prompt_last.npz", LAYER, weights, bias
    )
    baseline_rates = {
        tid: v["p_choose"] for tid, v in beh_data["conditions"]["baseline"]["task_rates"].items()
    }
    tasks = sorted(k for k in baseline_rates.keys() if k not in EXCLUDED_TASKS)

    condition_deltas: dict[str, tuple[dict, dict]] = {}
    for cid in cond_info:
        if cond_info[cid]["base_role"] not in SELECTED_ROLES:
            continue
        beh_d, probe_d = compute_condition_deltas(
            cid, baseline_rates, baseline_scores, beh_data, weights, bias, tasks
        )
        if beh_d:
            condition_deltas[cid] = (beh_d, probe_d)

    # Group by (base_role, target)
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for cid, info in cond_info.items():
        if info["base_role"] not in SELECTED_ROLES:
            continue
        key = (info["base_role"], info["target"])
        if key not in groups:
            groups[key] = {}
        groups[key][info["version"]] = cid

    all_beh, all_probe, all_is_target, all_probe_rank1 = [], [], [], []
    n_target_total, n_probe_rank1 = 0, 0

    for (base_role, target), version_cids in sorted(groups.items()):
        if "A" not in version_cids or "C" not in version_cids:
            continue
        cid_a, cid_c = version_cids["A"], version_cids["C"]
        if cid_a not in condition_deltas or cid_c not in condition_deltas:
            continue
        beh_a, probe_a = condition_deltas[cid_a]
        beh_c, probe_c = condition_deltas[cid_c]
        common = sorted(set(beh_a) & set(beh_c))

        beh_vp = {tid: beh_a[tid] - beh_c[tid] for tid in common}
        probe_vp = {tid: probe_a[tid] - probe_c[tid] for tid in common}
        probe_sorted = sorted(common, key=lambda t: probe_vp[t], reverse=True)
        target_tids = {tid for tid in common if target in EXP3_TASK_TARGETS.get(tid, set())}

        for tid in common:
            is_target = tid in target_tids
            all_beh.append(beh_vp[tid])
            all_probe.append(probe_vp[tid])
            all_is_target.append(is_target)
            if is_target:
                p_rank = probe_sorted.index(tid) + 1
                all_probe_rank1.append(p_rank == 1)
                n_target_total += 1
                if p_rank == 1:
                    n_probe_rank1 += 1
            else:
                all_probe_rank1.append(False)

    beh_arr = np.array(all_beh)
    probe_arr = np.array(all_probe)
    target_mask = np.array(all_is_target)
    rank1_mask = np.array(all_probe_rank1)
    target_not_rank1 = target_mask & ~rank1_mask

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(beh_arr[~target_mask], probe_arr[~target_mask],
               alpha=0.3, s=12, color=GREY, edgecolors="none", zorder=1,
               label="Other tasks")
    if rank1_mask.any():
        ax.scatter(beh_arr[rank1_mask], probe_arr[rank1_mask],
                   s=70, color="#e41a1c", marker="*", edgecolors="black",
                   linewidths=0.3, zorder=3,
                   label=f"Target (probe rank 1/{n_target_total})")
    if target_not_rank1.any():
        ax.scatter(beh_arr[target_not_rank1], probe_arr[target_not_rank1],
                   s=70, facecolors="none", marker="*", edgecolors="#e41a1c",
                   linewidths=1.0, zorder=3,
                   label=f"Target (not rank 1/{n_target_total})")

    # Trend line — dashed
    r_all = scipy_stats.pearsonr(beh_arr, probe_arr)[0]
    slope, intercept, _, _, _ = scipy_stats.linregress(beh_arr, probe_arr)
    x_fit = np.linspace(beh_arr.min(), beh_arr.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept,
            color="#2196F3", linewidth=1.2, linestyle="--", alpha=0.5,
            label=f"All (r = {r_all:.2f})")

    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)

    ax.set_xlabel(
        "Behavioral shift  ($\\Delta \\mathbf{P}$, A − C)\n"
        "$\\Delta \\mathbf{P} = \\mathbf{P}_{\\mathrm{A}} - \\mathbf{P}_{\\mathrm{C}}$",
        fontsize=9.5, linespacing=1.6,
    )
    ax.set_ylabel(
        "Probe shift  ($\\Delta\\mathbf{probe}$, A − C)\n"
        "$\\Delta\\mathbf{probe} = \\mathbf{probe}_{\\mathrm{A}} - \\mathbf{probe}_{\\mathrm{C}}$",
        fontsize=9.5, linespacing=1.6,
    )
    ax.set_title("A vs C (pro-interest vs anti-interest)", fontsize=13, fontweight="bold")

    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    out = ASSETS_DIR / filename
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    gt, ar = load_data()

    plot_scatter(
        "exp1b",
        "Simple preference",
        "plot_022626_s4_scatter_simple.png",
        gt["exp1b"],
    )
    plot_scatter(
        "exp1c",
        "Content-preference conflict",
        "plot_022626_s4_scatter_conflict.png",
        gt["exp1c"],
    )
    plot_scatter(
        "exp1d",
        "Opposing prompts",
        "plot_022626_s4_scatter_competing.png",
        gt["exp1d"],
    )

    plot_scatter_sidebyside("plot_030226_s4_scatter_conflict_opposing.png")

    plot_exp3_version_pairs("plot_022626_s4_scatter_fine_grained.png")
    plot_exp3_avc("plot_022626_s4_scatter_fine_grained_avc.png")


if __name__ == "__main__":
    main()
