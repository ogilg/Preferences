"""Exp 3 version-pair analysis: compare A vs B, B vs C, A vs C deltas.

For each version pair (X, Y) and each (base_role, target):
- Compute per-task version-pair delta = (condition_X delta) - (condition_Y delta)
  where delta = condition - baseline
- Focus on target task: does the version-pair delta go the right direction?
- Compute specificity: |target delta| / mean(|off-target deltas|)

Also produces a 3-panel scatter plot (beh vs probe version-pair deltas).

Usage: python -m scripts.ood_system_prompts.analyze_exp3_versions
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent.parent
ACTS_DIR = REPO_ROOT / "activations" / "ood" / "exp3_minimal_pairs"
PROBE_DIR = REPO_ROOT / "results" / "probes" / "gemma3_10k_heldout_std_demean" / "probes"
BEH_PATH = REPO_ROOT / "results" / "ood" / "minimal_pairs_v7" / "behavioral.json"
CFG_PATH = REPO_ROOT / "configs" / "ood" / "prompts" / "minimal_pairs_v7.json"
ASSETS_DIR = REPO_ROOT / "experiments" / "ood_system_prompts" / "assets"

LAYER = 31
SELECTED_ROLES = {"midwest", "brooklyn"}
VERSION_PAIRS = [("A", "B"), ("B", "C"), ("A", "C")]

# Task-target mapping (from analyze_ground_truth.py)
EXP3_TASK_TARGETS: dict[str, set[str]] = {
    "alpaca_14631": {"shakespeare"},
    "stresstest_73_1202_value1": {"lotr"},
    "stresstest_54_530_neutral": {"chess"},
    "alpaca_13003": {"convexhull"},
    "alpaca_3808": {"detective"},
    "alpaca_13255": {"haiku"},
    "stresstest_89_193_value1": {"haiku"},
    "alpaca_5529": {"pyramids"},
    "wildchat_35599": {"simpsons"},
    "stresstest_43_948_value2": {"wwii"},
}


def load_probe(layer: int) -> tuple[np.ndarray, float]:
    probe = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return probe[:-1], float(probe[-1])


def score_activations(npz_path: Path, layer: int, weights: np.ndarray, bias: float) -> dict[str, float]:
    data = np.load(npz_path, allow_pickle=True)
    acts = data[f"layer_{layer}"]
    scores = acts @ weights + bias
    task_ids = list(data["task_ids"])
    return dict(zip(task_ids, scores.tolist()))


def compute_condition_deltas(
    cid: str,
    baseline_rates: dict[str, float],
    baseline_scores: dict[str, float],
    beh_data: dict,
    weights: np.ndarray,
    bias: float,
    tasks: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-task behavioral and probe deltas for a condition vs baseline."""
    npz = ACTS_DIR / cid / "activations_prompt_last.npz"
    if not npz.exists():
        return {}, {}
    cond_scores = score_activations(npz, LAYER, weights, bias)

    if cid not in beh_data["conditions"]:
        return {}, {}
    cond_rates = {
        tid: v["p_choose"] for tid, v in beh_data["conditions"][cid]["task_rates"].items()
    }

    beh_deltas = {}
    probe_deltas = {}
    for tid in tasks:
        if tid in cond_rates and tid in baseline_rates and tid in cond_scores and tid in baseline_scores:
            beh_deltas[tid] = cond_rates[tid] - baseline_rates[tid]
            probe_deltas[tid] = cond_scores[tid] - baseline_scores[tid]
    return beh_deltas, probe_deltas


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    beh_data = json.load(open(BEH_PATH))
    cfg = json.load(open(CFG_PATH))
    cond_info = {c["condition_id"]: c for c in cfg["conditions"]}

    weights, bias = load_probe(LAYER)

    # Baseline
    baseline_scores = score_activations(
        ACTS_DIR / "baseline" / "activations_prompt_last.npz", LAYER, weights, bias
    )
    baseline_rates = {
        tid: v["p_choose"] for tid, v in beh_data["conditions"]["baseline"]["task_rates"].items()
    }
    tasks = sorted(baseline_rates.keys())

    # Collect per-condition deltas for all selected conditions
    condition_deltas: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for cid in cond_info:
        info = cond_info[cid]
        if info["base_role"] not in SELECTED_ROLES:
            continue
        beh_d, probe_d = compute_condition_deltas(
            cid, baseline_rates, baseline_scores, beh_data, weights, bias, tasks
        )
        if beh_d:
            condition_deltas[cid] = (beh_d, probe_d)

    # For each version pair, compute version-pair deltas
    pair_results: dict[str, list[dict]] = {}

    for v_x, v_y in VERSION_PAIRS:
        pair_key = f"{v_x}_vs_{v_y}"
        pair_results[pair_key] = []

        # Group conditions by (base_role, target)
        groups: dict[tuple[str, str], dict[str, str]] = {}
        for cid, info in cond_info.items():
            if info["base_role"] not in SELECTED_ROLES:
                continue
            key = (info["base_role"], info["target"])
            if key not in groups:
                groups[key] = {}
            groups[key][info["version"]] = cid

        for (base_role, target), version_cids in sorted(groups.items()):
            if v_x not in version_cids or v_y not in version_cids:
                continue
            cid_x = version_cids[v_x]
            cid_y = version_cids[v_y]
            if cid_x not in condition_deltas or cid_y not in condition_deltas:
                continue

            beh_x, probe_x = condition_deltas[cid_x]
            beh_y, probe_y = condition_deltas[cid_y]

            # Version-pair delta = X delta - Y delta
            common_tasks = sorted(set(beh_x.keys()) & set(beh_y.keys()))
            beh_vp = {tid: beh_x[tid] - beh_y[tid] for tid in common_tasks}
            probe_vp = {tid: probe_x[tid] - probe_y[tid] for tid in common_tasks}

            # Identify target task(s)
            target_tids = [tid for tid in common_tasks if target in EXP3_TASK_TARGETS.get(tid, set())]
            off_target_tids = [tid for tid in common_tasks if tid not in target_tids]

            # Target task metrics
            if target_tids:
                target_beh = np.mean([beh_vp[tid] for tid in target_tids])
                target_probe = np.mean([probe_vp[tid] for tid in target_tids])
            else:
                target_beh = float("nan")
                target_probe = float("nan")

            # Off-target mean absolute delta
            off_beh_abs = np.mean([abs(beh_vp[tid]) for tid in off_target_tids]) if off_target_tids else float("nan")
            off_probe_abs = np.mean([abs(probe_vp[tid]) for tid in off_target_tids]) if off_target_tids else float("nan")

            # Specificity ratio
            beh_specificity = abs(target_beh) / off_beh_abs if off_beh_abs > 0 else float("nan")
            probe_specificity = abs(target_probe) / off_probe_abs if off_probe_abs > 0 else float("nan")

            # Pearson r across all tasks
            beh_arr = np.array([beh_vp[tid] for tid in common_tasks])
            probe_arr = np.array([probe_vp[tid] for tid in common_tasks])
            if len(beh_arr) > 2:
                r, p = stats.pearsonr(beh_arr, probe_arr)
            else:
                r, p = float("nan"), float("nan")

            # Expected sign: for A vs B and A vs C, target should be positive
            # (A has pro-interest, B is neutral, C is anti)
            # For B vs C, target should be positive (B neutral > C anti, so B-C > 0)
            expected_sign = 1.0

            pair_results[pair_key].append({
                "base_role": base_role,
                "target": target,
                "n_tasks": len(common_tasks),
                "n_target": len(target_tids),
                "target_tids": target_tids,
                "beh_probe_r": r,
                "target_beh_delta": target_beh,
                "target_probe_delta": target_probe,
                "target_beh_correct_sign": bool(np.sign(target_beh) == expected_sign) if not np.isnan(target_beh) else False,
                "target_probe_correct_sign": bool(np.sign(target_probe) == expected_sign) if not np.isnan(target_probe) else False,
                "beh_specificity": beh_specificity,
                "probe_specificity": probe_specificity,
                "all_beh_vp": beh_vp,
                "all_probe_vp": probe_vp,
            })

    # Print summary table
    for pair_key, records in pair_results.items():
        print(f"\n{'='*60}")
        print(f"  {pair_key.replace('_', ' ').upper()}")
        print(f"{'='*60}")

        valid = [r for r in records if not np.isnan(r["target_beh_delta"])]
        if not valid:
            print("  No valid records")
            continue

        # Aggregate stats
        all_beh = []
        all_probe = []
        for rec in valid:
            for tid in rec["all_beh_vp"]:
                all_beh.append(rec["all_beh_vp"][tid])
                all_probe.append(rec["all_probe_vp"][tid])
        all_beh_arr = np.array(all_beh)
        all_probe_arr = np.array(all_probe)
        overall_r = stats.pearsonr(all_beh_arr, all_probe_arr)[0]

        beh_correct = sum(r["target_beh_correct_sign"] for r in valid)
        probe_correct = sum(r["target_probe_correct_sign"] for r in valid)
        mean_beh_spec = np.mean([r["beh_specificity"] for r in valid if not np.isnan(r["beh_specificity"])])
        mean_probe_spec = np.mean([r["probe_specificity"] for r in valid if not np.isnan(r["probe_specificity"])])

        print(f"  Overall beh↔probe r: {overall_r:.3f} (n={len(all_beh)})")
        print(f"  Target sign correct: beh {beh_correct}/{len(valid)}, probe {probe_correct}/{len(valid)}")
        print(f"  Mean specificity: beh {mean_beh_spec:.2f}, probe {mean_probe_spec:.2f}")
        print()
        print(f"  {'Role':<12} {'Target':<14} {'Beh Δ':>7} {'Probe Δ':>8} {'Beh spec':>9} {'Probe spec':>11} {'r':>6}")
        print(f"  {'-'*12} {'-'*14} {'-'*7} {'-'*8} {'-'*9} {'-'*11} {'-'*6}")
        for r in sorted(valid, key=lambda x: (x["base_role"], x["target"])):
            print(
                f"  {r['base_role']:<12} {r['target']:<14} "
                f"{r['target_beh_delta']:>+7.3f} {r['target_probe_delta']:>+8.2f} "
                f"{r['beh_specificity']:>9.1f} {r['probe_specificity']:>11.1f} "
                f"{r['beh_probe_r']:>6.3f}"
            )

    # --- Plot: 3-panel version-pair scatter ---
    fig = plt.figure(figsize=(15, 6.5))
    # Top area for the prompt table, bottom for scatter panels
    gs = fig.add_gridspec(2, 3, height_ratios=[0.25, 1], hspace=0.35)

    # --- Prompt version table ---
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis("off")
    table_data = [
        ["A (pro)", "You grew up in a mid-sized city in the Midwest... [8 sentences] ...You love discussing the plot structure of Shakespeare's plays."],
        ["B (neutral)", "You grew up in a mid-sized city in the Midwest... [8 sentences] ...You love discussing the best hiking trails near your home."],
        ["C (anti)", "You grew up in a mid-sized city in the Midwest... [8 sentences] ...You find discussing the plot structure of Shakespeare's plays painfully dull."],
    ]
    table = ax_table.table(
        cellText=table_data,
        colLabels=["Version", "System prompt (10 sentences, example: midwest role × shakespeare target)"],
        loc="center",
        cellLoc="left",
        colWidths=[0.1, 0.85],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E8E8E8")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#CCCCCC")

    # Target task annotation below the table
    ax_table.text(
        0.5, -0.15,
        'Target task for shakespeare: "Describe the plot of Shakespeare\'s play, Romeo and Juliet."',
        transform=ax_table.transAxes, fontsize=9, ha="center", style="italic", color="#555555",
    )

    pair_labels = {"A_vs_B": "A vs B\n(pro vs neutral)", "B_vs_C": "B vs C\n(neutral vs anti)", "A_vs_C": "A vs C\n(pro vs anti)"}

    axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    for idx, (pair_key, records) in enumerate(pair_results.items()):
        ax = axes[idx]

        all_beh = []
        all_probe = []
        all_is_target = []

        for rec in records:
            for tid in rec["all_beh_vp"]:
                all_beh.append(rec["all_beh_vp"][tid])
                all_probe.append(rec["all_probe_vp"][tid])
                all_is_target.append(tid in rec["target_tids"])

        beh_arr = np.array(all_beh)
        probe_arr = np.array(all_probe)
        target_mask = np.array(all_is_target)

        # Off-target
        ax.scatter(
            beh_arr[~target_mask], probe_arr[~target_mask],
            alpha=0.4, s=15, color="#BDBDBD", edgecolors="none", zorder=1,
            label="Other tasks",
        )
        # On-target
        ax.scatter(
            beh_arr[target_mask], probe_arr[target_mask],
            s=50, color="#e41a1c", marker="*", edgecolors="black", linewidths=0.3,
            zorder=3, label="Target task",
        )

        # Fit line (all points)
        if len(beh_arr) > 2:
            slope, intercept, r, p, se = stats.linregress(beh_arr, probe_arr)
            x_fit = np.linspace(beh_arr.min(), beh_arr.max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, color="#2196F3", linewidth=1.5, alpha=0.8)

        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5)

        ax.set_xlabel("Behavioral delta", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Probe delta", fontsize=9)
        ax.set_title(pair_labels[pair_key], fontsize=11, fontweight="bold")

        # Stats
        valid = [rec for rec in records if not np.isnan(rec["target_beh_delta"])]
        beh_correct = sum(rec["target_beh_correct_sign"] for rec in valid)
        probe_correct = sum(rec["target_probe_correct_sign"] for rec in valid)
        mean_beh_spec = np.mean([rec["beh_specificity"] for rec in valid if not np.isnan(rec["beh_specificity"])])
        mean_probe_spec = np.mean([rec["probe_specificity"] for rec in valid if not np.isnan(rec["probe_specificity"])])

        overall_r = stats.pearsonr(beh_arr, probe_arr)[0] if len(beh_arr) > 2 else float("nan")

        stats_text = (
            f"r = {overall_r:.2f} (n={len(beh_arr)})\n"
            f"Target sign: beh {beh_correct}/{len(valid)}, probe {probe_correct}/{len(valid)}\n"
            f"Specificity: beh {mean_beh_spec:.1f}×, probe {mean_probe_spec:.1f}×"
        )
        ax.text(
            0.05, 0.95, stats_text,
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="grey"),
        )

        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Exp 3: Version-pair deltas (★ = target task)", fontsize=12, y=0.98)
    out = ASSETS_DIR / "plot_022126_exp3_version_pairs.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    # Save JSON summary (without full per-task data)
    summary = {}
    for pair_key, records in pair_results.items():
        summary[pair_key] = []
        for rec in records:
            summary[pair_key].append({
                k: v for k, v in rec.items()
                if k not in ("all_beh_vp", "all_probe_vp")
            })
    out_json = ASSETS_DIR / "exp3_version_pair_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
