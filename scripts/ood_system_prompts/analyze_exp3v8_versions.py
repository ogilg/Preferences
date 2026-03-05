"""Exp 3 v8 version-pair analysis: compare A vs B, B vs C, A vs C deltas.

Replicates analyze_exp3_versions.py for the v8 experiment with 20 targets,
50 heldout tasks, 2 roles (midwest, brooklyn), 3 versions (A/B/C).

Usage: python -m scripts.ood_system_prompts.analyze_exp3v8_versions
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.ood.analysis import compute_p_choose_from_pairwise

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ACTS_DIR = REPO_ROOT / "activations" / "ood" / "exp3v8_minimal_pairs"
PROBE_DIR = REPO_ROOT / "results" / "probes" / "gemma3_10k_heldout_std_demean" / "probes"
PAIRWISE_PATH = REPO_ROOT / "results" / "ood" / "minimal_pairs_v8" / "pairwise.json"
CFG_PATH = REPO_ROOT / "configs" / "ood" / "prompts" / "minimal_pairs_v8.json"
PREFS_PATH = REPO_ROOT / "configs" / "ood" / "preferences" / "exp3_v8_preferences.json"
ASSETS_DIR = REPO_ROOT / "experiments" / "ood_system_prompts" / "exp3_v8" / "assets"

LAYER = 31
VERSION_PAIRS = [("A", "B"), ("B", "C"), ("A", "C")]


def load_probe(layer: int) -> tuple[np.ndarray, float]:
    probe = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return probe[:-1], float(probe[-1])


def score_activations(npz_path: Path, layer: int, weights: np.ndarray, bias: float) -> dict[str, float]:
    data = np.load(npz_path, allow_pickle=True)
    acts = data[f"layer_{layer}"]
    scores = acts @ weights + bias
    task_ids = list(data["task_ids"])
    return dict(zip(task_ids, scores.tolist()))


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Load pairwise data and aggregate to p_choose
    pairwise = json.load(open(PAIRWISE_PATH))
    rates = compute_p_choose_from_pairwise(pairwise["results"])

    cfg = json.load(open(CFG_PATH))
    cond_info = {c["condition_id"]: c for c in cfg["conditions"]}

    # Load preferences for task-target mapping
    prefs = json.load(open(PREFS_PATH))
    target_task_ids = {p["task_id"] for p in prefs}

    weights, bias = load_probe(LAYER)

    # Baseline
    baseline_scores = score_activations(
        ACTS_DIR / "baseline" / "activations_prompt_last.npz", LAYER, weights, bias
    )
    baseline_rates = rates["baseline"]
    tasks = sorted(baseline_rates.keys())
    print(f"Tasks: {len(tasks)}, Targets: {len(target_task_ids)}")

    # Per-condition deltas
    condition_deltas: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for cid in cond_info:
        npz = ACTS_DIR / cid / "activations_prompt_last.npz"
        if not npz.exists():
            print(f"  Missing activations: {cid}")
            continue
        cond_scores = score_activations(npz, LAYER, weights, bias)
        if cid not in rates:
            continue
        cond_rates = rates[cid]

        beh_d, probe_d = {}, {}
        for tid in tasks:
            if tid in cond_rates and tid in baseline_rates and tid in cond_scores and tid in baseline_scores:
                beh_d[tid] = cond_rates[tid] - baseline_rates[tid]
                probe_d[tid] = cond_scores[tid] - baseline_scores[tid]
        if beh_d:
            condition_deltas[cid] = (beh_d, probe_d)

    # Version-pair analysis
    pair_results: dict[str, list[dict]] = {}

    for v_x, v_y in VERSION_PAIRS:
        pair_key = f"{v_x}_vs_{v_y}"
        pair_results[pair_key] = []

        # Group conditions by (base_role, target)
        groups: dict[tuple[str, str], dict[str, str]] = {}
        for cid, info in cond_info.items():
            key = (info["base_role"], info["target"])
            if key not in groups:
                groups[key] = {}
            groups[key][info["version"]] = cid

        for (base_role, target), version_cids in sorted(groups.items()):
            if v_x not in version_cids or v_y not in version_cids:
                continue
            cid_x, cid_y = version_cids[v_x], version_cids[v_y]
            if cid_x not in condition_deltas or cid_y not in condition_deltas:
                continue

            beh_x, probe_x = condition_deltas[cid_x]
            beh_y, probe_y = condition_deltas[cid_y]

            common_tasks = sorted(set(beh_x.keys()) & set(beh_y.keys()))
            beh_vp = {tid: beh_x[tid] - beh_y[tid] for tid in common_tasks}
            probe_vp = {tid: probe_x[tid] - probe_y[tid] for tid in common_tasks}

            # Target = the task matching this condition's target
            target_tids = [tid for tid in common_tasks if tid == target]
            off_target_tids = [tid for tid in common_tasks if tid != target]

            target_beh = beh_vp[target_tids[0]] if target_tids else float("nan")
            target_probe = probe_vp[target_tids[0]] if target_tids else float("nan")

            off_beh_abs = np.mean([abs(beh_vp[tid]) for tid in off_target_tids]) if off_target_tids else float("nan")
            off_probe_abs = np.mean([abs(probe_vp[tid]) for tid in off_target_tids]) if off_target_tids else float("nan")

            beh_specificity = abs(target_beh) / off_beh_abs if off_beh_abs > 0 else float("nan")
            probe_specificity = abs(target_probe) / off_probe_abs if off_probe_abs > 0 else float("nan")

            beh_arr = np.array([beh_vp[tid] for tid in common_tasks])
            probe_arr = np.array([probe_vp[tid] for tid in common_tasks])
            r = stats.pearsonr(beh_arr, probe_arr)[0] if len(beh_arr) > 2 else float("nan")

            pair_results[pair_key].append({
                "base_role": base_role,
                "target": target,
                "n_tasks": len(common_tasks),
                "target_tids": target_tids,
                "beh_probe_r": r,
                "target_beh_delta": float(target_beh),
                "target_probe_delta": float(target_probe),
                "target_beh_correct_sign": bool(target_beh > 0) if not np.isnan(target_beh) else False,
                "target_probe_correct_sign": bool(target_probe > 0) if not np.isnan(target_probe) else False,
                "beh_specificity": float(beh_specificity),
                "probe_specificity": float(probe_specificity),
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

        all_beh, all_probe = [], []
        for rec in valid:
            for tid in rec["all_beh_vp"]:
                all_beh.append(rec["all_beh_vp"][tid])
                all_probe.append(rec["all_probe_vp"][tid])
        overall_r = stats.pearsonr(all_beh, all_probe)[0]

        beh_correct = sum(r["target_beh_correct_sign"] for r in valid)
        probe_correct = sum(r["target_probe_correct_sign"] for r in valid)
        mean_beh_spec = np.mean([r["beh_specificity"] for r in valid if not np.isnan(r["beh_specificity"])])
        mean_probe_spec = np.mean([r["probe_specificity"] for r in valid if not np.isnan(r["probe_specificity"])])

        print(f"  Overall beh↔probe r: {overall_r:.3f} (n={len(all_beh)})")
        print(f"  Target sign correct: beh {beh_correct}/{len(valid)}, probe {probe_correct}/{len(valid)}")
        print(f"  Mean specificity: beh {mean_beh_spec:.2f}×, probe {mean_probe_spec:.2f}×")
        print()
        print(f"  {'Role':<12} {'Target':<28} {'Beh Δ':>7} {'Probe Δ':>8} {'Beh spec':>9} {'Probe spec':>11} {'r':>6}")
        print(f"  {'-'*12} {'-'*28} {'-'*7} {'-'*8} {'-'*9} {'-'*11} {'-'*6}")
        for r in sorted(valid, key=lambda x: (x["base_role"], x["target"])):
            print(
                f"  {r['base_role']:<12} {r['target']:<28} "
                f"{r['target_beh_delta']:>+7.3f} {r['target_probe_delta']:>+8.2f} "
                f"{r['beh_specificity']:>9.1f}× {r['probe_specificity']:>10.1f}× "
                f"{r['beh_probe_r']:>6.3f}"
            )

    # --- Plot: 3-panel version-pair scatter ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pair_labels = {
        "A_vs_B": "A vs B\n(pro vs neutral)",
        "B_vs_C": "B vs C\n(neutral vs anti)",
        "A_vs_C": "A vs C\n(pro vs anti)",
    }

    for idx, (pair_key, records) in enumerate(pair_results.items()):
        ax = axes[idx]

        all_beh, all_probe, all_is_target = [], [], []
        for rec in records:
            for tid in rec["all_beh_vp"]:
                all_beh.append(rec["all_beh_vp"][tid])
                all_probe.append(rec["all_probe_vp"][tid])
                all_is_target.append(tid in rec["target_tids"])

        beh_arr = np.array(all_beh)
        probe_arr = np.array(all_probe)
        target_mask = np.array(all_is_target)

        ax.scatter(
            beh_arr[~target_mask], probe_arr[~target_mask],
            alpha=0.3, s=12, color="#BDBDBD", edgecolors="none", zorder=1,
            label="Off-target tasks",
        )
        ax.scatter(
            beh_arr[target_mask], probe_arr[target_mask],
            s=60, color="#e41a1c", marker="*", edgecolors="black", linewidths=0.3,
            zorder=3, label="Target task",
        )

        if len(beh_arr) > 2:
            slope, intercept, r, p, se = stats.linregress(beh_arr, probe_arr)
            x_fit = np.linspace(beh_arr.min(), beh_arr.max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, color="#2196F3", linewidth=1.5, alpha=0.8)

        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5)
        ax.set_xlabel("Behavioral delta", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Probe delta", fontsize=10)
        ax.set_title(pair_labels[pair_key], fontsize=11, fontweight="bold")

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

    fig.suptitle("Exp 3 v8: Version-pair deltas (20 targets, 50 heldout tasks)", fontsize=12, y=1.02)
    fig.tight_layout()
    out_plot = ASSETS_DIR / "plot_030426_exp3v8_version_pairs.png"
    fig.savefig(out_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_plot}")

    # --- Selectivity: probe rank of target task across version A conditions ---
    print(f"\n{'='*60}")
    print("  SELECTIVITY (Version A conditions)")
    print(f"{'='*60}")

    target_probe_ranks = []
    target_beh_ranks = []
    for cid, (beh_d, probe_d) in condition_deltas.items():
        info = cond_info.get(cid)
        if info is None or info["version"] != "A":
            continue
        target_tid = info["target"]
        if target_tid not in beh_d:
            continue

        # Rank by behavioral delta (descending)
        sorted_beh = sorted(beh_d.items(), key=lambda x: -x[1])
        beh_rank = next(i + 1 for i, (tid, _) in enumerate(sorted_beh) if tid == target_tid)

        # Rank by probe delta (descending)
        sorted_probe = sorted(probe_d.items(), key=lambda x: -x[1])
        probe_rank = next(i + 1 for i, (tid, _) in enumerate(sorted_probe) if tid == target_tid)

        target_beh_ranks.append(beh_rank)
        target_probe_ranks.append(probe_rank)
        print(f"  {cid:<40} beh_rank={beh_rank:>2}/50  probe_rank={probe_rank:>2}/50")

    print(f"\n  Mean behavioral rank: {np.mean(target_beh_ranks):.1f} / 50 (chance = 25.5)")
    print(f"  Mean probe rank:     {np.mean(target_probe_ranks):.1f} / 50 (chance = 25.5)")
    print(f"  Median probe rank:   {np.median(target_probe_ranks):.0f} / 50")

    # --- Save JSON summary ---
    summary = {}
    for pair_key, records in pair_results.items():
        summary[pair_key] = [
            {k: v for k, v in rec.items() if k not in ("all_beh_vp", "all_probe_vp")}
            for rec in records
        ]
    summary["selectivity"] = {
        "mean_beh_rank": float(np.mean(target_beh_ranks)),
        "mean_probe_rank": float(np.mean(target_probe_ranks)),
        "median_probe_rank": float(np.median(target_probe_ranks)),
        "n_conditions": len(target_probe_ranks),
    }
    out_json = ASSETS_DIR / "exp3v8_version_pair_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
