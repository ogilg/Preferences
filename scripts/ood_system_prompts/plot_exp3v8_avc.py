"""A vs C scatter for exp3 v8 fine-grained experiment (LW post style).

Single panel: filled stars for target tasks where probe ranks them #1/50,
unfilled stars otherwise. Dashed fit line.

Usage: python -m scripts.ood_system_prompts.plot_exp3v8_avc
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from src.ood.analysis import compute_p_choose_from_pairwise

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ACTS_DIR = REPO_ROOT / "activations" / "ood" / "exp3v8_minimal_pairs"
PROBE_DIR = REPO_ROOT / "results" / "probes" / "gemma3_10k_heldout_std_demean" / "probes"
PAIRWISE_PATH = REPO_ROOT / "results" / "ood" / "minimal_pairs_v8" / "pairwise.json"
CFG_PATH = REPO_ROOT / "configs" / "ood" / "prompts" / "minimal_pairs_v8.json"
PREFS_PATH = REPO_ROOT / "configs" / "ood" / "preferences" / "exp3_v8_preferences.json"
OUT_DIR = REPO_ROOT / "experiments" / "ood_system_prompts" / "exp3_v8" / "assets"

LAYER = 31
GREY = "#BDBDBD"
N_TASKS = 50


def load_probe(layer: int) -> tuple[np.ndarray, float]:
    probe = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return probe[:-1], float(probe[-1])


def score_activations(npz_path: Path, layer: int, weights: np.ndarray, bias: float) -> dict[str, float]:
    data = np.load(npz_path, allow_pickle=True)
    acts = data[f"layer_{layer}"]
    scores = acts @ weights + bias
    return dict(zip(list(data["task_ids"]), scores.tolist()))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pairwise = json.load(open(PAIRWISE_PATH))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    cfg = json.load(open(CFG_PATH))
    cond_info = {c["condition_id"]: c for c in cfg["conditions"]}

    prefs = json.load(open(PREFS_PATH))
    target_task_ids = {p["task_id"] for p in prefs}

    weights, bias = load_probe(LAYER)
    baseline_scores = score_activations(
        ACTS_DIR / "baseline" / "activations_prompt_last.npz", LAYER, weights, bias
    )
    baseline_rates = rates["baseline"]
    tasks = sorted(baseline_rates.keys())

    # Per-condition deltas
    condition_deltas: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for cid in cond_info:
        npz = ACTS_DIR / cid / "activations_prompt_last.npz"
        if not npz.exists():
            continue
        cond_scores = score_activations(npz, LAYER, weights, bias)
        cond_rates = rates.get(cid, {})
        beh_d, probe_d = {}, {}
        for tid in tasks:
            if tid in cond_rates and tid in baseline_rates and tid in cond_scores and tid in baseline_scores:
                beh_d[tid] = cond_rates[tid] - baseline_rates[tid]
                probe_d[tid] = cond_scores[tid] - baseline_scores[tid]
        if beh_d:
            condition_deltas[cid] = (beh_d, probe_d)

    # Group by (base_role, target)
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for cid, info in cond_info.items():
        key = (info["base_role"], info["target"])
        if key not in groups:
            groups[key] = {}
        groups[key][info["version"]] = cid

    # Compute A vs C version-pair deltas
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

        for tid in common:
            is_target = (tid == target)
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

    print(f"Total points: {len(beh_arr)}")
    print(f"Target points: {n_target_total}, probe rank 1: {n_probe_rank1}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(beh_arr[~target_mask], probe_arr[~target_mask],
               alpha=0.3, s=12, color=GREY, edgecolors="none", zorder=1,
               label="Other tasks")
    if rank1_mask.any():
        ax.scatter(beh_arr[rank1_mask], probe_arr[rank1_mask],
                   s=70, color="#e41a1c", marker="*", edgecolors="black",
                   linewidths=0.3, zorder=3,
                   label=f"Target (probe rank 1/{N_TASKS})")
    if target_not_rank1.any():
        ax.scatter(beh_arr[target_not_rank1], probe_arr[target_not_rank1],
                   s=70, facecolors="none", marker="*", edgecolors="#e41a1c",
                   linewidths=1.0, zorder=3,
                   label=f"Target (not rank 1/{N_TASKS})")

    # Dashed fit line
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
    out = OUT_DIR / "plot_030526_exp3v8_avc.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
