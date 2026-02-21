"""Exp 3 selectivity analysis: does the probe fire selectively on the target task?

For each (base_role, target) pair with version A (pro):
- Identify the "target task" = task with largest behavioral delta under A
- Compare behavioral and probe deltas for target task vs other tasks
- Plot: for each condition, highlight the target task in the scatter
"""

import json
import os
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


def load_probe(layer: int):
    probe = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return probe[:-1], float(probe[-1])


def score_activations(npz_path: Path, layer: int, weights: np.ndarray, bias: float):
    data = np.load(npz_path, allow_pickle=True)
    acts = data[f"layer_{layer}"]
    scores = acts @ weights + bias
    task_ids = list(data["task_ids"])
    return dict(zip(task_ids, scores.tolist()))


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    beh = json.load(open(BEH_PATH))
    cfg = json.load(open(CFG_PATH))
    cond_info = {c["condition_id"]: c for c in cfg["conditions"]}

    weights, bias = load_probe(LAYER)

    # Score baseline
    baseline_scores = score_activations(
        ACTS_DIR / "baseline" / "activations_prompt_last.npz", LAYER, weights, bias
    )
    baseline_rates = {
        tid: v["p_choose"] for tid, v in beh["conditions"]["baseline"]["task_rates"].items()
    }

    tasks = sorted(baseline_rates.keys())

    # For each A condition: compute per-task deltas (behavioral and probe)
    # Also compute B condition deltas for comparison
    a_conditions = sorted(
        cid for cid in os.listdir(ACTS_DIR) if cid.endswith("_A")
    )

    # Collect: for each condition, (target_task, target_beh_delta, target_probe_delta, other_beh, other_probe)
    records = []

    for cid_a in a_conditions:
        info = cond_info[cid_a]
        target_name = info["target"]
        base_role = info["base_role"]

        # Score activations for A
        npz_a = ACTS_DIR / cid_a / "activations_prompt_last.npz"
        if not npz_a.exists():
            continue
        scores_a = score_activations(npz_a, LAYER, weights, bias)

        # Behavioral rates for A
        if cid_a not in beh["conditions"]:
            continue
        rates_a = {
            tid: v["p_choose"] for tid, v in beh["conditions"][cid_a]["task_rates"].items()
        }

        # Compute deltas
        beh_deltas = {}
        probe_deltas = {}
        for tid in tasks:
            if tid in rates_a and tid in baseline_rates and tid in scores_a and tid in baseline_scores:
                beh_deltas[tid] = rates_a[tid] - baseline_rates[tid]
                probe_deltas[tid] = scores_a[tid] - baseline_scores[tid]

        # Find target task = task with largest positive behavioral delta
        target_task = max(beh_deltas, key=lambda t: beh_deltas[t])

        other_tids = [t for t in beh_deltas if t != target_task]

        records.append({
            "condition": cid_a,
            "target_name": target_name,
            "base_role": base_role,
            "target_task": target_task,
            "target_beh": beh_deltas[target_task],
            "target_probe": probe_deltas[target_task],
            "other_beh": [beh_deltas[t] for t in other_tids],
            "other_probe": [probe_deltas[t] for t in other_tids],
            "all_beh": beh_deltas,
            "all_probe": probe_deltas,
        })

    # --- Plot 1: Per-condition scatter highlighting target task ---
    n_conds = len(records)
    ncols = 5
    nrows = (n_conds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for idx, rec in enumerate(records):
        ax = axes[idx]
        ob = np.array(rec["other_beh"])
        op = np.array(rec["other_probe"])

        ax.scatter(ob, op, alpha=0.4, s=15, color="#377eb8", zorder=2)
        ax.scatter(
            rec["target_beh"], rec["target_probe"],
            s=80, color="#e41a1c", zorder=3, marker="*", edgecolors="black", linewidths=0.5,
        )
        ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.4, alpha=0.5)
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(f"{rec['base_role']}_{rec['target_name']}", fontsize=8)
        ax.tick_params(labelsize=6)

        # Annotate: rank of target task by probe delta
        all_probe_sorted = sorted(rec["all_probe"].values(), reverse=True)
        probe_rank = all_probe_sorted.index(rec["target_probe"]) + 1
        ax.text(
            0.97, 0.03,
            f"beh rank: 1/50\nprobe rank: {probe_rank}/50",
            transform=ax.transAxes, fontsize=6, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    for idx in range(n_conds, len(axes)):
        axes[idx].set_visible(False)

    fig.supxlabel("Behavioral delta", fontsize=10)
    fig.supylabel("Probe delta", fontsize=10)
    fig.suptitle(
        "Exp 3: Minimal Pairs — Target task selectivity (★ = target task, highest behavioral delta)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    save_path = ASSETS_DIR / "plot_022126_exp3_selectivity.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

    # --- Plot 2: Summary — target task probe rank distribution ---
    probe_ranks = []
    for rec in records:
        all_probe_sorted = sorted(rec["all_probe"].values(), reverse=True)
        rank = all_probe_sorted.index(rec["target_probe"]) + 1
        probe_ranks.append(rank)

    probe_ranks = np.array(probe_ranks)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Histogram of probe ranks
    ax1.hist(probe_ranks, bins=np.arange(0.5, 51.5, 1), color="#377eb8", alpha=0.7, edgecolor="white")
    ax1.axvline(probe_ranks.mean(), color="#e41a1c", linewidth=1.5, linestyle="--",
                label=f"Mean rank: {probe_ranks.mean():.1f}")
    ax1.axvline(25.5, color="gray", linewidth=1, linestyle=":", label="Chance (25.5)")
    ax1.set_xlabel("Probe delta rank of target task (1=highest)")
    ax1.set_ylabel("Count")
    ax1.set_title("Target task rank by probe delta")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 51)

    # Scatter: behavioral delta of target vs probe delta of target
    target_behs = [r["target_beh"] for r in records]
    target_probes = [r["target_probe"] for r in records]
    ax2.scatter(target_behs, target_probes, s=40, color="#e41a1c", alpha=0.7, zorder=2)
    r_target, _ = stats.pearsonr(target_behs, target_probes)
    ax2.set_xlabel("Target task behavioral delta")
    ax2.set_ylabel("Target task probe delta")
    ax2.set_title(f"Target task: behavioral vs probe delta (r={r_target:.3f})")
    ax2.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax2.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    n_top5 = (probe_ranks <= 5).sum()
    n_top10 = (probe_ranks <= 10).sum()
    fig.suptitle(
        f"Exp 3: Target task probe selectivity — "
        f"top 5: {n_top5}/{len(probe_ranks)}, top 10: {n_top10}/{len(probe_ranks)}, "
        f"mean rank: {probe_ranks.mean():.1f}/50",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    save_path2 = ASSETS_DIR / "plot_022126_exp3_selectivity_summary.png"
    fig.savefig(save_path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path2}")

    # Print summary
    print(f"\nTarget task probe rank distribution:")
    print(f"  Mean: {probe_ranks.mean():.1f}, Median: {np.median(probe_ranks):.0f}")
    print(f"  Top 1: {(probe_ranks == 1).sum()}/{len(probe_ranks)}")
    print(f"  Top 5: {n_top5}/{len(probe_ranks)}")
    print(f"  Top 10: {n_top10}/{len(probe_ranks)}")
    print(f"  Chance mean: 25.5")


if __name__ == "__main__":
    main()
