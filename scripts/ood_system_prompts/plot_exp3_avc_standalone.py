"""Standalone A vs C version-pair scatter for the LW post.

Single panel with prompt table at top, target tasks highlighted.

Usage: python -m scripts.ood_system_prompts.plot_exp3_avc_standalone
"""

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
OUT_PATH = REPO_ROOT / "docs" / "lw_post" / "assets" / "plot_022126_exp3_avc.png"

LAYER = 31
SELECTED_ROLES = {"midwest", "brooklyn"}

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


def load_probe(layer: int):
    probe = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return probe[:-1], float(probe[-1])


def score_activations(npz_path, layer, weights, bias):
    data = np.load(npz_path, allow_pickle=True)
    acts = data[f"layer_{layer}"]
    scores = acts @ weights + bias
    task_ids = list(data["task_ids"])
    return dict(zip(task_ids, scores.tolist()))


def main():
    beh_data = json.load(open(BEH_PATH))
    cfg = json.load(open(CFG_PATH))
    cond_info = {c["condition_id"]: c for c in cfg["conditions"]}

    weights, bias = load_probe(LAYER)
    baseline_scores = score_activations(
        ACTS_DIR / "baseline" / "activations_prompt_last.npz", LAYER, weights, bias
    )
    baseline_rates = {
        tid: v["p_choose"] for tid, v in beh_data["conditions"]["baseline"]["task_rates"].items()
    }
    tasks = sorted(baseline_rates.keys())

    # Group by (base_role, target)
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for cid, info in cond_info.items():
        if info["base_role"] not in SELECTED_ROLES:
            continue
        key = (info["base_role"], info["target"])
        if key not in groups:
            groups[key] = {}
        groups[key][info["version"]] = cid

    # Compute A vs C version-pair deltas
    all_beh = []
    all_probe = []
    all_is_target = []

    for (base_role, target), version_cids in sorted(groups.items()):
        if "A" not in version_cids or "C" not in version_cids:
            continue
        cid_a = version_cids["A"]
        cid_c = version_cids["C"]

        npz_a = ACTS_DIR / cid_a / "activations_prompt_last.npz"
        npz_c = ACTS_DIR / cid_c / "activations_prompt_last.npz"
        if not npz_a.exists() or not npz_c.exists():
            continue

        scores_a = score_activations(npz_a, LAYER, weights, bias)
        scores_c = score_activations(npz_c, LAYER, weights, bias)
        rates_a = {tid: v["p_choose"] for tid, v in beh_data["conditions"][cid_a]["task_rates"].items()}
        rates_c = {tid: v["p_choose"] for tid, v in beh_data["conditions"][cid_c]["task_rates"].items()}

        for tid in tasks:
            if all(tid in d for d in [rates_a, rates_c, baseline_rates, scores_a, scores_c, baseline_scores]):
                beh_a = rates_a[tid] - baseline_rates[tid]
                beh_c = rates_c[tid] - baseline_rates[tid]
                probe_a = scores_a[tid] - baseline_scores[tid]
                probe_c = scores_c[tid] - baseline_scores[tid]
                all_beh.append(beh_a - beh_c)
                all_probe.append(probe_a - probe_c)
                all_is_target.append(target in EXP3_TASK_TARGETS.get(tid, set()))

    beh_arr = np.array(all_beh)
    probe_arr = np.array(all_probe)
    target_mask = np.array(all_is_target)

    # --- Plot ---
    fig = plt.figure(figsize=(8, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.3, 1], hspace=0.35)

    # Table
    ax_table = fig.add_subplot(gs[0])
    ax_table.axis("off")
    table_data = [
        ["A (pro)", "You grew up in a mid-sized city in the Midwest...\n[8 sentences] ...You love discussing the plot structure of Shakespeare's plays."],
        ["B (neutral)", "You grew up in a mid-sized city in the Midwest...\n[8 sentences] ...You love discussing the best hiking trails near your home."],
        ["C (anti)", "You grew up in a mid-sized city in the Midwest...\n[8 sentences] ...You find discussing Shakespeare's plays painfully dull."],
    ]
    table = ax_table.table(
        cellText=table_data,
        colLabels=["Version", "System prompt (10 sentences, example: midwest × shakespeare)"],
        loc="center",
        cellLoc="left",
        colWidths=[0.12, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E8E8E8")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#CCCCCC")

    ax_table.text(
        0.5, -0.15,
        'Target task for shakespeare: "Describe the plot of Shakespeare\'s play, Romeo and Juliet."',
        transform=ax_table.transAxes, fontsize=9, ha="center", style="italic", color="#555555",
    )

    # Scatter
    ax = fig.add_subplot(gs[1])

    ax.scatter(
        beh_arr[~target_mask], probe_arr[~target_mask],
        alpha=0.4, s=15, color="#BDBDBD", edgecolors="none", zorder=1,
        label="Other tasks",
    )
    ax.scatter(
        beh_arr[target_mask], probe_arr[target_mask],
        s=70, color="#e41a1c", marker="*", edgecolors="black", linewidths=0.3,
        zorder=3, label="Target task",
    )

    # Fit line
    slope, intercept, r, p, se = stats.linregress(beh_arr, probe_arr)
    x_fit = np.linspace(beh_arr.min(), beh_arr.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, color="#2196F3", linewidth=1.5, alpha=0.8)

    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("Behavioral delta (A minus C)", fontsize=10)
    ax.set_ylabel("Probe delta (A minus C)", fontsize=10)
    ax.set_title("A vs C (pro-interest vs anti-interest)", fontsize=12, fontweight="bold")

    # Stats
    n_target = int(target_mask.sum())
    target_beh = beh_arr[target_mask]
    target_probe = probe_arr[target_mask]
    beh_correct = int((target_beh > 0).sum())
    probe_correct = int((target_probe > 0).sum())
    off_beh_abs = np.mean(np.abs(beh_arr[~target_mask]))
    off_probe_abs = np.mean(np.abs(probe_arr[~target_mask]))
    mean_beh_spec = np.mean(np.abs(target_beh)) / off_beh_abs
    mean_probe_spec = np.mean(np.abs(target_probe)) / off_probe_abs

    overall_r = stats.pearsonr(beh_arr, probe_arr)[0]

    stats_text = (
        f"r = {overall_r:.2f} (n={len(beh_arr)})\n"
        f"Target sign correct: beh {beh_correct}/{n_target}, probe {probe_correct}/{n_target}\n"
        f"Specificity: beh {mean_beh_spec:.1f}×, probe {mean_probe_spec:.1f}×"
    )
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="grey"),
    )

    ax.legend(fontsize=9, loc="lower right")

    out = OUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
