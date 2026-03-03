"""Plot probe r vs baseline utilities r for exp1b only (all tasks)."""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats

plt.style.use("seaborn-v0_8-whitegrid")


def pairwise_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    n = len(predicted)
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if actual[i] == actual[j]:
                continue
            total += 1
            if (predicted[i] - predicted[j]) * (actual[i] - actual[j]) > 0:
                correct += 1
    return correct / total if total > 0 else float("nan")

BASE = Path(".")
LAYER = 31
RUN_PREFIX = "completion_preference_gemma-3-27b_completion_canonical_seed0"

ASSETS = BASE / "docs/lw_post/assets"
ASSETS.mkdir(parents=True, exist_ok=True)

EXP_INFO = {
    "config_dir": BASE / "configs/measurement/active_learning/ood_exp1b",
    "results_dir": BASE / "results/experiments/ood_exp1b/pre_task_active_learning",
    "act_dir_base": BASE / "activations/ood/exp1_prompts",
}


def load_thurstonian_latest(run_dir: Path) -> dict[str, float]:
    csvs = sorted(run_dir.glob("thurstonian_*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No thurstonian CSVs in {run_dir}")
    utilities = {}
    with open(csvs[-1]) as f:
        next(f)
        for line in f:
            task_id, mu, _ = line.strip().split(",")
            utilities[task_id] = float(mu)
    return utilities


def main():
    probe_dir = BASE / "results/probes/gemma3_10k_heldout_std_raw"
    probe_weights = np.load(probe_dir / "probes" / f"probe_ridge_L{LAYER:02d}.npy")

    results_dir = EXP_INFO["results_dir"]
    config_dir = EXP_INFO["config_dir"]
    act_dir_base = EXP_INFO["act_dir_base"]

    baseline_dir = results_dir / RUN_PREFIX
    baseline_utils = load_thurstonian_latest(baseline_dir)
    baseline_task_ids = set(baseline_utils.keys())

    cond_map = {}
    for f in sorted(config_dir.glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text())
        sp = cfg.get("measurement_system_prompt", "")
        if sp:
            h = hashlib.sha256(sp.encode()).hexdigest()[:8]
            cond_map[f.stem] = f"_sys{h}"
        else:
            cond_map[f.stem] = ""

    baseline_rs, probe_rs = [], []
    baseline_accs, probe_accs = [], []

    for cond_name, suffix in cond_map.items():
        if cond_name == "baseline":
            continue
        cond_result_dir = results_dir / f"{RUN_PREFIX}{suffix}"
        if not cond_result_dir.exists():
            continue
        try:
            cond_utils = load_thurstonian_latest(cond_result_dir)
        except FileNotFoundError:
            continue

        act_path = act_dir_base / cond_name / "activations_prompt_last.npz"
        if not act_path.exists():
            continue

        d = np.load(act_path)
        act_task_ids = list(d["task_ids"])
        acts = d[f"layer_{LAYER}"]
        act_idx = {t: i for i, t in enumerate(act_task_ids)}

        shared = sorted(baseline_task_ids & set(cond_utils.keys()) & set(act_idx.keys()))
        if len(shared) < 5:
            continue

        b_vals = np.array([baseline_utils[t] for t in shared])
        c_vals = np.array([cond_utils[t] for t in shared])
        aligned_acts = np.array([acts[act_idx[t]] for t in shared])
        p_scores = aligned_acts @ probe_weights[:-1] + probe_weights[-1]

        bu_r, _ = stats.pearsonr(b_vals, c_vals)
        pr_r, _ = stats.pearsonr(p_scores, c_vals)
        baseline_rs.append(bu_r)
        probe_rs.append(pr_r)
        baseline_accs.append(pairwise_accuracy(b_vals, c_vals))
        probe_accs.append(pairwise_accuracy(p_scores, c_vals))

    def mean_se(vals):
        a = np.array(vals)
        return np.mean(a), np.std(a) / np.sqrt(len(a))

    bu_r_mean, bu_r_se = mean_se(baseline_rs)
    pr_r_mean, pr_r_se = mean_se(probe_rs)
    bu_acc_mean, bu_acc_se = mean_se(baseline_accs)
    pr_acc_mean, pr_acc_se = mean_se(probe_accs)

    print(f"Baseline utilities: r = {bu_r_mean:.3f} ± {bu_r_se:.3f}, acc = {bu_acc_mean:.3f} ± {bu_acc_se:.3f}")
    print(f"Probe: r = {pr_r_mean:.3f} ± {pr_r_se:.3f}, acc = {pr_acc_mean:.3f} ± {pr_acc_se:.3f}")
    print(f"N conditions: {len(baseline_rs)}")

    fig, (ax_r, ax_acc) = plt.subplots(1, 2, figsize=(8, 5))
    x = np.array([0, 1])
    colors = ["#B0B0B0", "#6675B0"]

    # Left: Pearson r
    bars_r = ax_r.bar(
        x, [bu_r_mean, pr_r_mean], yerr=[bu_r_se, pr_r_se],
        width=0.5, color=colors, capsize=3, error_kw={"linewidth": 0.8, "alpha": 0.5},
    )
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(["Baseline\nutilities", "Probe"], fontsize=11)
    ax_r.set_ylim(0, 1.0)
    ax_r.set_ylabel("Pearson r", fontsize=11)
    ax_r.set_title("Pearson r", fontsize=13)
    for bar, val in zip(bars_r, [bu_r_mean, pr_r_mean]):
        ax_r.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06,
                  f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Right: Pairwise accuracy
    bars_acc = ax_acc.bar(
        x, [bu_acc_mean, pr_acc_mean], yerr=[bu_acc_se, pr_acc_se],
        width=0.5, color=colors, capsize=3, error_kw={"linewidth": 0.8, "alpha": 0.5},
    )
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(["Baseline\nutilities", "Probe"], fontsize=11)
    ax_acc.set_ylim(0.5, 1.0)
    ax_acc.set_ylabel("Pairwise accuracy", fontsize=11)
    ax_acc.set_title("Pairwise accuracy", fontsize=13)
    ax_acc.axhline(y=0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    for bar, val in zip(bars_acc, [bu_acc_mean, pr_acc_mean]):
        ax_acc.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.suptitle("Simple preference shifts", fontsize=14)
    fig.tight_layout()
    out = ASSETS / "plot_030226_s4_exp1b_probe_vs_baseline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
