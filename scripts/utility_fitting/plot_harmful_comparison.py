"""Plot probe r and baseline utils r with/without harmful tasks for exp1b, 1c, 1d.

Two panels: left = all tasks, right = excluding harmful.
Each panel has grouped bars: baseline utils r and probe r per experiment.
"""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats

BASE = Path(".")
LAYER = 31
RUN_PREFIX = "completion_preference_gemma-3-27b_completion_canonical_seed0"

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


EXPERIMENTS = {
    "Exp 1b\n(Hidden)": {
        "config_dir": BASE / "configs/measurement/active_learning/ood_exp1b",
        "results_dir": BASE / "results/experiments/ood_exp1b/pre_task_active_learning",
        "act_dir_base": BASE / "activations/ood/exp1_prompts",
    },
    "Exp 1c\n(Crossed)": {
        "config_dir": BASE / "configs/measurement/active_learning/ood_exp1c",
        "results_dir": BASE / "results/experiments/ood_exp1c/pre_task_active_learning",
        "act_dir_base": BASE / "activations/ood/exp1_prompts",
    },
    "Exp 1d\n(Competing)": {
        "config_dir": BASE / "configs/measurement/active_learning/ood_exp1d",
        "results_dir": BASE / "results/experiments/ood_exp1d/pre_task_active_learning",
        "act_dir_base": BASE / "activations/ood/exp1_prompts",
    },
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


def map_configs_to_conditions(config_dir: Path) -> dict[str, str]:
    """Map config stem -> sys hash suffix (or empty for baseline)."""
    mapping = {}
    for f in sorted(config_dir.glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text())
        sp = cfg.get("measurement_system_prompt", "")
        if sp:
            h = hashlib.sha256(sp.encode()).hexdigest()[:8]
            mapping[f.stem] = f"_sys{h}"
        else:
            mapping[f.stem] = ""
    return mapping


def compute_per_condition(exp_info: dict, probe_weights: np.ndarray):
    """For one experiment, compute baseline_utils_r and probe_r per condition,
    both with all tasks and excluding harmful."""
    results_dir = exp_info["results_dir"]
    config_dir = exp_info["config_dir"]
    act_dir_base = exp_info["act_dir_base"]

    baseline_dir = results_dir / RUN_PREFIX
    baseline_utils = load_thurstonian_latest(baseline_dir)
    baseline_task_ids = set(baseline_utils.keys())
    harmful_ids = {t for t in baseline_task_ids if "_harmful" in t}

    cond_map = map_configs_to_conditions(config_dir)

    records = []
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

        act_dir = act_dir_base / cond_name
        act_path = act_dir / "activations_prompt_last.npz"
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

        shared_nh = [t for t in shared if t not in harmful_ids]

        for label, task_list in [("all", shared), ("excl_harmful", shared_nh)]:
            if len(task_list) < 5:
                continue
            idx = [shared.index(t) for t in task_list]
            bu_r, _ = stats.pearsonr(b_vals[idx], c_vals[idx])
            pr_r, _ = stats.pearsonr(p_scores[idx], c_vals[idx])
            bu_acc = pairwise_accuracy(b_vals[idx], c_vals[idx])
            pr_acc = pairwise_accuracy(p_scores[idx], c_vals[idx])
            records.append({
                "condition": cond_name,
                "subset": label,
                "baseline_utils_r": bu_r,
                "probe_r": pr_r,
                "baseline_utils_acc": bu_acc,
                "probe_acc": pr_acc,
            })

    return records


def aggregate(records: list[dict], subset: str) -> dict:
    sub = [r for r in records if r["subset"] == subset]
    n = len(sub)
    result = {}
    for key in ["baseline_utils_r", "probe_r", "baseline_utils_acc", "probe_acc"]:
        vals = np.array([r[key] for r in sub])
        result[f"{key}_mean"] = np.mean(vals)
        result[f"{key}_se"] = np.std(vals) / np.sqrt(n)
    return result


def main():
    probe_dir = BASE / "results/probes/gemma3_10k_heldout_std_raw"
    probe_weights = np.load(probe_dir / "probes" / f"probe_ridge_L{LAYER:02d}.npy")

    exp_names = list(EXPERIMENTS.keys())
    all_agg = {}
    excl_agg = {}

    for name, info in EXPERIMENTS.items():
        records = compute_per_condition(info, probe_weights)
        all_agg[name] = aggregate(records, "all")
        excl_agg[name] = aggregate(records, "excl_harmful")

    x = np.arange(len(exp_names))
    width = 0.3
    assets = BASE / "experiments/ood_system_prompts/utility_fitting/assets"

    for suffix, agg, suptitle in [
        ("all", all_agg, "All tasks"),
        ("excl_harmful", excl_agg, "Excluding harmful tasks"),
    ]:
        fig, (ax_r, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Pearson r
        ax_r.bar(
            x - width / 2,
            [agg[n]["baseline_utils_r_mean"] for n in exp_names], width,
            yerr=[agg[n]["baseline_utils_r_se"] for n in exp_names],
            label="Baseline utilities", color="#B0B0B0", capsize=4,
        )
        ax_r.bar(
            x + width / 2,
            [agg[n]["probe_r_mean"] for n in exp_names], width,
            yerr=[agg[n]["probe_r_se"] for n in exp_names],
            label="Probe", color="#6675B0", capsize=4,
        )
        ax_r.set_title("Pearson r", fontsize=13)
        ax_r.set_xticks(x)
        ax_r.set_xticklabels(exp_names, fontsize=11)
        ax_r.set_ylim(0, 1.0)
        ax_r.set_ylabel("Pearson r", fontsize=12)
        ax_r.legend(loc="upper left", fontsize=10)

        # Right: Pairwise accuracy
        ax_acc.bar(
            x - width / 2,
            [agg[n]["baseline_utils_acc_mean"] for n in exp_names], width,
            yerr=[agg[n]["baseline_utils_acc_se"] for n in exp_names],
            label="Baseline utilities", color="#B0B0B0", capsize=4,
        )
        ax_acc.bar(
            x + width / 2,
            [agg[n]["probe_acc_mean"] for n in exp_names], width,
            yerr=[agg[n]["probe_acc_se"] for n in exp_names],
            label="Probe", color="#6675B0", capsize=4,
        )
        ax_acc.set_title("Pairwise accuracy", fontsize=13)
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(exp_names, fontsize=11)
        ax_acc.set_ylim(0.5, 1.0)
        ax_acc.set_ylabel("Pairwise accuracy", fontsize=12)
        ax_acc.legend(loc="upper left", fontsize=10)

        fig.suptitle(suptitle, fontsize=15)
        plt.tight_layout()

        out = assets / f"plot_030226_probe_vs_baseline_{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
        plt.close(fig)

    # Print numbers
    for label, agg in [("ALL TASKS", all_agg), ("EXCL HARMFUL", excl_agg)]:
        print(f"\n{label}:")
        for n in exp_names:
            a = agg[n]
            print(f"  {n.replace(chr(10), ' ')}: "
                  f"bu_r={a['baseline_utils_r_mean']:.3f}±{a['baseline_utils_r_se']:.3f}, "
                  f"probe_r={a['probe_r_mean']:.3f}±{a['probe_r_se']:.3f}, "
                  f"bu_acc={a['baseline_utils_acc_mean']:.3f}±{a['baseline_utils_acc_se']:.3f}, "
                  f"probe_acc={a['probe_acc_mean']:.3f}±{a['probe_acc_se']:.3f}")


if __name__ == "__main__":
    main()
